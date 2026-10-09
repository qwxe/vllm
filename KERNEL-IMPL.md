# MSA: hole-aware attention kernel as built (agent KIMPL, 2026-10-09)

Implements `KERNEL-ALGO.md` section 2 and section 8.2 step 3: one position-explicit work-list kernel over a paged
KV cache. Variants (a) one tile chain per segment, (b) position-packed q tiles and (c) split-KV with an LSE merge
are planner settings. The host planner implements the selection rule. A vLLM 0.19 attention backend `MSA` wraps
it, and adapters serve the CP layer's local-attention interface.

All numbers are measured on the local RTX 4050 Laptop: SM 8.9, 20 SMs, 6 GB, 24 MB L2, 144 GB/s device-to-device
copy. Every GPU job went through `.wt/bin/gpurun`, using the venv `/home/asd/LLMServingSim/.wt/vllm-venv` (torch
2.10, Triton 3.6, vLLM 0.19.0).

## 0. Summary

1. **Correct.**
   - Standalone suite: 249/249 checks pass against a pure-PyTorch fp32 reference. Max error is 9e-3 on O (bf16
     rounding) and 3e-6 on LSE. The suite covers random holes, 1-16 segments, pieces of 1 to 16K tokens, depth to
     128K, GQA 32/8, 8/2, 4/1, 8/1, 2/2 and 14/2, head_dim 64 and 128, fp32 inputs, every planner mode with and
     without forced splitting, packing with straddling pieces, interleaved pass-Q shards, and empty rows.
   - CP adapters: 13/13 checks against the CP branch's own `RefLocalAttention`.
   - End to end on the holed vLLM (holes/core + MSA): the core agent's `gpu_holes_e2e.py` **passes with
     `--backend MSA`** in fp32. Greedy tokens are identical in 8/8 cases. The first-token logprob differs from a
     full recompute by at most 0.011, against stock noise 0.009 and a limit of 0.019. The bf16 compiled run
     also passes (8/8 tokens; dlp ≤ 0.19 against stock noise 0.25).
2. **Split-KV removes the deep-small-piece penalty.**
   - One 16-token piece at 100K on a 1-KV-head shard (Ulysses d = 8, heads 4/1): FA2 takes 3.03 ms, which is
     **20.2x its work fit**. This reproduces KERNEL.md's ~21x. MSA (a) takes 4.06 ms; MSA (c) / auto takes
     0.43 ms, 2.9x the work fit and at the KV-read floor. That is **7.0x faster than FA2 and 9.3x faster than
     (a)**.
   - With all 8 KV heads (32/8), MSA auto runs at about the HBM floor: 2.4 ms at 100K, against 6.0 ms for FA2
     and 3.9 ms for vLLM Triton.
3. **Packing pays where the selection rule says it should.**
   - Eight 20-token pieces within 2K of 50K: (b) 5.8 ms against 11.5 ms for (a), 10.7 ms for vLLM Triton and
     20.9 ms for FA2.
   - Thirty-two 6-token pieces: 14.5 ms against 18.5 ms for (a).
   - Pieces that already fill whole tiles (8 x 64 tokens) gain nothing (see 2.3 on GQA-packed tile height).
4. **No regression on ordinary shapes.**
   - On big single segments (q 2K-16K) and on hole + 2K-tail launches MSA is within ±5 % of FA2 (vLLM's default
     here) and **1.7-1.9x faster than vLLM's Triton backend**, the code it derives from.
   - One shape is worse: 16 tiny pieces spread over 50K (no packing, no split), where MSA is 25 % behind vLLM
     Triton (section 4).

## 1. Code

**Standalone repo `/home/asd/vllm-dev-kernels`** (own git, branch `master`, not pushed):

| file | what |
|---|---|
| `msa/kernel.py` | Triton: `_msa_fwd` (work-list attention), `_msa_merge` (split groups), `_merge_states` (two partial states) |
| `msa/planner.py` | `plan_msa` (segments), `plan_msa_rows` (per-row positions), shared `_finish` (key bounds, items, split, LPT), launch config |
| `msa/api.py` | `msa_attention`, `merge_states`, one-shot `msa(...)` |
| `msa/cp_adapter.py` | `MsaLocalAttention` (CP `LocalAttention` protocol over a `KeySet`), `msa_paged_local` (paged, in place) |
| `msa/reference.py` | pure-PyTorch reference |
| `msa/vllm_backend.py` | the same backend, registrable as `CUSTOM` on a stock vLLM 0.19 |
| `tests/test_correctness.py`, `tests/test_cp_adapter.py` | correctness (standalone, CP adapters) |
| `tests/e2e_vllm.py`, `tests/e2e_compare.py` | Qwen2.5-0.5B on stock vLLM, MSA as CUSTOM vs TRITON_ATTN vs FLASH_ATTN |
| `bench/bench_holes.py`, `bench/tune.py`, `bench/show.py` | microbenchmarks, config sweep |
| `results/` | `correctness.{log,tsv}`, `cp_adapter.log`, `bench_4050.jsonl` (the tables below), `tune.jsonl`, `e2e_*.json`, `e2e_compare.txt`, `raw/` |

**vLLM fork, branch `holes/kernel`** (worktree `/home/asd/vllm-dev-wt/kernel`, 2 commits on v0.19.0). It only adds
attention-backend files:
- `vllm/v1/attention/ops/msa/`: the package above, synced, with SPDX headers;
- `vllm/v1/attention/backends/msa_attn.py`: `MsaAttentionBackend` / `MsaAttentionMetadataBuilder` /
  `MsaAttentionImpl`;
- `vllm/v1/attention/backends/registry.py`: one enum line, `MSA`.

**Integration branch `holes/kernel-int`** (worktree `/home/asd/vllm-dev-wt/kernel-int`): `holes/core` +
`holes/kernel` merged with no conflicts, used for the holed end-to-end runs. It is throwaway, and nobody needs to
merge it.

## 2. Kernel design as built

### 2.1 Work list
- **Item** (int32 x 8): `tile, kv_head, k0, k1, kfull, slot, req`.
- **Tile:** `tile_rows[tile, BQ]` holds row indices into Q, with -1 as padding. Rows of a tile belong to one
  request and may come from several segments, in any order.
- **CTA work.** One CTA per item. It loads the GQA-packed Q tile: M = BQ x g_pad rows of (token, head), the
  layout of vLLM's Triton kernel. It then runs an online softmax (exp2 domain, fp32) over the local keys
  [k0, k1) in BN-key tiles.
- **Unmasked tiles.** Key tiles below `kfull` are visible to every row: no mask and no bounds checks.
- **Masked tiles.** The rest use `mask = kpos <= qpos[row]` and `k < k1`.
- **Key positions** (`KPOS_MODE`), a compile-time switch:
  - 0: the local index is the position (single-GPU holes; contiguous cache);
  - 1: `kblk_pos[req, block] + slot` (strided / interleaved local blocks: pass-Q, PagedKVStore);
  - 2: `kpos[req, local key]` (arbitrary, e.g. a gathered ring buffer).
- **Output.** An unsplit item writes O (the output dtype) and LSE (natural log, -inf for an empty row, with
  O = 0). A split item writes a normalised fp32 partial plus its LSE to `slot`. A separate merge kernel then
  folds each split group in an online LSE merge: one CTA per (tile, KV head), about 10 us.
- **Order and grid.** Items run in LPT order (longest key range first) on an ordinary grid, so the hardware
  scheduler dispatches long chains first. A persistent loop was not needed at this size.
- **KV write.** KV is written before attention by the stock `reshape_and_cache_flash` on `slot_mapping`, which is
  inherited from the Triton backend. Resident pages, pages recomputed in this step and pages recomputed earlier
  are indistinguishable to the kernel (KERNEL-ALGO 2.1).

### 2.2 Planner (`plan_msa`, per step, on the CPU)
1. **Tile height follows the pieces.** BQ = clamp(next_pow2(longest segment), 32 / g_pad, 128 / g_pad) tokens,
   i.e. M from 32 to 128 rows. This generalises rule 3 ("BM 64 when every segment is ≤ 64").
2. **Launch config** (tuned with `bench/tune.py`):

   | tile height | BN | warps | stages | regime |
   |---|---|---|---|---|
   | M ≤ 64 | 128 | 4 | 3 | streaming KV |
   | M = 128 | 64 | 8 | 2 | compute-bound |
   | fp32 inputs | 32 | (as above) | 2 | shared-memory limit |

3. **(a)** One tile chain per segment, cut at BQ from the segment start.
4. **(b)** Per request, rows in position order fill tiles greedily. A piece may straddle two tiles, and segment
   remainders merge with the next piece. A tile closes early when its position spread would exceed
   W_pack = max(BN, eps · deepest position), with eps = 0.05. Whole tiles of a long segment stay aligned. This is
   rule 1: only pieces within W_pack of each other ever share a tile.
5. **Key bounds per tile.** `kend` = the number of local keys with pos ≤ max qpos; `kfull` = keys with pos ≤ min
   qpos. These come from the key layout: contiguous, block starts (searchsorted), or a per-key table.
6. **(c)** Items = tiles x local KV heads. The planner computes L = max(4 · BN, Σ keys / (c · SMs)), rounded up to
   BN, with c = 2, and splits every item longer than L into ⌈len / L⌉ pieces.
   - When items ≥ c · SMs and they are uniform, no item exceeds L, so the launch runs unsplit. That is rule 2 as
     written.
   - The chain cap also splits a lone deep item inside an otherwise full launch, which the count-only rule
     would miss.
7. **Output.** LPT-sorted item tensors are copied to the GPU (non-blocking).
   - Planner CPU cost: 0.7-4 ms per step, from 1 segment to 16 requests x 2 segments to 16K rows. It runs once
     per step, not per layer (vLLM builds metadata once per KV-cache group).
   - It is numpy, and a C++ or vectorised pass would cut it if it ever shows up.

`plan_msa_rows` is the same planner fed with per-row (qpos, qreq), in any order, instead of segments. Rows are
sorted per request. A tile never spans more than BQ positions under (a), or more than W_pack under (b). CP callers
use it.

### 2.3 Why (b) gains less than KERNEL-ALGO 2.6 estimated
KERNEL-ALGO assumed 128-token q tiles. With GQA packing (needed to read each KV head once), a 128-row Triton tile
holds 128 / g = 32 tokens at g = 4, and larger tiles do not fit the register file at d_h = 128. So:
- only pieces shorter than 32 tokens (or segment remainders) can share a tile;
- 8 x 64-token pieces already fill 16 tiles in both (a) and (b);
- KERNEL-ALGO's "8 pieces q = 64 → ~0.15 ms with (b)+(c)" does not materialise.

Two pieces of the gap remain:
- **L2 sharing.** The 16 tiles stream the same prefix concurrently, so L2 serves part of the re-reads. On the
  4050, 8 x 64 runs at 20.9 ms for 1.5 GB of logical KV reads, about 70 GB/s from DRAM.
- **Fix if needed.** Process two q tiles per CTA against one KV stream (2x accumulator registers; needs M = 64
  per tile) or drop GQA packing for packed tiles. Neither is built. The planner's packing already covers the
  fragmented-eviction case, with pieces under 32 tokens.

### 2.4 vLLM backend `MSA` (`--attention-backend MSA`)
- Derived from `TritonAttentionBackend`: the same KV layout `[num_blocks, 2, block_size, Hkv, D]` and the same KV
  write. Only `forward` and the metadata builder change.
- **Segments.**
  - With holes/core, `CommonAttentionMetadata` has one varlen entry per segment, with `seq_lens` = segment end
    and the block-table row repeated. `hole_segments` (`seg_req_idx`, `seg_query_lens`, `seg_ends`) maps
    segments to requests.
  - The builder puts all segments of a request on that request's first row, so they can share packed tiles, and
    sets the request's key count to its deepest segment end.
  - Without `hole_segments`, each request is one segment with position start `seq_len - query_len`. That is
    stock semantics, so MSA also runs ordinary prefill, decode and mixed batches. Decode is flash-decoding-like
    through (c).
- Knobs (env): `MSA_MODE` = a | b | c | bc | auto (default auto), `MSA_EPS`, `MSA_C`.
- `AttentionCGSupport.NEVER`: the work list changes size every step, so attention stays outside CUDA graphs.
  Piecewise graphs still cover the rest of the model.
- Not supported (they raise at init): alibi, sliding window, softcap, sinks, fp8 KV, encoder attention, fused
  output quantization.

## 3. API for CP callers

```python
from vllm.v1.attention.ops.msa import (plan_msa, plan_msa_rows, msa_attention, merge_states,
                                       MsaLocalAttention, msa_paged_local)

# (1) CP LocalAttention protocol (holes/cp, local_attn.py): gathered keys, any order
attend = MsaLocalAttention(mode="auto")             # set_local_attention(attend)
O, LSE = attend(q, qpos, qreq, keyset, scale)       # O fp32 [rows,Hq,D], LSE fp32 [rows,Hq], -inf = empty

# (2) Paged, in place (no gather): pass-Q on interleaved blocks, PagedKVStore
O, LSE = msa_paged_local(q, qpos, qreq, k_cache, v_cache, block_table, kblk_pos, kv_lens, scale)
#   qreq indexes block_table rows; kblk_pos[r, j] = absolute start of local block j (int32), None = contiguous;
#   kv_lens[r] = local keys of row r.

# (3) Plan once per step, launch per layer (cheapest in a model)
plan = plan_msa_rows(qpos_cpu, qreq_cpu, kv_lens, Hkv_local, Hq_local // Hkv_local,
                     block_size=bs, kblk_pos=kblk_pos_cpu, device=dev)      # or plan_msa(segments, ...)
O, LSE = msa_attention(q, k_cache, v_cache, block_table, plan, kblk_pos=kblk_pos_gpu,
                       softmax_scale=scale, out=o_buf, lse=lse_buf)          # out dtype = what you pass

# (4) Ring step / partial reduction
O, LSE = merge_states(O_a, LSE_a, O_b, LSE_b)       # natural-log LSE, -inf safe
```

- **Mask.** `qreq == req(key)` and `kpos <= qpos`. A row with no visible key gets O = 0 and LSE = -inf, the
  identity of the merge.
- **Precision.** O comes back in the caller's `out` dtype: bf16 for the wire (decision 4.D), fp32 by default in
  the adapters. LSE is fp32.
- **Paged path invariant.** Every local slot below a row's position must hold valid KV. That holds under
  front-first hole order: slots of holes not yet computed always lie above every query of the step. Mode 2 can
  mask an absent slot by giving it position 2^30.
- **Planning syncs.** `MsaLocalAttention` gathers into a scratch paged buffer and reads `qpos` / `kpos` on the
  CPU (one sync). `plan_msa_rows` also needs `qpos` / `qreq` on the CPU. CP code that already knows its row
  layout on the host should pass CPU arrays and reuse one plan for all layers.

## 4. Benchmarks (RTX 4050 Laptop, one layer, bf16, block 16, head_dim 128)

**Method.**
- Baselines get variant (a) metadata, exactly what vLLM launches with holes/core: one varlen entry per segment,
  `seqused_k` = segment end, and the block table repeated.
- FA2 is `vllm_flash_attn` with `fa_version=2`; vLLM Triton is `unified_attention`.
- L2 is flushed before each call. Each value is the median of 21 round-robin rounds. Run-to-run spread is ±5-10 %:
  this is a laptop GPU that also drives the display.
- FlashInfer **could not run**: its prefill kernels are JIT-only and the machine has no `nvcc`.
- **Work fit** = causal FLOPs / 21.8 TFLOP/s, the large-q rate measured at q 16K.
- **KV floor** = the bytes of [0, deepest end) for all KV heads, at the 144 GB/s copy rate. Pure reads reach about
  170 GB/s, so a value below the floor is possible.
- Decode (FA2, one token, 32/8) as a reference: 0.48 / 1.83 / 2.71 ms at 16K / 64K / 100K.

Times in us:

| shape | heads | FA2 | vLLM Triton | MSA (a) | MSA (b) | MSA (c) | **MSA auto** | auto vs FA2 | auto vs (a) | KV floor | work fit | FA2 / fit | auto / fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q16 @ 4K | 32/8 | 335 | 230 | 252 | 245 | 172 | **169** | 1.98x | 1.49x | 117 | 49 | 6.8 | 3.4 |
| q16 @ 16K | 32/8 | 1289 | 823 | 898 | 900 | 519 | **507** | 2.54x | 1.77x | 467 | 197 | 6.5 | 2.6 |
| q16 @ 32K | 32/8 | 2111 | 1357 | 1523 | 1541 | 902 | **882** | 2.39x | 1.73x | 934 | 394 | 5.4 | 2.2 |
| q16 @ 64K | 32/8 | 3898 | 2525 | 2868 | 2864 | 1625 | **1625** | 2.40x | 1.76x | 1868 | 789 | 4.9 | 2.1 |
| q16 @ 100K | 32/8 | 6006 | 3869 | 4336 | 4278 | 2394 | **2395** | 2.51x | 1.81x | 2850 | 1203 | 5.0 | 2.0 |
| q64 @ 64K | 32/8 | 3991 | 5856 | 3646 | 3604 | 3243 | **3127** | 1.28x | 1.17x | 1868 | 3153 | 1.3 | 1.0 |
| q256 @ 64K | 32/8 | 14006 | 21414 | 14927 | 15330 | 15022 | **14727** | 0.95x | 1.01x | 1868 | 12595 | 1.1 | 1.2 |
| q2048 @ 16K | 32/8 | 28811 | 46611 | 28811 | 28075 | 27473 | **27252** | 1.06x | 1.06x | 525 | 26818 | 1.1 | 1.0 |
| q8192 @ 0 | 32/8 | 26276 | 42655 | 26713 | 26151 | 24906 | **24547** | 1.07x | 1.09x | 234 | 25242 | 1.0 | 1.0 |
| q16384 @ 16K | 32/8 | 305971 | 552933 | 303955 | 296490 | 298086 | **302878** | 1.01x | 1.00x | 934 | 302878 | 1.0 | 1.0 |
| 8 x q20 within 2K of 50K | 32/8 | 20918 | 10654 | 11512 | 5834 | 12074 | **6044** | 3.46x | 1.90x | 1476 | 6124 | 3.4 | 1.0 |
| 8 x q64 within 2K of 50K | 32/8 | 21047 | 34176 | 22449 | 21814 | 21282 | **20858** | 1.01x | 1.08x | 1477 | 19606 | 1.1 | 1.1 |
| 32 x q6 within 3K of 50K | 32/8 | 81792 | 16473 | 18504 | 14725 | 19153 | **14538** | 5.63x | 1.27x | 1476 | 7279 | 11.2 | 2.0 |
| 16 x q8 spread 0..50K | 32/8 | 19519 | **4450** | 5538 | 5542 | 5854 | 5542 | 3.52x | 1.00x | 1286 | 2176 | 9.0 | 2.5 |
| hole q16 @ 30K + tail 2048 @ 100K | 32/8 | 160643 | 291272 | 155486 | 149379 | 149520 | **150261** | 1.07x | 1.03x | 2850 | 152833 | 1.1 | 1.0 |
| hole 4K @ 8K + tail 2048 @ 40K | 32/8 | 93523 | 162934 | 91566 | 88688 | 88419 | **89484** | 1.05x | 1.02x | 1140 | 91593 | 1.0 | 1.0 |
| q16 @ 100K, Ulysses d=4 shard | 8/2 | 3033 | 3355 | 4103 | 4119 | 756 | **726** | 4.18x | 5.65x | 713 | 301 | 10.1 | 2.4 |
| tail 2048 @ 64K, d=4 shard | 8/2 | 25356 | 43294 | 27604 | 26365 | 25435 | **25090** | 1.01x | 1.10x | 467 | 24845 | 1.0 | 1.0 |
| q16 @ 100K, Ulysses d=8 shard | 4/1 | 3032 | 3333 | 4056 | 4042 | 444 | **434** | **6.98x** | **9.34x** | 356 | 150 | **20.2** | 2.9 |
| tail 2048 @ 64K, d=8 shard | 4/1 | 13707 | 21023 | 14673 | 14799 | 14736 | **14634** | 0.94x | 1.00x | 234 | 12423 | 1.1 | 1.2 |

### Reading the table

**The 21x effect.**
- A deep small piece is slow when too few CTAs stream its prefix. Each CTA reads serially at about 12-13 GB/s
  here (MSA (a) on the 4/1 shard: 51 MB in 4.06 ms). FA2's figure on the RTX PRO 6000 was about 22 GB/s.
- On the 4050 with all 8 KV heads, FA2's 32 CTAs (one per q head) already cover the 20 SMs. So the penalty is
  "only" 5x the work fit, or 2.2x a decode of the same depth.
- On a 1-KV-head shard it is 20x the work fit, KERNEL.md's number.
- Split-KV brings every q16 shape to the KV floor: 2.0-3.4x the work fit, which for this shape is the
  bandwidth-bound minimum.
- vLLM's Triton kernel (BLOCK_M = 16, i.e. 4 tokens x 4 heads per CTA) is better than FA2 here because it
  launches 4x more CTAs. It is still 1.6x behind split-KV.

**Packing.** 8 x q20 drops from 11.5 to 5.8 ms (5 packed tiles instead of 8). 32 x q6 drops from 18.5 to
14.5 ms. Pieces that are whole tiles (8 x q64) do not change.

**Big segments.**
- The MSA tile config ties FA2: 21.8 vs 21.4 TFLOP/s at q 16K, -1 to +7 % across the q ≥ 2K rows.
- MSA is 1.7-1.9x faster than vLLM Triton, whose 16-row tiles re-read KV 8x more often.
- The two 0.94-0.95x rows (q256 @ 64K, the 4/1 tail) are within noise to 6 % slower: 64 items on 20 SMs leave a
  tail.

**Hole + tail.** One launch is dominated by the deep tail, so all variants tie. The simulator's "sum of
per-segment lookups" over-charges this shape (KERNEL-ALGO 2.2). The real cost is the tail's.

**Where MSA loses.** 16 x q8 spread over 50K is 1.25x behind vLLM Triton.
- The pieces are too far apart to pack, and with 128 items ≥ 2 x SMs the planner does not split.
- vLLM Triton wins by running many small CTAs that walk the shared prefix in lockstep, so L2 catches the
  re-reads.
- Possible fix: lower c, or split when the tiles' prefixes overlap. Not tuned, since the workload rarely
  produces this shape (pieces of 256 tokens or fewer are 0.3-0.8 % of work, KERNEL.md).

## 5. Tests

| suite | command | result |
|---|---|---|
| standalone correctness | `gpurun python tests/test_correctness.py` | 249/249 PASS (`results/correctness.tsv`) |
| CP adapters vs holes/cp `RefLocalAttention` | `gpurun python tests/test_cp_adapter.py` | 13/13 PASS (`results/cp_adapter.log`) |
| stock vLLM 0.19, MSA as CUSTOM, Qwen2.5-0.5B bf16 (chunked prefill 512, prefix-cache hits, decode) | `gpurun python tests/e2e_vllm.py CUSTOM` | correct text. Greedy tokens match TRITON_ATTN on 5/7 prompts. The other two are near-ties, where FA2 and Triton also disagree with each other. First-token logprob deltas: MSA-Triton up to 0.25, FA2-Triton up to 0.30 (`results/e2e_compare.txt`) |
| holed vLLM (holes/core + MSA), fp32 eager | `gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/kernel-int python tests/v1/holes/gpu_holes_e2e.py --eager --dtype float32 --gpu-util 0.75 --backend MSA` | **PASS**: 8/8 cases token-identical to a full recompute. Holed dlp 0.001-0.011, stock noise 0.009, limit 0.019. Only hole + tail tokens were computed (e.g. 1032 of 3000) |
| holed vLLM, bf16, torch.compile + piecewise graphs (vLLM drops FULL graphs for MSA, as declared) | same without `--eager --dtype float32`, `--gpu-util 0.7` | **PASS**: 8/8 token-identical to full recompute; holed dlp 0-0.19 vs stock noise 0.25 (limit 0.50; bf16 logprobs move in 0.0625 steps) |

Correctness tolerance: |O - ref| < 1.5e-2 (bf16 output) and |LSE - ref| < 2e-3. Empty rows must come back with
LSE = -inf.

## 6. What changes on bigger GPUs (RTX PRO 6000 Blackwell, 188 SMs, ~1.6 TB/s; H100 132 SMs)

1. **Split engages more often, which is the point.** The split threshold is c · SMs = 376 items on the PRO 6000,
   against 40 here.
   - A 2048-token tail at 8 KV heads has 512 items: no split.
   - A 1024-token tail, any piece under about 1.5K tokens alone, and every Ulysses rank with 1-2 KV heads are
     split.
   - Projection from the per-CTA chain (est.): q16 @ 100K at 32/8 goes from about 2.3 ms (FA2, measured in
     KERNEL.md) to about 0.28 ms. That matches KERNEL-ALGO's estimate, because the 4050's split already reaches
     the KV floor.
2. **Tile configs must be re-tuned** with `bench/tune.py` on the target, and `_launch_config` updated.
   - SM120 has a different shared-memory size and more warps per SM.
   - BN 128 with 3 stages for short tiles may change.
   - KERNEL-ALGO estimated Triton at 70-90 % of FA2 there. On SM 8.9 it ties FA2 on big shapes, but that has to
     be re-measured on SM120 before trusting it.
3. **Packing helps more** when SMs are plentiful and HBM is the limit. The 2.3 caveat (32-token tiles) does not
   change.
4. **The planner's CPU cost is per step** (0.7-4 ms here). On a fast GPU a 2048-token step at 64K is about 12 ms
   per layer x 32 layers, so it stays below 1 %. For decode-heavy steps it should be vectorised further, or
   overlapped with the previous step (vLLM's async scheduling already overlaps metadata building).
5. **FlashInfer** runs there (nvcc available). Profile it next to FA2 and MSA as KERNEL-ALGO 7.2 plans.

## 7. Not done / next
- FlashInfer comparison (needs nvcc). The profiler's `segmented_attention` category and the `MSA:{a,c,bc}` rows
  (KERNEL-ALGO 7.1/7.2) belong to the profiler agent. The kernel and backend are ready for `--attention-backend
  MSA` with `MSA_MODE`.
- Two q tiles per CTA (or non-GQA-packed tiles) to make (b) pay for 32-128-token pieces (2.3).
- The 16 x q8 spread case: an L2-aware split rule (section 4).
- fp8 KV, sliding window, softcap, sinks; CUDA-graph support (it needs a fixed-size, padded work list).
- The CP branch has not switched to `MsaLocalAttention` / `msa_paged_local` yet. The adapters are tested against
  its reference.
