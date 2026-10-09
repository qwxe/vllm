# CP for holed prefills in vLLM 0.19: implementation notes (branch `holes/cp`)

Implements KERNEL-ALGO.md section 4 (and 8.2 steps 4-5): Ulysses, pass-Q, ring pass-KV, all-gather KV, and the
per-request ring / pass-Q choice, for prefill steps whose computed tokens are arbitrary segments (hole pieces +
tail). The communication and partition layer is tested over gloo at d = 1/2/4/8. The local attention is the
hole-aware MSA kernel from `holes/kernel` on CUDA (in place on the paged cache for Ulysses / pass-Q), with
PyTorch implementations for CPU tests. The backend runs inside real vLLM (one GPU, CP world size 1) on holed
prefills from `holes/core`.

**Where the code is.** Branch `holes/cp`, worktree `/home/asd/vllm-dev-wt/cp` (on `holes/base` = v0.19.0).
- The first 10 commits were written in a staging repo (`/home/asd/LLMServingSim/.wt/cp-staging`) while the fork
  was being set up, and moved here with `format-patch` + `git am`. They add new files only.
- `holes/core` is **merged** into `holes/cp` (merge commit "Merge holes/core ..."): the backend reads its
  `CommonAttentionMetadata.hole_segments`.
- `holes/kernel` is **merged** (merge commit "Merge holes/kernel ..."): MSA is the CP local attention.
- The branch also patches vLLM core in a few places (section 3.1): the `--cp-kv-placement` switch, the
  sharded-dense model-runner path, the CP step budgets in holes/core's planner, and the missing-KV ranges from
  the scheduler.

| file | what |
|---|---|
| `vllm/v1/attention/cp_holes/local_attn.py` | `KeySet`, `LocalAttention` protocol, `RefLocalAttention`, `TiledLocalAttention`, `merge_attn_states` (vLLM's DCP Triton combine on CUDA, PyTorch otherwise) |
| `vllm/v1/attention/cp_holes/plan.py` | `StepPlan` (global step description), `home_rank` (interleave u), `rows_contiguous` (micro-batched), `rows_c2e` (COBAL c2e, cuts on u) |
| `vllm/v1/attention/cp_holes/comm.py` | byte-packed variable-size all-to-all, all-gather, ring send/recv (`batch_isend_irecv`) |
| `vllm/v1/attention/cp_holes/kv_store.py` | `RefKVStore`, `PagedKVStore` (vLLM paged cache), `vllm_local_layout` / `vllm_slot_mapping` (vLLM's PCP placement) |
| `vllm/v1/attention/cp_holes/schemes.py` | `UlyssesCP`, `PassQCP`, `RingCP`, `AllGatherKVCP`, `AdaptiveCP`, `make_scheme` |
| `vllm/v1/attention/backends/cp_holes.py` | `CPHolesBackend` / builder / impl, `register()` (as `AttentionBackendEnum.CUSTOM`) |
| `tests/v1/attention/cp_holes/test_cp_gloo.py` | scheme math vs single-device attention, gloo, world 1/2/4/8 |
| `tests/v1/attention/cp_holes/test_backend_gloo.py` | the backend glue (builder + impl) on vLLM's paged layout, with holes/core's `hole_segments` and without holes, replicated and sharded dense, gloo, world 2/4 |
| `tests/v1/attention/cp_holes/test_kv_placement.py` | `--cp-kv-placement`: capacity, KV spec heads, vLLM's slot-mapping kernel under a faked PCP group (GPU part) |
| `tests/v1/attention/cp_holes/test_msa_paged_gpu.py` | MSA as the local attention, gathered and paged in place, vs the reference (GPU) |
| `tests/v1/attention/cp_holes/gpu_cp_holes_e2e.py`, `gpu_cp_holes_vs_stock.py` | real vLLM with CP_HOLES on one GPU: holes/core's holed e2e check; greedy output vs a stock backend |
| `tests/v1/holes/test_holes_cp_budget.py`, `test_holes_missing_kv.py` | CP step budgets in the planner; missing-KV ranges scheduler -> runner -> slot validity |
| `benchmarks/cp_holes/bench_cp_holes.py`, `run_server.sh` | NCCL correctness check and per-scheme step time (torchrun); `--local-attn msa --store paged` = the backend's path |

## 1. Common design

- **Step plan.** `StepPlan(segments, d, u)` is built from scheduler metadata that every CP rank receives
  identically. Per request it holds the computed segments; E_step(i) is the request's largest computed position.
  Everything in [0, E_step) that is not computed this step is resident; nothing past E_step is read (causal
  truncation, decision 4.F). From the plan every rank derives the row ownership, every collective's split sizes and
  every key position. So no metadata travels and no size exchange happens: only Q / K / V / O / LSE payloads.
- **Interface per layer.** `scheme.own_rows` lists the rows (global indices into the step's computed list) that
  this rank runs the dense layers on. `scheme.forward(q, k, v, store, scale)` takes `qkv_proj` + rope outputs for
  those rows and returns O for them. `store` is the rank's KV of the layer: `write(KeySet)` and
  `gather(reqs, limits)`. The scheme writes the step's new K/V wherever its placement homes them. That is the cache
  update, so the backend sets `forward_includes_kv_cache_update = True`.
- **Local primitive.** `attend(Q, qpos, qreq, KeySet(k, v, kpos, kreq)) -> (O fp32, LSE fp32)`, with mask
  `qreq == kreq and kpos <= qpos` (KERNEL-ALGO 2.1). A row with no key gives O = 0, LSE = -inf. The pass-Q
  per-row limit and the ring's buffer positions are this mask, so no scheme needs a special kernel.
  - Selection (`CP_HOLES_LOCAL_ATTN`): `auto` (default: MSA `MsaLocalAttention` for CUDA tensors, the tiled
    PyTorch version on CPU), `msa`, `tiled`, `ref`, or `module:attr`. `set_local_attention(fn)` or the
    `local_attn=` argument override it.
  - **In place.** Ulysses and pass-Q attend after the step's keys are written, so they call
    `store.attend(...)`. `PagedKVStore.attend` runs MSA's paged mode on the vLLM cache with per-block start
    positions (`kblk_pos`, strided under interleaving). It plans once per step and micro-batch
    (`plan_msa_rows`), and every layer reuses that plan (`CPHolesMetadata.step_cache`).
  - Ring buffers (network buffers) and all-gather's cached-key pass (run before the new keys are written) use
    the gathered `KeySet` path through `MsaLocalAttention`.
- **Wire.** Q / K / V / O travel in the model dtype (bf16), LSE in fp32 (decision 4.D). Each collective packs its
  tensors row-wise into one uint8 buffer, so one collective carries mixed dtypes. Gloo and NCCL move the same
  bytes.
- **Placement** for position-sharded schemes: position p lives on rank (p // u) % d. This is exactly vLLM's PCP
  placement when `block_size == cp_kv_cache_interleave_size == u` (checked against a port of
  `_compute_slot_mapping_kernel`: `vllm_slot_mapping`, test `make_paged_store`). u = 128 therefore means
  `--block-size 128 --cp-kv-cache-interleave-size 128`. A smaller block size with u = 128 would need a slot
  mapping change in vLLM.

## 2. Per scheme, as built

### 2.1 Ulysses (`UlyssesCP`)
- **Rows.** `n_mb` micro-batches (default 2) over the computed list, each cut into d contiguous pieces. Rank r
  owns piece r of every micro-batch. Within a request, micro-batch k precedes k+1 in position, so micro-batch k's
  attention needs only keys of micro-batches <= k.
- **Per micro-batch:**
  1. a2a#1 sends each rank its head slice of Q, K, V for the own rows;
  2. the rank writes the micro-batch's K/V (its KV heads, all rows);
  3. it attends all rows of the micro-batch at Hq/d heads over its resident head slice (truncated at E_step)
     plus the new keys;
  4. a2a#2 returns O to the row owners, which concatenate the head slices.
- **Overlap.** a2a#1 of k+1 and a2a#2 of k are async (`async_op=True`) and in flight during micro-batch k's /
  k+1's attention.
- **d > N_KV (U-R, decision 4.A).** Rank r uses KV head r·N_KV/d, so each KV head is replicated on d/N_KV ranks.
  a2a#1 sends the new K/V to every replica. Requires d | Hq and (d | Hkv or Hkv | d).
- **Split-KV.** At N_H/d heads (one KV head at d = 8) the kernel must split the key range. That belongs to the
  kernel (`holes/kernel`, variant (c)). The reference has no occupancy notion.
- **Bytes per rank per layer.** (d-1)·n_own·(Q+KV)/d + (m-n_own)·O/d.

### 2.2 Pass-Q (`PassQCP`)
- **Rows.** Contiguous, micro-batched like Ulysses (`n_mb`, default 2).
- **Per micro-batch, a2a#1** is one all-to-all whose per-destination payload is [Q of all own rows] (the
  all-gather of Q) + [K/V of the own rows homed on that destination]. Each new K/V row goes only to its home rank.
- The rank writes the K/V it homes, then attends all rows of the micro-batch over its local keys (strided
  positions; the causal mask is the per-row limit).
- **a2a#2** returns (O bf16, LSE fp32) rows to their owners. The owner merges the d partials by LSE.
- This is vLLM DCP's `dcp_comm_backend="a2a"` path (`dcp_a2a_lse_reduce` + `dcp_lse_combine_triton`), with
  partials scattered by **rows** to their owners instead of by heads. It moves the same bytes as a merging
  reduce-scatter and needs no LSE all-gather first (vLLM's `cp_lse_ag_out_rs` needs one).
- **Overlap.** As Ulysses: a2a#2 of micro-batch k during the attention of k+1.
- **Bytes per rank per layer.** (d-1)·n_own·Q + ~(d-1)/d·n_own·KV + (m-n_own)·(O+LSE).

### 2.3 Ring pass-KV (`RingCP`)
- **Rows.** Per-request c2e (`rows_c2e`: equal tokens and equal attention, at most 2 pieces per rank per
  request, cuts on u, ported from `cobal.py::p_two_ended`, kernel cost = causal pairs per u-unit).
- **Buffer.** Rank r's initial buffer = its homed resident K/V of the step's requests, truncated at E_step, + the
  new K/V of its own rows (homed anywhere). Its positions come from the plan.
- **d steps.** Each step:
  - posts `batch_isend_irecv` (send to r+1, receive from r-1; double-buffered);
  - attends the own rows over the current buffer (partial O, LSE);
  - writes the buffer's new K/V that is homed on r into the cache;
  - waits for the exchange.

  At the end it merges the d partials.
- **Causal truncation.** Only [0, E_step) of each request circulates, so a shallow hole step moves less.
- **Bytes per rank per layer.** Σ over the d-1 received buffers ≈ (d-1)/d·(E_step + m)·KV.

### 2.4 All-gather KV (`AllGatherKVCP`, AG-R)
- **Rows.** c2e. Resident K/V is replicated on every rank.
- **Flow.**
  1. Start an async all-gather of the new K/V only.
  2. Meanwhile attend the own rows over the cached keys (gathered before any write).
  3. Wait, write all new K/V (replicated), attend over the new keys.
  4. LSE-merge the two partials (the cascade-attention shape).
- **Bytes per rank per layer.** (d-1)·n_own·KV. AG-S (sharded resident + prefetch) is not built.

### 2.5 Adaptive ring / pass-Q (`AdaptiveCP`)
- Per request-step: pass-Q when computed / E_step < `threshold` (default 0.2, KERNEL-ALGO 4.6), ring otherwise.
- Both run in the same layer over the shared interleaved placement, so the choice moves no KV. Own rows = the
  union of the two schemes' rows.
- Meta's Alg. 5 with a calibrated bandwidth can replace the fixed ratio once 7.4 gives `link_bw`.

## 3. vLLM plug-in points (names verified in the installed 0.19.0 source)

- **CP group.** `vllm.distributed.get_pcp_group()` (`--prefill-context-parallel-size`, `ParallelConfig.
  prefill_context_parallel_size`). It is a separate group, not the TP ranks (DCP reuses TP ranks).
  `CPComm.from_coordinator(gc)` uses `gc.device_group`.
- **Backend.** `vllm/v1/attention/backends/cp_holes.py`, registered through `register_backend(
  AttentionBackendEnum.CUSTOM, ...)` (`backends/registry.py`). It needs no edit to vLLM files.
  - `CPHolesImpl.supports_pcp = True`: `v1/worker/cp_utils.py::check_attention_cp_compatibility` asserts it.
    **No GPU backend in 0.19 sets it**: PCP exists in config, groups, KV manager and slot mapping, but no GPU
    attention uses it.
  - The builder builds the `StepPlan` and the scheme once per step and caches the device index tensors, so
    every layer reuses them.
  - The impl wraps the layer's `kv_cache` in a `PagedKVStore` (layout `[2, num_blocks, block_size, Hkv, D]`,
    as FlashAttention's) and runs the scheme.
- **KV placement: `--cp-kv-placement interleave | head | replicate`** (new `ParallelConfig.cp_kv_placement`,
  part of the config hash).
  - `interleave` (default) is vLLM's PCP layout: `single_type_kv_cache_manager.py` multiplies the block size by
    pcp·dcp, and `_compute_slot_mapping_kernel` interleaves by `cp_kv_cache_interleave_size`. pass-Q / ring /
    adaptive use it unchanged (`vllm_local_layout` is its inverse).
  - `head` (Ulysses) and `replicate` (all-gather KV) put every position on every PCP rank.
    `ParallelConfig.kv_interleave_pcp_size` is then 1: the scheduler's KV manager (`pcp_world_size`), the
    capacity (`max_memory_usage_bytes`, `kv_cache_utils`, `max_num_kv_tokens`), `BlockTable`'s slot mapping and
    `get_total_cp_world_size` all see no interleave (worker side via `cp_utils.set_cp_kv_placement`, set by the
    model runner).
  - `head` also builds the KV cache spec with `cp_kv_heads_per_rank(Hkv, pcp)` heads (Hkv/pcp, or 1 when
    pcp > Hkv) in `Attention.get_kv_cache_spec`.
  - The builder asserts that the scheme matches the placement (`REQUIRED_PLACEMENT`).
- **Hole ranges (holes/core).** When a step has holes, holes/core's model runner hands the backend
  segment-level metadata: one entry per segment (`query_start_loc` / `seq_lens` per segment, the request's
  block-table row repeated) plus `common_attn_metadata.hole_segments` (`HoleSegments`: `seg_req_idx`,
  `seg_query_lens`, `seg_ends`, `req_hole_tokens`).
  - `request_segments(cam)` groups the segments back into requests: per request `(seg_end - q, q)` pairs, the
    block-table row of the request's first segment, and E = max segment end.
  - holes/core lays tokens out per request as hole pieces in position order, then the tail. That is exactly
    the `StepPlan` computed-list order, so plan row indices are the model's token indices and `own_rows`
    indexes the model's token tensors directly.
  - Without holes (`hole_segments is None`) every entry is a request with the stock segment
    `[seq_len - q_len, seq_len)`.
- **KV-slot validity from the hit mask.** At the end of `schedule()`, the scheduler reports per request the
  ranges whose KV is still missing after the step (`SchedulerOutput.missing_kv_ranges`, i.e. the pending hole
  ranges: the hit-mask misses not computed yet). The model runner carries them in
  `HoleSegments.req_missing_ranges`. `kv_slot_validity` then marks a slot valid when its position is below the
  request's end, not computed this step, and not missing. A missing range below E_step would be read by a query,
  so it raises. Under front-first order every missing range lies past E_step.
- **Dense-layer modes** (`VLLM_CP_HOLES_DENSE`):
  - `replicated` (default): every PCP rank runs the dense layers on all tokens. Attention runs the scheme on
    `own_rows` and all-gathers O (`gather_rows`). Logits are identical on all ranks. This is the correctness
    reference; it costs d x dense compute and an extra all-gather of O.
  - `sharded` (`gpu_model_runner.execute_model` patch):
    - After `_preprocess`, `cp_holes_sharded_scheme(attn_metadata)` returns the step's scheme. The runner then
      keeps only `own_rows` of `input_ids` / `positions` / `inputs_embeds` (`shard_inputs`) and runs the model
      eagerly on them (`num_tokens = n_own`, no CUDA graph). The attention metadata still describes the whole
      step.
    - The hidden states of the logits rows (each request's tail-last token) come from whichever rank owns
      them: `gather_logits_hidden`, one all-reduce of num_logits x hidden over the CP group, zeros elsewhere.
    - Every rank then computes logits and samples exactly as without CP. That keeps the per-rank request state
      identical (as under TP), and PCP rank 0 returns the output.
    - This deliberately deviates from "logits only on the owner": owner-only logits plus a broadcast of the
      sampled tokens would also need logprobs and sampler state broadcast, for no saving beyond one
      lm_head on num_reqs rows.
    - Spec decode, pooling and PP are refused in this mode.
- **Merge.** On CUDA it uses `vllm.v1.attention.ops.dcp_alltoall.dcp_lse_combine_triton` (handles -inf
  partials; head_dim a power of 2), otherwise PyTorch. `cp_lse_ag_out_rs` is at `v1/attention/ops/common.py`
  (not `vllm/attention/ops`, as the doc assumed). `merge_attn_states` is at
  `v1/attention/ops/merge_attn_states.py`.
- **Arctic Inference** is not installed in the venv and was not vendored. Ulysses follows its design
  (all-to-all around attention, TP-like head-sliced KV, KV-head replication for SP > KV heads), reimplemented in
  `UlyssesCP`.

### 3.1 Core patches on this branch (beyond the backend files)
1. `--cp-kv-placement` (above): `config/parallel.py`, `engine/arg_utils.py`, `v1/kv_cache_interface.py`,
   `v1/core/kv_cache_utils.py`, `v1/core/sched/scheduler.py`, `v1/worker/{cp_utils,block_table,gpu_model_runner}.py`,
   `model_executor/layers/attention/attention.py`.
2. Sharded dense mode: `v1/worker/gpu_model_runner.py` (execute_model).
3. **CP step budgets in holes/core's planner** (KERNEL-ALGO 5.3). New holes config keys:
   - `cp_scheme` (`ulysses|passq|allgather|ring|adaptive`), `cp_size` (0 = PCP size), `cp_rank_tokens` (2048),
     `cp_attn_tflops` (280), `cp_link_gbps` (16), `cp_passq_ratio` (0.2), set via `additional_config["holes"]`
     or `VLLM_HOLES`.
   - `cp_step_budget`:
     - Ulysses / pass-Q / all-gather: d·cp_rank_tokens;
     - ring: d·max(cp_rank_tokens, C_ring), with C_ring = F_attn/(4·BW) rounded up to the 128 granule (4480 at
       280 TFLOP/s and 16 GB/s);
     - adaptive: the ring's budget when the request's remaining work / context ≥ ratio, else pass-Q's.
   - The scheduler's `_plan_hole_step` caps a holed request's step with it. `step_budget`,
     `long_prefill_token_threshold` and `max_num_batched_tokens` still apply, so set `max_num_batched_tokens`
     ≥ the CP budget.
   - The planner covers holed requests only. Non-holed prefills keep vLLM's stock chunking.
4. Missing-KV ranges: `v1/core/sched/{output,scheduler}.py`, `v1/worker/hole_segments.py`, the runner.

Still not supported: CUDA graphs (eager only; host-side plan per step, fine for prefill-only P/D instances), fp8
KV.

## 4. Tests run here (all pass; final rerun on the merged branch)

| test | what it checks | result |
|---|---|---|
| `test_cp_gloo.py --world 1 2 4 8` (gloo, CPU) | every scheme vs single-device attention over the full context. 7 random holed requests per step (several holes, sub-tile pieces, tail-only, fresh, hole-only steps, keys past E_step); Hq 8, Hkv 2/4/8 (Ulysses d > Hkv); fp32 / bf16; u = 16 / 128; 1-2 micro-batches; adaptive with both sub-schemes; a vLLM-layout paged store; bit-exact cache writes; all rows covered once | 160 / 160 OK; fp32 max error 8.3e-7, bf16 1.05e-2 |
| `test_backend_gloo.py --world 2 4` | builder + impl on a paged cache filled at vLLM's slots, vs reference: holes/core `hole_segments` and stock metadata (replicated dense), and the sharded mode (own rows only + the logits-row gather); cache writes at vLLM's slots | 30 / 30 OK, max error 6.6e-7 |
| `test_kv_placement.py` (CPU + GPU) | `kv_interleave_pcp_size`, per-rank capacity, config hash, KV spec heads under `head`; vLLM's real slot-mapping Triton kernel under a faked PCP group (2 / 4 ranks, every rank): `interleave` = `vllm_slot_mapping`, `head` / `replicate` = the plain layout, all slots local | OK |
| `tests/v1/holes/test_holes_cp_budget.py` | budget values (all schemes, fast link, adaptive); scheduler with a 4800-token hole: pass-Q / Ulysses d = 2 -> 4096 + 704 (+1-token tail), ring -> one step, capped by max_num_batched_tokens | OK |
| `tests/v1/holes/test_holes_missing_kv.py` | scheduler reports [(2304, 5056)], [(4352, 5056)], none over a chunked hole; `build_hole_segments` carries them; slot validity; a missing range below E_step raises | OK |
| holes/core's own CPU tests after the merges | `test_holes_plan` 13, `test_holes_kv` 10, `test_holes_segments` 3 | all PASS |
| `test_msa_paged_gpu.py` (GPU) | `AutoLocalAttention` (MSA) vs reference on random holed key sets; `PagedKVStore.attend` in place (MSA paged, strided `kblk_pos` for rank 1 of 4 / rank 0 of 2, plain layout) vs gather + reference, with the plan reused | fp32 3.3e-3 (TF32 dots), bf16 2.9e-3 |
| `bench_cp_holes.py --check` (GPU, world 1) | 7 layouts x 5 schemes, bf16; tiled / MSA gathered / MSA paged in place | CHECK PASS each (max error 7.8e-3) |
| **real vLLM**, `gpu_cp_holes_vs_stock.py`, Qwen2.5-0.5B, fp32, eager | greedy 24 tokens + first-token top-5 logprobs vs TRITON_ATTN, 3 prompts (one ~2.4K tokens, chunked at 1024) | tiled local attention: ulysses / ring / allgather / adaptive sharded, passq replicated: tokens equal, dlp <= 0.019. MSA in place: passq / ulysses sharded: tokens equal, dlp <= 0.0052 |
| same, bf16 | noise reference: FLASH_ATTN vs TRITON_ATTN dlp <= 0.079 | MSA in place, passq sharded: dlp <= 0.10, 1 of 3 greedy sequences flips (near-tie, bf16). The tiled stand-in in bf16 is worse (dlp <= 0.42; P@V in bf16), so it is for fp32 / CPU only |
| **real vLLM**, `gpu_cp_holes_e2e.py` (holes/core's holed e2e: mid / front / two holes, a 4800-token hole chunked over 3 steps, 1-token tail, mixed batch) | holed run vs full recompute with CP_HOLES | fp32: passq / ulysses / adaptive sharded and ring replicated (tiled), passq sharded (MSA in place): 18/18 sequences equal, prefill computes exactly hole + tail, PASS. bf16 ulysses sharded (MSA): 17/18 equal, PASS by holes/core's noise rule |

The CP world size is 1 in the real-vLLM runs (one GPU). They exercise the backend inside vLLM: the builder on
holes/core metadata, the paged store and cache writes, decodes, the sharded runner path (identity shard and
gather) and MSA in place. They do not exercise collectives.

**Bench on the RTX 4050** (world 1, E = 8192, m = 2048, 2 layers; per-layer ms; `attn` = local attention).
Laptop numbers, relative only:

| layout | scheme | tiled attn | MSA gathered attn | MSA paged in place attn | comm-only |
|---|---|---|---|---|---|
| tail | passq | 178.5 | 17.1 | 11.4 | 3.7 |
| tail | ulysses | 178.5 | 16.7 | 11.4 | 3.4 |
| tail | ring | 177.6 | 14.4 | 14.3 (gathered by design) | 3.5 |
| tiny8 | passq | 160.7 | 15.3 | 10.1 | 3.6 |
| deep_small | ulysses | 183.1 | 18.1 | 11.5 | 3.4 |

The comm-only column (3.4-3.7 ms per layer at world 1, with no network) is per-layer Python, packing and
merging overhead. It is the next thing to cut before the server runs, e.g. by precomputing the pack index
tensors and avoiding the per-write `unique` / `nonzero` syncs in `PagedKVStore.write`.

Commands (from anywhere; GPU jobs through gpurun):
- `env PYTHONPATH=/home/asd/vllm-dev-wt/cp /home/asd/LLMServingSim/.wt/vllm-venv/bin/python
  /home/asd/vllm-dev-wt/cp/tests/v1/attention/cp_holes/test_cp_gloo.py --world 1 2 4 8` (the same for
  `test_backend_gloo.py --world 2 4`).
- `tests/v1/holes/*.py`: the same, with `PYTHONPATH=<wt>:<wt>/tests/v1/holes`.
- `/home/asd/LLMServingSim/.wt/bin/gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/cp <venv python> <script>` for
  `test_kv_placement.py`, `test_msa_paged_gpu.py`, `bench_cp_holes.py --check [--local-attn msa --store paged]`,
  `gpu_cp_holes_vs_stock.py` (`--backend TRITON_ATTN --out ref.json`, then `--cp-scheme S --cp-dense sharded
  --out x.json --compare ref.json`) and `gpu_cp_holes_e2e.py --cp-scheme S --cp-dense sharded --eager --dtype
  float32 --gpu-util 0.6`.

## 5. What needs the rented server

1. **NCCL correctness at d = 2/4/8.** `run_server.sh` runs `bench_cp_holes.py --check` (tiled; `--hkv 2` for
   replicated Ulysses heads; MSA paged in place) and records `nvidia-smi topo -m` / `-p2p`.
2. **Per-scheme step time.** `run_server.sh` (layouts rate / tail / deep_small / tiny8 / hole_only / multi, E
   32K / 128K, m 2K / 16K, d 1-8) runs MSA paged in place (the backend's path) and the tiled stand-in, and
   writes JSONL with `total`, `attn`, `comm` (null attention) and bytes sent per rank.
3. **Overlap check.**
   - Run `nsys` on one Ulysses / pass-Q step (2 micro-batches) and one ring step: do the NCCL kernels overlap
     MSA?
   - Measure the compute slowdown from NCCL's SMs (KERNEL-ALGO 7.4 item 4).
4. **End-to-end vLLM with pcp > 1.**
   - Flags: `--prefill-context-parallel-size d --attention-backend CUSTOM`, with `register()` called in the
     engine process (`VLLM_ENABLE_V1_MULTIPROCESSING=0` or a plugin) and `VLLM_CP_HOLES_SCHEME` set.
   - Per scheme:
     - pass-Q / ring / adaptive: `--cp-kv-placement interleave --block-size 128
       --cp-kv-cache-interleave-size 128`;
     - Ulysses: `--cp-kv-placement head`;
     - all-gather: `--cp-kv-placement replicate`.
   - Run replicated vs sharded dense, and compare logits with pcp = 1 (the scripts above take the same
     arguments).
   - Unverified here: the rest of the 0.19 GPU runner / executor with pcp > 1 (worker layout, output rank =
     PCP rank 0).

## 6. Integration with the sibling branches (merged)

- **`holes/kernel`.** MSA through `MsaLocalAttention` (gathered: ring buffers, all-gather cached keys) and
  `msa_attention` + `plan_msa_rows` on the paged cache (Ulysses, pass-Q; the plan is built once per step and
  micro-batch and reused by every layer).
  - MSA's split-KV covers Ulysses at 1-2 KV heads per rank and pass-Q's short local key ranges.
  - MSA's paged invariant (every local slot below a row's position is valid) holds: the step's keys are written
    before attention, and missing ranges lie past E_step (they are checked).
- **`holes/core`.** Segments come from `hole_segments`, and slot validity from `missing_kv_ranges`. The CP
  budgets live in its planner.
