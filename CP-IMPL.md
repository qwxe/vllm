# CP for holed prefills in vLLM 0.19: implementation notes (branch `holes/cp`)

Implements KERNEL-ALGO.md section 4 (and 8.2 steps 4-5): Ulysses, pass-Q, ring pass-KV, all-gather KV, and the
per-request ring / pass-Q choice, for prefill steps whose computed tokens are arbitrary segments (hole pieces +
tail). The communication and partition layer is complete and tested over gloo. The local attention is a
pure-PyTorch reference behind an interface that the hole-aware kernel (branch `holes/kernel`) plugs into.

**Where the code is.** Branch `holes/cp`, worktree `/home/asd/vllm-dev-wt/cp` (on `holes/base` = v0.19.0).
- The first 10 commits were written in a staging repo (`/home/asd/LLMServingSim/.wt/cp-staging`) while the fork
  was being set up, and moved here with `format-patch` + `git am`. They add new files only.
- `holes/core` is **merged** into `holes/cp` (merge commit "Merge holes/core ..."): the backend reads its
  `CommonAttentionMetadata.hole_segments`.

| file | what |
|---|---|
| `vllm/v1/attention/cp_holes/local_attn.py` | `KeySet`, `LocalAttention` protocol, `RefLocalAttention`, `TiledLocalAttention`, `merge_attn_states` (vLLM's DCP Triton combine on CUDA, PyTorch otherwise) |
| `vllm/v1/attention/cp_holes/plan.py` | `StepPlan` (global step description), `home_rank` (interleave u), `rows_contiguous` (micro-batched), `rows_c2e` (COBAL c2e, cuts on u) |
| `vllm/v1/attention/cp_holes/comm.py` | byte-packed variable-size all-to-all, all-gather, ring send/recv (`batch_isend_irecv`) |
| `vllm/v1/attention/cp_holes/kv_store.py` | `RefKVStore`, `PagedKVStore` (vLLM paged cache), `vllm_local_layout` / `vllm_slot_mapping` (vLLM's PCP placement) |
| `vllm/v1/attention/cp_holes/schemes.py` | `UlyssesCP`, `PassQCP`, `RingCP`, `AllGatherKVCP`, `AdaptiveCP`, `make_scheme` |
| `vllm/v1/attention/backends/cp_holes.py` | `CPHolesBackend` / builder / impl, `register()` (as `AttentionBackendEnum.CUSTOM`) |
| `tests/v1/attention/cp_holes/test_cp_gloo.py` | scheme math vs single-device attention, gloo, world 1/2/4/8 |
| `tests/v1/attention/cp_holes/test_backend_gloo.py` | the backend glue (builder + impl) on vLLM's paged layout, with holes/core's `hole_segments` and without holes, gloo, world 2/4 |
| `benchmarks/cp_holes/bench_cp_holes.py`, `run_server.sh` | NCCL correctness check and per-scheme step time (torchrun) |

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
  per-row limit and the ring's buffer positions are this mask, so no scheme needs a special kernel. Switch point:
  `set_local_attention(fn)`, or the `local_attn=` argument.
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
- **KV placement.** PCP already provides the virtual block. `single_type_kv_cache_manager.py` multiplies the
  block size by pcp·dcp, and `v1/worker/block_table.py::_compute_slot_mapping_kernel` interleaves by
  `cp_kv_cache_interleave_size`. pass-Q / ring / adaptive use that unchanged (`vllm_local_layout` is its
  inverse).
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
  `PagedKVStore.present` marks hole slots that are allocated in place but not yet computed, so they are never
  read before they are written.
- **Dense-layer modes** (`VLLM_CP_HOLES_DENSE`):
  - `replicated` (default) works with the **stock model runner**. Every PCP rank runs the dense layers on all
    tokens; attention runs the scheme on `own_rows` and all-gathers O (`gather_rows`). Logits are identical on
    all ranks, so sampling is unchanged. This is the end-to-end correctness mode: it costs d x dense compute and
    an extra all-gather of O.
  - `sharded` is the performance mode. The runner must feed each rank only `metadata.scheme.own_rows` (helper
    `shard_inputs` for `input_ids` / `positions`) and gather the logits rows (`gather_rows`). That is a
    `gpu_model_runner._prepare_inputs` patch, not written yet: it needs the attention metadata before the input
    slice, so the builder's plan has to be built first.
- **Merge.** On CUDA it uses `vllm.v1.attention.ops.dcp_alltoall.dcp_lse_combine_triton` (handles -inf
  partials; head_dim a power of 2), otherwise PyTorch. `cp_lse_ag_out_rs` is at `v1/attention/ops/common.py`
  (not `vllm/attention/ops`, as the doc assumed). `merge_attn_states` is at
  `v1/attention/ops/merge_attn_states.py`.
- **Arctic Inference** is not installed in the venv and was not vendored. Ulysses follows its design
  (all-to-all around attention, TP-like head-sliced KV, KV-head replication for SP > KV heads), reimplemented in
  `UlyssesCP`.

### Core patches still needed (outside this branch's files)
1. **Ulysses and AG-R KV accounting.** With pcp > 1 the KV manager multiplies the block size by pcp, which is
   right only for position sharding. Ulysses (head slices of every position) and AG-R (replicated) need a
   placement switch (proposed: `ParallelConfig.cp_kv_placement = interleave | head | replicate`) that leaves the
   block size alone. Ulysses also needs the KV cache spec at Hkv/d heads (1 when d > Hkv).
2. The `sharded` dense mode's runner patch (above).
3. (Done via the holes/core merge: `hole_segments`.) Still open: CP + holes in the scheduler. holes/core's
   chunk planner (5.2 cuts) does not know the per-scheme budgets of KERNEL-ALGO 5.3 (2048·d, ring
   max(2048, C_ring) per rank).
4. CUDA graphs: the backend is eager-only (`_cudagraph_support = NEVER`, host-side plan per step). This is fine
   for prefill-only P/D instances.
5. fp8 KV is refused (`kv_cache_dtype == "auto"` only).

## 4. Tests run here (all pass)

| test | what it checks | result |
|---|---|---|
| `test_cp_gloo.py --world 1 2 4 8` (gloo, CPU, `.wt/vllm-venv` python) | every scheme vs single-device attention over the full context. 7 random holed requests per step (several holes, sub-tile pieces, tail-only, fresh, hole-only steps, keys past E_step); Hq 8, Hkv 2/4/8 (Ulysses d > Hkv at d = 4, 8: Hkv 2, 4); fp32 (tol 2e-5) and bf16 (tol 3e-2); u = 16 and u = 128; 1 and 2 micro-batches; adaptive with both sub-schemes active; cache writes (each rank holds exactly what its placement homes, values bit-exact); all rows covered once | 160 / 160 OK (8 configs x 5 schemes x 4 world sizes); fp32 max error 8.3e-7, bf16 1.05e-2 |
| same, `store="paged"` case | the schemes over `PagedKVStore` with vLLM's PCP slot layout and random block ids; store slots equal to vLLM's slot-mapping formula | OK at world 1/2/4/8 |
| `test_backend_gloo.py --world 2 4` | `CPHolesMetadataBuilder.build` + `CPHolesImpl.forward` (replicated dense) on a paged cache filled at vLLM's slots, vs reference; cache writes at vLLM's slots | 20 / 20 OK (5 schemes x {hole_segments, stock} x world 2/4), max error 6.6e-7 |
| `test_merge_and_c2e` | LSE merge over key chunks = unsplit; c2e covers every row once | OK |
| GPU smoke, world 1 (RTX 4050, `gpurun`): `bench_cp_holes.py --check` | all schemes, 7 layouts, bf16, `TiledLocalAttention`, fused Triton merge | CHECK PASS, 35 cases (max error 7.8e-3, bf16 O rounding on the fresh layout) |
| fused vs PyTorch merge on GPU | `dcp_lse_combine_triton` path incl. -inf partials | fp32 1.7e-6, bf16 7.8e-3 (bf16 output rounding) |

Commands (all rerun from the worktree after the move and the holes/core merge):
- `env PYTHONPATH=/home/asd/vllm-dev-wt/cp /home/asd/LLMServingSim/.wt/vllm-venv/bin/python
  /home/asd/vllm-dev-wt/cp/tests/v1/attention/cp_holes/test_cp_gloo.py --world 1 2 4 8`
- the same for `test_backend_gloo.py --world 2 4`;
- `/home/asd/LLMServingSim/.wt/bin/gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/cp
  /home/asd/LLMServingSim/.wt/vllm-venv/bin/python /home/asd/vllm-dev-wt/cp/benchmarks/cp_holes/bench_cp_holes.py
  --check` (allocates well under 1 GB; no vLLM engine, so `gpu_memory_utilization` does not apply).

Pytest works as well (`pytest` is not in the venv).

## 5. What needs the rented server

1. **NCCL correctness at d = 2/4/8:**
   - `torchrun --nproc-per-node d benchmarks/cp_holes/bench_cp_holes.py --check` (plus `--hkv 2` for U-R).
   - `run_server.sh` does both, and records `nvidia-smi topo -m` / `-p2p`.
2. **Per-scheme step time.** `run_server.sh` (layouts rate / tail / deep_small / tiny8 / hole_only / multi, E
   32K / 128K, m 2K / 16K, d 1-8) writes JSONL. It reports `total`, `attn` (local attention alone), `comm` (the
   same forward with a null attention: collectives + packing + merges + cache writes) and bytes sent per rank.
   - The timings are only as good as the local attention. `TiledLocalAttention` is an unfused PyTorch stand-in.
   - Pass `--local-attn module:Class` for the `holes/kernel` adapter once it exists.
   - The reference stores gather resident KV into contiguous buffers (one HBM copy per layer). A paged in-place
     kernel removes that copy (see 6).
3. **Overlap check.**
   - Run `nsys` on one Ulysses / pass-Q step (2 micro-batches) and one ring step. Confirm that NCCL kernels on
     the NCCL stream overlap the attention on the compute stream.
   - Measure the compute slowdown from NCCL's SMs (KERNEL-ALGO 7.4 item 4).
4. **End-to-end vLLM, replicated-dense mode.**
   - Run `--prefill-context-parallel-size d --block-size 128 --cp-kv-cache-interleave-size 128
     --attention-backend CUSTOM` with `register()` called (e.g. from a plugin) and `VLLM_CP_HOLES_SCHEME=passq|ring|adaptive`.
   - Compare logits with pcp = 1 on the same prompts. This also checks whether the 0.19 GPU runner otherwise
     tolerates pcp > 1, which is unverified here (needs ≥ 2 GPUs).

## 6. Integration with the sibling branches (pending)

- **`holes/kernel`.** Wrap its O+LSE API as a `LocalAttention`: `(q, qpos, qreq, KeySet, scale) -> (O, LSE)`.
  - For the paged path, extend the store protocol with a view that returns `(kv_cache, block_table, kblk_pos,
    limits)` instead of a gathered `KeySet`. `PagedKVStore` already holds these. Ring buffers stay gathered (they
    are network buffers).
  - Ulysses at 1-2 KV heads per rank and pass-Q with short local key ranges need its split-KV (variant (c)).
  - `KERNEL-IMPL.md` was not present when this was written.
- **`holes/core`** (merged). Segments come from `hole_segments`. Open items:
  - `PagedKVStore.present` is derived as "position < E_step and not computed this step". That matches holes/core,
    which allocates hole blocks in place and schedules front first, but it is not read from its block hit mask.
  - The end-to-end path (holes/core runner + CP_HOLES with pcp > 1) needs ≥ 2 GPUs (section 5, item 4).
