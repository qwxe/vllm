# holes/core: hole-aware vLLM core (d = 1), as built

Branch `holes/core` of the vLLM v0.19.0 fork. Implements KERNEL-ALGO.md section 3 and section 8.2 step 1:
a prefill reuses cached KV blocks that lie *after* a missing range (a hole), recomputes only the hole(s)
and the tail, and runs every hole piece as its own varlen entry on the stock attention backends
(variant (a)). Off by default; with holes off every code path is the stock one.

## Enabling and knobs

```python
LLM(..., enable_prefix_caching=True,
    additional_config={"holes": {"enabled": True}})
# or env: VLLM_HOLES=1   /   VLLM_HOLES='{"enabled": true, "min_piece": 128}'
```

`vllm/v1/core/holes.py::HoleConfig` (env overrides `additional_config`):

| key | default | meaning (KERNEL-ALGO 5.2) |
|---|---|---|
| `enabled` | false | hole-aware lookup + scheduling |
| `chunk_granule` | 128 | a cut inside a segment falls at segment start + k * 128 (segment ends always allowed) |
| `min_piece` | 256 | q_min: no remainder 0 < r < 256 behind a cut (cut moves to the segment end if that overshoots the budget by < 256 and the step's hard token budget allows, else back by 128-multiples); a whole hole < 256 is not left as the last item of a step that has other work of the request (deferred to go with the next segment) |
| `step_budget` | 0 | per-request tokens per step while the request has holes; 0 = `max_num_batched_tokens` (set that to 2048 for the paper's B) |
| `min_reuse_blocks` | 1 | hit runs after a hole shorter than this are recomputed instead of reused |
| `min_mixed_tail` | 2 | a tail sharing a step with hole pieces needs >= this many tokens (else it runs alone next step); keeps multi-segment requests above FlashInfer's decode threshold (q <= 1) |
| `test_hooks` | false | enables `Scheduler.holes_evict_token_range` |

`long_prefill_token_threshold` still caps a holed request's per-step tokens. With holes, chunking rules
apply only while a request has pending holes (its hole steps and the tail chunks that share them);
requests without holes are chunked exactly as stock vLLM.

## Design as built (layer by layer)

### KV cache manager (`v1/core/kv_cache_manager.py`, `kv_cache_coordinator.py`, `single_type_kv_cache_manager.py`)
- `FullAttentionManager.find_cache_hits_with_holes(block_hashes, max_length, ...)`: looks up every full block
  of the first `num_tokens - 1` tokens, continuing past misses (stock `find_longest_cache_hit` breaks at the
  first miss and its comment "the following block hashes are not computed yet for sure" no longer holds once
  blocks can be evicted out of order). Block hashes are chained over token ids, so a block after a miss is
  found by its own hash.
- `KVCacheManager.get_cache_hits_with_holes(request)` -> per-block hits (None = miss), trailing misses trimmed,
  `min_reuse_blocks` applied.
- `KVCacheManager.get_computed_blocks_with_holes(request, external_hit_mask=None)` -> `(KVCacheBlocks, frontier,
  holes, load_ranges)`. The blocks list covers [0, frontier) with the **null block** at every non-hit position;
  frontier = end of the last hit block (becomes `request.num_computed_tokens`); holes = block-aligned token
  ranges to recompute; load_ranges = connector-only blocks (see connector).
- `allocate_slots(..., hole_ranges=...)`: for a new request with null placeholders, counts the hole blocks
  against the free pool, then `allocate_new_computed_blocks_with_holes` touches the hits and allocates a fresh
  block **in place** of every hole (block table stays position ordered). `num_new_tokens` counts only the tail
  (it may be 0 for a hole-only step). `can_fit_full_sequence` counts hole blocks too.
- Range-aware caching: `num_cached_block[req]` keeps meaning "frontier" for the stock tail caching;
  hole blocks live in `SingleTypeKVCacheManager.uncached_hole_blocks[req]` and `cache_hole_blocks(request,
  ranges)` caches the ones fully covered by the ranges scheduled this step (via `BlockPool.cache_full_blocks`
  on each contiguous run, so KV events carry the right parent hash). Caching at *schedule* time is vLLM's own
  convention for the tail (blocks are hashed when scheduled; a reader in the same step sees the KV because each
  layer writes KV before attention; a reader in a later step is ordered by the stream).
- Supported only for one full-attention KV cache group (`UnitaryKVCacheCoordinator.supports_holes`: prefix
  caching on, no EAGLE, DCP = PCP = 1). Otherwise a warning and the stock lookup.
- Test hook: `KVCacheManager.evict_cached_token_range(request, start, end)` drops the cached blocks of a prompt
  range from the prefix cache by hash (`BlockPool.evict_blocks`), as a lower tier would.

### Request / scheduler (`v1/request.py`, `v1/core/sched/scheduler.py`, `output.py`)
- `Request.pending_hole_ranges`: holes below the frontier still to compute. `num_computed_tokens` = the tail
  frontier (one frontier, unchanged meaning for everything else). Preemption clears it (re-admission looks up
  again; hole blocks already computed were cached and are found).
- Waiting requests: hole-aware lookup when enabled (`hole_mode`), then `plan_step` (holes front first, then
  the tail) picks this step's hole pieces and tail tokens; `allocate_slots(request, n_tail, frontier, blocks,
  hole_ranges=pieces)`. Running requests with pending holes: the same planner instead of
  `num_tokens - num_computed_tokens`.
- `SchedulerOutput.scheduled_hole_ranges: dict[req_id, list[(start, end)]]` (only holed requests).
  `num_scheduled_tokens[req]` = piece tokens + tail tokens (budget accounting unchanged).
- `_update_after_schedule`: only the tail advances `num_computed_tokens`; scheduled pieces are subtracted
  from `pending_hole_ranges`.
- Entries cap: hole pieces are extra varlen entries; the planner limits a request's segments so that the step
  has at most `max_num_seqs` entries (FlashInfer sizes buffers by it).
- `num_cached_tokens` = frontier - hole tokens. Prefix-cache stats count hit blocks.
- PP (non-async) `new_token_ids` excludes hole tokens.
- Holes are off for: V2 model runner, encoder inputs (multimodal / enc-dec), a KV connector that does not
  implement the mask API, `skip_reading_prefix_cache` requests (prompt logprobs, pooling).

### Model runner (`v1/worker/gpu_model_runner.py`, `v1/worker/hole_segments.py`)
- `_prepare_inputs`: when `scheduled_hole_ranges` is non-empty, `build_hole_segments` lays each holed
  request's tokens out as pieces in position order, then the tail, rewrites the CPU positions (so token ids
  are gathered from the right positions), copies them to the GPU positions before `compute_slot_mapping`
  (so `slot_mapping` follows), and subtracts the hole tokens from the request-level `seq_lens` /
  `optimistic_seq_lens_cpu` (a request's sequence ends at its tail; the discard mask therefore drops the sample
  of a step that has not reached the prompt end).
- Request-level `query_start_loc` is unchanged, so `logits_indices = query_start_loc[1:] - 1` is the tail's
  last token. A hole-only step's sample is discarded.
- `_build_attention_metadata(..., hole_segments=...)`: `segment_common_attn_metadata` turns the request-level
  `CommonAttentionMetadata` into one entry per segment: `query_start_loc` per segment, `seq_lens` = segment end,
  `block_table_tensor` rows gathered per segment (request's table repeated), `num_reqs` = #segments,
  `max_query_len` / `max_seq_len` from the segments; `slot_mapping` is per token and unchanged. Bottom-right
  causal masking makes query `s_i + t` see keys `[0, s_i + t]`.
- Steps with holes: no cascade attention, no FULL cudagraph (piecewise / eager only for that step), no
  microbatching (DBO). Holes are rejected with M-RoPE / XD-RoPE.
- No attention backend file is touched; `CommonAttentionMetadata` (in `v1/attention/backend.py`) gained one
  optional field, `hole_segments`.

### KV connector API (`distributed/kv_transfer/kv_connector/v1/hole_mask.py`)
Design + stub (the CPU-pool `rate` connector comes later):
- `SupportsBlockHitMask.get_block_hit_mask(request, local_hit) -> (mask | None, load_async)`: per-block mask of
  blocks the connector can load (not a prefix token count).
- `update_state_after_alloc_with_ranges(request, blocks, load_ranges)`: the token ranges to load into the
  in-place blocks (`blocks` = the request's full block list).
- Scheduler: combined hits = GPU hit OR connector hit; frontier = last combined hit; holes = misses in both;
  connector-only blocks are allocated in place, cached at schedule time and loaded synchronously before the
  forward (`start_load_kv` / `wait_for_layer_load`, as for stock external tokens).
- Not yet: asynchronous loads (`load_async=True` -> treated as misses, warning). A connector without the mixin
  keeps prefix semantics and turns holes off (stock behaviour).

## Interfaces for other agents (kernels / CP)

How a request's computed ranges reach attention:

1. Scheduler -> `SchedulerOutput.scheduled_hole_ranges[req_id] = [(s_i, e_i), ...]` plus
   `num_scheduled_tokens[req_id]`; the tail is `[num_computed_tokens, num_computed_tokens + n_tail)` with
   `n_tail = num_scheduled_tokens - sum(e_i - s_i)`. `num_computed_tokens` comes with the request data as
   usual.
2. Model runner -> `HoleSegments` (`vllm/v1/worker/hole_segments.py`): `seg_req_idx`, `seg_query_lens`,
   `seg_ends` (seg_starts = ends - q), `req_hole_tokens`, `cu_query_lens()`. Segments are in token order and
   cover all requests of the step (a request without holes is one segment).
3. Attention backends see `CommonAttentionMetadata` whose entries are segments, and
   `common_attn_metadata.hole_segments` (None when the step has no holes). A backend that wants the
   request-level view (e.g. an MSA planner with qpos / kblk_pos, or a CP split) maps segments to requests with
   `seg_req_idx`; per-token absolute positions are the model's `positions` tensor (= qpos).
4. The MSA backend should read per-segment `query_start_loc` / `seq_lens` exactly as the stock backends do
   (variant (a) is the planner setting "one tile chain per segment"); its own metadata builder can add
   `qpos` / `kblk_pos` / the plan from `hole_segments`.

## Deviations from KERNEL-ALGO.md

- Names: every vLLM-internal name cited by the doc exists in v0.19.0 (`find_longest_cache_hit` in
  `kv_cache_coordinator.py` + `single_type_kv_cache_manager.py`; `allocate_slots`, `cache_blocks`
  (`BlockPool.cache_full_blocks` underneath), `get_num_new_matched_tokens`, `update_state_after_alloc`,
  `start_load_kv`, `wait_for_layer_load`, `_prepare_inputs`, `CommonAttentionMetadata` (in
  `v1/attention/backend.py`), `cp_lse_ag_out_rs` (`v1/attention/ops/common.py`), `triton_unified_attention`
  (`v1/attention/ops/`), `merge_attn_states`, `step_with_batch_queue`, `OffloadingConnector`,
  `reshape_and_cache(_flash)`, `cp_kv_cache_interleave_size`). The simulator's `num_tokens_reached` is
  vLLM's `Request.num_tokens`. The doc's `find_cache_hits_with_holes` is implemented under that name; the
  per-block hit vector is returned as blocks-with-null-placeholders + hole ranges.
- "Cache by computed ranges": hole blocks are cached when their range is *scheduled* (vLLM's existing
  convention for the tail), not when the step finishes. Equivalent under vLLM's stream ordering.
- `min_mixed_tail` (new): the doc does not cover a 1-token tail next to hole pieces; FlashInfer's decode/prefill
  split requires all q <= 1 entries first, so such a tail runs one step later. Cost: one extra step for
  requests whose whole tail is 1 token (prompt length = 1 mod 16 with every block cached but a hole).
- Segment cap at `max_num_seqs` (new; backend buffers).
- The q_min "small hole goes with the next segment" rule is implemented as "a whole hole < q_min is not the
  last item of a step that holds other work of the request"; if it is the only work it runs.
- Chunking rules apply to holed requests only (stock behaviour elsewhere, by requirement).
- Asynchronous connector loads with holes: not supported yet.
- KV events: a hole block's `BlockStored` names its position predecessor as parent even if that block is not on
  the GPU (consumers that rebuild chains must tolerate it).

## Tests and results

CPU unit tests (`tests/v1/holes/`, run with pytest or `python <file>`; no GPU, no network):
- `test_holes_plan.py` (13): planner rules (front first, 128 cuts, q_min overshoot / cut back, short hole
  deferral, min_mixed_tail, budget < granule, configurability, segment cap), range helpers.
- `test_holes_kv.py` (10): lookup continues past a miss (stock stops); in-place allocation keeps hit block ids
  and position order; range caching (block 3 cached after its range, 4-5 not yet; all after the second step);
  free returns every block; `min_reuse_blocks`; `can_fit_full_sequence` counts holes; scheduler: hole + tail
  in one step, a 4800-token hole chunked 2048/2048/704 front first then a 1-token tail, preemption clears holes
  and re-admission reuses the computed hole blocks, mask connector (connector-held blocks loaded, the rest
  recomputed), and **no-hole traces identical to stock** (scheduled tokens, computed tokens, block ids,
  per step, holes on vs off).
- `test_holes_segments.py` (3): segment layout and positions; a CPU reference of the backends' varlen
  bottom-right-causal paged attention over the segment metadata equals full causal attention at the same
  positions (paged cache with shuffled block table, holes written this step).

GPU end-to-end (`tests/v1/holes/gpu_holes_e2e.py`, Qwen2.5-0.5B-Instruct, RTX 4050 6 GB, through `gpurun`,
async scheduling on): per case, full recompute (empty cache) vs. the same prompt after evicting hole ranges
with the test hook; also the stock arms "full prefix hit" and "prefix-only" (evict from the hole start to
the end, which is what stock vLLM recomputes) as the numeric-noise reference.
Cases: mid hole [512, 1536) of 3000; front hole [0, 512); two holes; a 4544-token hole chunked over 3 steps
(6000 tokens); a 1-token tail; a batch with a holed, a fresh and a fully cached request.

Compute check (all runs): the holed run's prefill computes exactly hole + tail tokens (e.g. mid hole: 1032
tokens instead of 3000; 4544-token hole: 4560 = 2048 + 2048 + 448 hole pieces + a 16-token tail, cut front
first at 2048 / 128 multiples), and the blocks after the hole are reused (`cached_tokens` = frontier - hole).

Numerics: `dlp` = max |logprob difference| over the top-5 of the first generated token (the prefill's output)
against the full recompute. "Stock noise" = the same metric for stock-path arms (full prefix hit,
prefix-only recompute, a fresh request in a different batch); stock vLLM itself moves logprobs this much.

| run | backend | cases | greedy tokens = full recompute (24/24) | dlp, holed | stock noise max | result |
|---|---|---|---|---|---|---|
| fp32, eager | TRITON_ATTN | 8 | 8/8 | 0.0010-0.0070 | 0.0127 | PASS |
| bf16, torch.compile + CUDA graphs | FLASH_ATTN (FA2) | 8 | 8/8 | 0.0625-0.125 | 0.187 | PASS |
| bf16, eager (earlier build, before the segment cap) | FLASH_ATTN | 8 | 7/8 (two_holes diverged at step 6, a near tie) | <= 0.125 | 0.375 | PASS |
| bf16, eager (earlier build) | TRITON_ATTN | 8 | 8/8 | <= 0.125 | 0.250 | PASS |
| bf16, eager | FLASHINFER | - | - | - | - | not runnable here (FlashInfer JIT needs nvcc, absent) |

fp32 is the sharp check: the holed result agrees with the full recompute to <= 0.007 in logprob, inside the
noise of stock vLLM's own recomputation paths (<= 0.0127, TF32 GEMMs), with identical greedy tokens. In bf16
logprobs move in steps of 0.0625 and greedy near-ties can flip on any path (the stock prefix-only arm itself
diverged once), so bf16 is checked against the stock noise. Pass rule in the script: dlp(holed) <= 2 * stock
noise + 1e-3, and identical tokens in fp32. Async scheduling was on in all runs (vLLM 0.19 default).

Commands (inside `tests/v1/holes/`, one GPU job at a time):
```
gpurun python gpu_holes_e2e.py --eager --dtype float32 --gpu-util 0.75   # Triton, fp32
gpurun python gpu_holes_e2e.py --gpu-util 0.7                            # FA2, bf16, compiled
python test_holes_plan.py; python test_holes_kv.py; python test_holes_segments.py   # CPU
```
Rerun from this worktree (DEV.md recipe, `vllm.__file__` = `/home/asd/vllm-dev-wt/core/vllm/__init__.py`):
CPU 13 + 10 + 3 PASS; GPU fp32/TRITON_ATTN eager PASS (8/8 tokens equal, dlp 0.0010-0.0070, stock noise
0.0127) and bf16/FLASH_ATTN compiled + CUDA graphs PASS (8/8 tokens equal, dlp <= 0.125, stock noise 0.187),
the same numbers as the first runs from the identical patched copy of the wheel.
```
cd tests/v1/holes
env PYTHONPATH=/home/asd/vllm-dev-wt/core <venv>/bin/python test_holes_kv.py        # etc.
gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/core <venv>/bin/python gpu_holes_e2e.py --eager --dtype float32 --gpu-util 0.75
gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/core <venv>/bin/python gpu_holes_e2e.py --gpu-util 0.7
```

## Not done / next
- CPU-pool connector with `rate` eviction implementing `SupportsBlockHitMask` (async loads, layer-wise wait).
- MSA backend reading `hole_segments` (variants (b)/(c)); profiler `segmented_attention` category.
- V2 model runner, hybrid / sliding-window models, EAGLE, DCP/PCP with holes.
