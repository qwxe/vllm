# holes/rate: the session-state `rate` eviction policy in vLLM (GPU only), as built

Branch `holes/rate`, from `holes/cp` (= holes/core + holes/kernel + holes/cp). Worktree
`/home/asd/vllm-dev-wt/rate`. The branch implements claim 1 of PAPER-ALGORITHM.md section 2 for the GPU
prefix cache **only**. There is no DRAM/CPU tier, no offload and no prefetch (future work). `rate` decides
which cached KV blocks (ref_cnt == 0) vLLM evicts when it needs free blocks. It evicts partially, by
256-token chunk, so it can leave holes, and holes/core then reuses the blocks after a hole.

## Enabling

```python
LLM(..., enable_prefix_caching=True,
    additional_config={"holes": {"enabled": True},
                       "eviction": {"policy": "rate", "params": "/path/params.json"}})
# or env: VLLM_EVICTION=rate VLLM_EVICTION_PARAMS=/path/params.json
#         VLLM_EVICTION='{"policy": "rate", "params": "...", "chunk_tokens": 256}'
# session identity, per request:
SamplingParams(..., extra_args={"session_id": "codex:b5a9..."})   # OpenAI server: vllm_xargs
```

`vllm/v1/core/eviction.py::EvictionConfig` (env overrides `additional_config`):

| key | default | meaning |
|---|---|---|
| `policy` | `lru` | `lru` = stock vLLM (no policy object is installed); `rate` |
| `params` | - | path or dict: `tool_gaps`, `decode_durs` (s), `p_end`, `floor` (the simulator's `rate_mc/params/*.json` format) |
| `chunk_tokens` | 256 | eviction chunk; rounded down to a multiple of the block size (16 blocks at block_size 16) |
| `rescore_s` | 1.0 | full heap rebuild at most this often (0 = every eviction) |
| `cost_b1`, `cost_b2` | 3.87e-5, 1.85e-9 | recompute cost line (the simulator's fit; see below) |
| `running_owner` | `protect` | owner past its prefill but unfinished (= preempted): `protect`, or `decode` (the simulator's rule) |
| `time_scale` | 1.0 | multiplies the params' time samples and the floor (time-compressed experiments) |

rate is ignored with a warning, and LRU is used, when prefix caching is off, when there is more than one
KV cache group, when a KV connector is attached (connectors such as `simple_kv_offload` manipulate the
free queue directly), or with DCP/PCP > 1. rate without holes logs a warning: a stock lookup stops at
the first evicted chunk, which PAPER-ALGORITHM.md says makes rate worse than LRU.

## Design as built

### Where (`block_pool.py`, `single_type_kv_cache_manager.py`)
- `BlockPool.eviction_policy` is `None` by default. Every hook is guarded by `is not None`, so with
  `lru` all code paths are stock and scheduling is byte-identical (tested).
- With a policy, `FreeKVCacheBlockQueue` still holds every free block (membership, count, O(1)
  `remove`), but `get_new_blocks` asks the policy (`take_free(n)`) instead of popping the head.
- Hooks into the policy:
  - `free_blocks` -> `on_free` for each block that reaches ref_cnt 0;
  - `touch` -> `on_remove` when a hit takes a block off the free queue;
  - `_maybe_evict_cached_block` -> `on_uncache` when a *free* block loses its hash (connector or test
    eviction, or the policy's own victim; KV events and metrics go through the stock path);
  - `reset_prefix_cache` -> `reset`;
  - `SingleTypeKVCacheManager.free` -> `on_request_free(request_id, block_table)` before freeing. The
    freeing request becomes the block's **owner**, and the block's index in the table becomes its
    position.
- **Owner = last holder**: the request that released the block last. For a block shared by concurrent
  requests, that is the one that ran longest. It matches the simulator's "last request over the chunk"
  (touched every step) for a single tier.

### Chunks and groups
- A chunk is `chunk_tokens / block_size` consecutive positions of a block table: chunk k = blocks
  [16k, 16k+16) at block size 16.
- The eviction unit is a **group**: the free cached blocks with the same (owner, chunk k). All blocks
  of a group have the same score, and a group is evicted **whole**. Holes are therefore chunk aligned
  (>= holes/core's `min_piece` = 256), and holes/core's planner is not fed 16-token fragments.
- Blocks a group frees beyond what the allocation needs become free blocks without a hash. They are
  used first by later allocations, so no capacity is lost, only cache contents earlier.
- Free blocks **without** a hash are always used before any cached block is evicted. This differs
  from stock vLLM, whose single queue can pop a cached block ahead of a later-freed uncached one.
- Evicting a middle chunk only removes those blocks' hashes from `cached_block_hash_to_block`. Block
  hashes are chained over token ids, not over KV presence, so later blocks keep their hashes and
  holes/core's `find_cache_hits_with_holes` finds them past the miss (tested at scheduler level:
  `test_rate_holes_front_first_and_reuse`, `test_rate_dead_session_first`).

### Score (identical to the simulator's `SessRatePool`, mode `rate`)
`score(group) = rate(owner) * c(k)`. The lowest score is evicted first. Ties go by chunk index (front
first), then by age.

| owner state | rate |
|---|---|
| prefill not finished | protected (1e6, as the simulator's `_BIG`) |
| session's next request arrived, not through prefill | protected |
| session's next request through prefill | 0 (dead) |
| past prefill, not finished (preempted) | protected (`running_owner=protect`), or `q / max(MRL_dec(e) + mean gap, floor)` |
| finished t s ago | `p / max(MRL_tool(t), floor)`, `p = qS(t)/(qS(t)+p_end)`, `q = 1-p_end` |
| no record | `1 / max(mean gap, floor)` |

- **Protected chunks** stay evictable as a last resort, front first.
- **The decode state.** In the simulator's P/D setup an owner decodes on another instance while its
  chunks sit in the prefill tier. On one GPU a decoding request holds its blocks (ref_cnt > 0), so
  they are never candidates. The only unfinished owner with free cached blocks is a preempted one,
  which vLLM re-admits first, so the default protects it (deviation; the `decode` option restores the
  simulator's rule).
- **c(k)** is the recompute seconds of tokens [kC, (k+1)C) from `S(L,H) = A + b1(L-H) + b2(L²-H²)/2`,
  i.e. `b1 C + b2 ((k+1)² - k²) C² / 2`, as the simulator's `_chunk_s`. Only b2/b1 changes the order.
  - The defaults are the simulator's fit.
  - The e2e script's `calib` command fits `A, b1, b2` on the target GPU from fresh prefills of
    256-8000 tokens. RTX 4050, Qwen2.5-0.5B, fp32 eager: A = 4.5 ms, b1 = 1.98e-4 s/token,
    b2 = 2.26e-8 s/token², so a token at 8K costs about 1.9x a token at the front.
  - Fitting directly from the profiler's dense/attention CSVs would also work (sum of dense slopes ->
    b1, prefill-attention slope over kv -> b2), but it is not automated.

### Session identity and state (`SessionTracker`)
- `session_id` comes from `sampling_params.extra_args["session_id"]`. A request without one is a
  session of its own. A request's successor is the next request of its session to arrive.
- The scheduler feeds the tracker. Times come from `time.monotonic()` in the engine-core process
  (the clock can be injected for tests).
  - `add_request` -> arrival;
  - `update_from_output`, the first non-empty sampled tokens -> prefill done (same as the simulator's
    first-token time);
  - `_free_request` -> finished, before its blocks are freed, so the freed groups are scored as idle.
- The tracker never needs to know the session length. Like the simulator, the end of a session is
  handled only through p_end.
- Records are dropped once no cached group depends on them: the request is finished, it is not its
  session's latest request, and it has no predecessor left.

### Victim selection
- A lazy min-heap over groups. Each entry carries (score, chunk, seq, key, heap epoch, owner version,
  state). An entry is valid only if its epoch and owner version are current and the group is
  non-empty.
- A new group (empty -> non-empty) is pushed when it appears.
- A session event (successor arrival, successor prefill end, own finish, abort before prefill) bumps
  the owner's version and re-pushes that owner's groups with fresh scores. That is O(groups of 1-2
  owners) per event, instead of the simulator's full rebuild per epoch.
- Idle scores drift with time, so the heap is rebuilt at most once per `rescore_s` (1 s), when an
  eviction is needed. This is the same staleness bound as the simulator's `rescore_ns`. With
  `rescore_s=0` the order equals the simulator's exactly (tested).

## Deviations from the simulator
1. GPU tier only. The candidates are free cached blocks, not every resident copy. Chunks in use are
   not evictable at all (in the simulator they are, as protected).
2. Unfinished, past-prefill owners (preempted) are protected by default (the `decode` option keeps the
   simulator's rule).
3. Owner = last holder at free time, not the last request touched per step.
4. Incremental per-owner rescoring on events, plus a periodic full rebuild, instead of a full rebuild
   per event epoch.
5. A group can be smaller than a chunk: shared prefixes, a request's last partial chunk, or blocks
   still in use. It is scored per chunk, which equals the per-block cost of a full chunk, so
   per-capacity comparisons hold.
6. No session-length oracle is used (the simulator does not use one in mode `rate` either).

## Tests

CPU (`tests/v1/holes/test_rate_eviction.py`, `python <file>` or pytest; no GPU) - 10/10 pass:
- `test_parity_with_simulator`:
  - 9 owners cover every state (idle long and short, deep chunk, successor queued, dead, prefilling,
    decoding, idle at a mid position).
  - The eviction order taken through the real `BlockPool.get_new_blocks` path equals the order of the
    simulator's `SessRatePool` (imported from LLMServingSim when reachable; a recorded order
    otherwise). Scores agree to 1e-9 relative.
- `test_running_owner_protected_by_default`, `test_chunk_cost_position` (100K/front ratio about 5.7x
  with the paper fit).
- Pool level:
  - `test_uncached_blocks_first_and_whole_chunks`: a 1-block allocation evicts one whole front chunk,
    and the rest of it serves the next 15 allocations.
  - `test_touch_protects_and_owner_moves`, `test_evict_blocks_hook_moves_to_uncached`.
  - `test_invariants_fuzz`: 5 seeds x 400 random allocate / free / touch / evict ops. Free queue =
    uncached ∪ grouped, hashes are consistent.
- Scheduler level (holes on):
  - `test_lru_default_is_stock`: the per-step trace is identical with no config and with
    `policy=lru`, and no policy object is installed.
  - `test_rate_holes_front_first_and_reuse`: rate evicts the front chunk of the longest-idle session,
    and its next turn recomputes hole [0, 256) and reuses the 60 blocks after it.
  - `test_rate_dead_session_first`: the chunks a successor did not reuse go first. LRU instead takes
    the older live session's tail.
- The existing holes / CP CPU tests still pass: kv 10, plan 13, segments 3, cp_budget, missing_kv.

GPU end-to-end (`tests/v1/holes/gpu_rate_e2e.py`, RTX 4050 6 GB, Qwen2.5-0.5B-Instruct, fp32, eager,
TRITON_ATTN, gpu_memory_utilization 0.7, max_model_len 8192, max_num_seqs 6, holes on in every arm,
greedy, ignore_eos):
- **Workload.** The first 24 sessions of TraceLab `codex_n100_s400_k600_seed0`, scaled down:
  - first prompt /16 (capped at 1024), new tokens per turn /8, output /8 (capped at 48), at most 10
    turns;
  - real-time arrivals and tool gaps x0.5;
  - 146 requests, 313K prompt tokens, final contexts up to 6.7K tokens.
  - Each turn's prompt = the previous prompt + the previous *generated* output + new random tokens.
    A wrong KV would therefore change the output and every later turn.
- **rate params** come from the same dataset: all 897 codex tool gaps, `p_end` = 100/997. They use
  `time_scale` 0.5 and the 4050 cost fit above.
- **Arms.**
  - `big`: 5863 blocks, no eviction; this is the reference.
  - `lru` and `rate`: `num_gpu_blocks_override` 1500 or 900.
  - The 900-block arms ran a warm-up first; the 1500-block arms and `big` did not.
- **Results.** In the run tabled below, outputs were identical to the no-eviction reference in every
  arm: **146/146 requests, token for token**, for both LRU and rate at both cache sizes. This identity
  is run-dependent, not guaranteed: fp32 greedy decoding over a different batch / chunk / hole layout
  can flip a near-tie. A verifier rerun matched 145/146 for rate; the one divergence (session 10,
  turn 4, token 2) was a near-tie with a logprob gap of 0.0016, consistent with numerics rather than
  a wrong KV.

| cache | arm | hit tokens | recomputed prompt tokens | scheduled tokens | hole tokens | preempt | TTFT mean / p50 / p90 (s) | evicted (rate) |
|---|---|---|---|---|---|---|---|---|
| 5863 | big (ref) | 251,840 (80.4 %) | 61,410 | 65,745 | 0 | 0 | 0.245 / 0.205 / 0.439 (no warm-up) | - |
| 1500 | lru | 240,336 (76.7 %) | 72,914 | 77,249 | 0 | 0 | 0.167 / 0.147 / 0.349 | - |
| 1500 | rate | 238,656 (76.2 %) | 74,594 | 78,929 | 3,072 | 0 | 0.169 / 0.166 / 0.304 | idle 3,360 blocks (221 groups) |
| 900 | lru | 151,472 (48.4 %) | 161,778 | 166,113 | 0 | 0 | 0.903 / 0.234 / 3.054 | - |
| 900 | rate | **160,768 (51.3 %)** | **152,482 (-5.7 %)** | 157,079 | 34,304 | 2 | **0.742 / 0.221 / 2.383** (-18 % / -6 % / -22 %) | idle 8,112, queued 722, prefill 16 |

Notes:
- **1500 blocks, a tie.** Both arms lose almost the same 12-16 requests. All of those losses fall in
  the arrival burst around t = 30-45 s, when nearly every free block is evicted whatever the order, so
  the policy has little to choose.
- **900 blocks, rate won in this run.** It keeps more of the sessions likely to return. It pays in holes:
  34K hole tokens recomputed, and 2 preemptions. The protected 'queued' / 'prefill' chunks were
  evicted only as a last resort.
- **Hole tokens include the gaps between holes.** In the hole-tokens column the LRU arms have 0 not
  because LRU leaves no gaps: LRU evicts tail first, so its misses are suffixes, which are recomputed
  as the tail, not as holes.
- **The lru-vs-rate comparison is run-dependent.** The verifier rerun reversed it: hit rate 50.3 %
  for rate vs 51.5 % for LRU. With one run per arm the difference between the arms is within
  run-to-run variation, so the table does not establish that rate beats LRU on this workload.
- **The numbers are indicative only.** Each arm ran once, in real time on a laptop GPU (TTFT includes
  scheduling jitter). The small model also makes position cost almost flat (b2/b1 is small at <= 8K
  tokens). The paper's regime (100K-token contexts, a lower tier) is not reproduced here.

Commands (the one GPU job at a time, as `gpurun env PYTHONPATH=/home/asd/vllm-dev-wt/rate <venv python> ...`):
```
python tests/v1/holes/test_rate_eviction.py                                   # CPU
python tests/v1/holes/gpu_rate_e2e.py workload --dataset JSONL --out D        # CPU
python tests/v1/holes/gpu_rate_e2e.py calib --out D --max-model-len 8192      # GPU, cost fit
python tests/v1/holes/gpu_rate_e2e.py run --arm big  --out D --max-model-len 8192
python tests/v1/holes/gpu_rate_e2e.py run --arm lru  --out D --max-model-len 8192 --num-blocks 900
python tests/v1/holes/gpu_rate_e2e.py run --arm rate --out D --max-model-len 8192 --num-blocks 900
python tests/v1/holes/gpu_rate_e2e.py report --out D                          # CPU, output equality
```
Each run takes about 5 min (real-time workload). The result JSONs of these runs were scratch output
and are not committed.

## Not done / next
- A lower tier (CPU pool connector, `SupportsBlockHitMask`) with prefetch / offload. This is future
  work by decision.
- Online learning of the tool-gap / decode samples and p_end from finished sessions (PAPER-ALGORITHM
  open item).
- CP-degree-aware cost (Algorithm 1): the hook is `owner_rate` x `chunk_cost`; M_k and the group
  crossing need the CP group queues.
- Repeated GPU runs or a larger GPU and model for confidence intervals; a run with `rescore_s` 0.
- Hybrid / multi-group KV caches, connectors, DCP/PCP with a policy.
