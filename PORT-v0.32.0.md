# vLLM v0.32.0 fork port

## cbb2490b6d — KV cache manager: lookup past misses, in-place hole blocks

Status: adapted. Original commit: cbb2490b6dd6ce580d8c75c55d0dec10c011d4fc.
This section covers the stopped cherry-pick on `port/v0.32.0/holes-core`;
HEAD remains `63ca7eb8f3369d36b27ae8e03ffb70ebe1aa57f1` until the master
stages the resolution and continues the cherry-pick. AI assistance was used.

Preserved fork behavior:

- Look up prompt blocks past misses, trim trailing misses and short reuse runs,
  combine local hits with an optional connector mask, and return the frontier,
  recompute ranges and load ranges. The final token still requires computation.
- Touch reused blocks and replace null hole entries with fresh blocks in position
  order. Cache hole blocks only when a scheduled range fully covers the block;
  keep their bookkeeping separate from the ordinary cached frontier.
- Preserve the hash-based `evict_cached_token_range` test hook.
- Keep holes disabled by default and restrict support to one exact
  `FullAttentionSpec` / `FullAttentionManager` group with caching enabled,
  no DCP/PCP and no EAGLE. Other configurations retain prefix-only behavior.

Adaptations (upstream references below refer to tag `v0.32.0`, not fork line
numbers):

- `vllm/v1/core/kv_cache_coordinator.py:559` already establishes the unitary
  group's block size; `:565` requires hash block size to match it. Keep that
  validation and the upstream EAGLE group setup at `:572`, adding the original
  support predicate and hole lookup alongside them. Thus the original per-block
  hash indexing needs no new semantics for different hash/KV block sizes.
  Preserve hybrid managers and upstream partial-hit lookup unchanged.
- The old `can_fit_full_sequence` method is replaced by the
  `full_sequence_must_fit` path in `vllm/v1/core/kv_cache_manager.py:522`.
  Add hole allocations to both this admission check and the per-step check at
  upstream `:561`; preserve the watermark and reserved-block checks at `:573`.
  The ported full-sequence adjustment is at `kv_cache_manager.py:661`.
- Extend the zero-token validation at upstream `kv_cache_manager.py:474` with
  `not hole_ranges`, preserving upstream acceptance of computed tokens being
  adopted without new tail tokens. Append the new constructor/allocation
  arguments after existing v0.32.0 parameters. Preserve the ordinary lookup
  triple and connector lookup method; the separate hole lookup API retains the
  fork's four-result contract (`kv_cache_manager.py:368`).
- Record fresh hole block IDs using the upstream worker-zeroing predicate at
  `vllm/v1/core/single_type_kv_cache_manager.py:115`, rather than the fork's
  unconditional exact-full-attention check. This follows the same allocation
  tracking contract as v0.32.0 (`single_type_kv_cache_manager.py:443` in the port).
- Move pending-hole cleanup into `pop_blocks_for_free`, next to cached-frontier
  and partial-hit cleanup (upstream `single_type_kv_cache_manager.py:576`).
  Both direct frees and coordinator-mediated frees then clear hole bookkeeping.
- Keep v0.32.0's full-attention fine-grained lookup and tail caching intact;
  insert the separate `find_cache_hits_with_holes` class method at ported
  `single_type_kv_cache_manager.py:897` instead of replacing that upstream logic.

Checks performed:

```bash
PYTHONPYCACHEPREFIX=/tmp/vllm-port-cbb2490b6d-pycache /home/asd/LLMServingSim/.wt/testenv/bin/python -m py_compile vllm/v1/core/kv_cache_coordinator.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/single_type_kv_cache_manager.py
```

Result: PASS (exit 0). Used the brief's existing host test environment; no
packages were installed and no project modules were imported or executed.
Behavioral tests, independent verification, model evaluation, GPU jobs and
simulator runs were not performed; verification is assigned to the caller.
No tests were modified. No git command that writes was run; git's unmerged
index entries remain for the master to resolve by staging the edited files.

Questions: none for this commit. Hybrid/Mamba hole behavior remains outside the
original commit's supported scope; this port does not introduce it.

## ba53aee742 — Scheduler: per-request hole ranges and hole chunking

Status: adapted; ready for caller review, not independently verified. Original
commit: ba53aee74216af475500d75485b099bc8d13550c.
This section covers the stopped cherry-pick on `port/v0.32.0/holes-core`;
HEAD remains `188090aceea741d870e1a7a7e47e2f9a3e64364f` until the master
stages the resolution and continues the cherry-pick. AI assistance was used.
The branch, stopped files and original diff match this call's brief. Addendum 1
supersedes the brief's original git-write instructions; no git write was run.

Preserved fork behavior:

- Store sorted pending hole ranges on each request. Plan waiting and running
  requests front first through `plan_step`, then compute the tail. Allocate
  missing blocks in position order and cache only the scheduled hole pieces.
- Carry `scheduled_hole_ranges` alongside the total scheduled token count.
  Only tail work advances `num_computed_tokens`; subtract completed pieces
  from pending ranges and clear pending ranges when preempting a request.
  PP token-ID payloads exclude hole tokens already available in the prompt.
- Keep holes off by default, with the previous cache-manager support predicate;
  disable them for the V2 runner, EAGLE-family speculation, encoder inputs and
  connectors without `SupportsBlockHitMask`. Unknown masks defer admission;
  asynchronous masks are treated as misses with a warning once. Synchronous
  range loads are recorded with `update_state_after_alloc_with_ranges`.
- Retain the scheduler's `holes_evict_token_range` test hook.

Adaptations (upstream references refer to tag `v0.32.0`):

- Preserve all cache-manager constructor arguments at
  `vllm/v1/core/sched/scheduler.py:328`, including separate scheduler/hash block
  sizes, watermark and Mamba shared-prefix checkpoints; append `hole_config`.
  Upstream `:315` separates `use_eagle` from `use_eagle_block_drop`. Gate holes
  on `not self.use_eagle` as well as the manager predicate, preserving the
  fork's no-EAGLE restriction even when upstream disables trailing-block drops.
- Keep the upstream running-request cadence, DP prefill and encoder-connector
  guards at `scheduler.py:680`. Feed both scheduler and remaining worker-input
  budgets (minus drafting slots) to the planner, using upstream's effective
  long-prefill threshold at `:645`. Preserve the model-specific sampled-token
  context reservation at `:152` when bounding the planner's tail endpoint.
  Soft chunk limits can still extend to a segment end within the hard budget,
  as the fork planner specifies. Ordinary requests keep the upstream path.
- Preserve the upstream active-request admission limit at `scheduler.py:923`.
  Account for extra hole segments before admitting another request so later
  admissions cannot consume metadata entries already reserved by hole pieces;
  `plan_step` retains the fork's per-request segment cap.
- Keep `_get_local_prefix_cache_hit` at `scheduler.py:558` and its four-result
  hybrid/partial-prefix contract for ordinary requests. Add the separate
  four-result hole lookup ahead of it; do not replace the upstream lookup with
  the fork's old two-result unpack. Use the upstream queue helper at `:904`
  for an unavailable mask, preserving deferred-waiting and KV-holding queues.
- Upstream `vllm/v1/request.py:218` uses `PrefillStats`, not the fork's removed
  `Request.num_cached_tokens`. Populate it as at `scheduler.py:1075`, counting
  local hits as frontier minus recompute holes and connector-load ranges, and
  external hits as the load-range length. The existing hole manager records
  its own prefix-hit statistics, so do not also record the frontier as hits.
- Preserve DP prefill deferral at `scheduler.py:1113`, bounded SWA replay and
  ordinary Mamba/encoder/MTP scheduling. At the waiting allocation at `:1257`,
  pass only work beyond the adopted hit (subtract hole pieces and replay),
  retain full-sequence admission, reservation and zeroing arguments, and append
  `hole_ranges`. Count only tail work when tracking in-flight prefills and
  determining speculative-token counts (`:1377` and `:862`).
- Preserve every upstream output field at `scheduler.py:1516` and
  `vllm/v1/core/sched/output.py:299`, including zeroing, CoW, connector block
  state, dynamic speculation and encoder-manager metadata. Append the defaulted
  hole-range field; mark range loads in `has_sync_kv_loads` as well.
- Upstream `scheduler.py:1630` tracks all scheduled work in
  `num_in_flight_tokens`, and `:2082` drains the total scheduled count. Increment
  that counter before subtracting hole work for frontier advancement; otherwise
  a hole-only step would make the counter negative when its output returns.

Checks performed:

```bash
PYTHONPYCACHEPREFIX=/tmp/vllm-port-ba53aee742-pycache /home/asd/LLMServingSim/.wt/testenv/bin/python -I -m py_compile vllm/v1/core/sched/output.py vllm/v1/core/sched/scheduler.py vllm/v1/request.py
git diff --check HEAD -- vllm/v1/core/sched/output.py vllm/v1/core/sched/scheduler.py vllm/v1/request.py
```

Results: both PASS (exit 0, no diagnostics). The three Python files contain no
conflict markers. Used the brief's existing host test environment; no packages
were installed and no project modules or downloaded sources were executed.
Behavioral tests, independent verification, model evaluation, GPU jobs and
simulator runs were not performed; verification is assigned to the caller.
Worker consumption of the hole metadata belongs to subsequent fork commits
and is not established by syntax compilation. No tests were modified.
Git's unmerged index entries remain for the master to stage and continue.

Questions: none for this commit. No hybrid/Mamba or other unsupported hole
semantics were introduced. Implementation adaptations above are inferred from
the original fork contracts and pinned upstream structures; syntax/whitespace
checks are the only executed checks, not behavioral acceptance.

## 775f1ebad3 — Model runner: explicit positions and per-segment attention

Status: adapted; ready for caller review, not independently verified. Original
commit: 775f1ebad3b1cd258518c6f8039e592a52f1d639.
This section covers the stopped cherry-pick on `port/v0.32.0/holes-core`;
HEAD remains `03deb36689a6e4048789ad36ec62433e91700b75` until the master
stages the resolution and continues the cherry-pick. AI assistance was used.
Addendum 1 governs this call; no git command that writes was run.

Preserved fork behavior:

- Lay out each request's scheduled hole pieces in position order, followed by
  its tail. Gather token IDs using those explicit positions, copy the positions
  to the GPU before computing slot mappings, and subtract hole-token counts
  from both CPU optimistic and GPU request-level sequence lengths.
- Keep request-level query boundaries for logits and sampling; give attention
  one varlen entry per segment, with its end as sequence length and its
  request's block-table row repeated. Attach the `HoleSegments` view to common
  metadata. Keep the original M-RoPE / XD-RoPE rejection and single-group,
  no-DCP, no-microbatching assertions.
- Disable FULL CUDA graphs and microbatching on steps with holes. Preserve
  the existing scheduler/cache-manager support restrictions from prior ports.

Adaptations (upstream references refer to tag `v0.32.0`):

- Keep every upstream common-metadata field at
  `vllm/v1/attention/backend.py:444`–`:474`; add only the optional
  `hole_segments` field (`backend.py:473` in the port). Do not restore the
  deprecated `_seq_lens_cpu` / `_num_computed_tokens_cpu` fields removed
  upstream. In `vllm/v1/worker/hole_segments.py:126`, populate
  `seq_lens_cpu_upper_bound` with segment ends and clear both derived caches.
  Upstream `backend.py:486` derives computed-token counts from device sequence
  lengths and query boundaries; those now give each segment's start directly.
- Repeat optional row metadata alongside the block table: prefill flags,
  tensor causal flags, encoder/DCP fields, request indices, multimodal ranges,
  R-SWA prefix lengths and ReplaySSM origins. This preserves the row contracts
  at upstream `backend.py:415`–`:471` when a request becomes several segments;
  token-level positions and slot mappings keep their flattened token order.
  This transformation does not enable hybrid/Mamba or context-parallel holes.
- Preserve upstream's linear-attention import and metadata-builder handling;
  place the new helper import with worker imports. Keep upstream construction
  at `vllm/v1/worker/gpu_model_runner.py:2397`, then convert the common metadata
  to segments (`gpu_model_runner.py:2456` in the port).
- Upstream `gpu_model_runner.py:531` has removed cascade attention and
  `:3724` marks `use_cascade_attn` ignored. Do not restore the removed cascade
  computation. Append a defaulted `disable_full_cudagraph` argument to the
  execution/padding helper and apply its FULL exclusion in the dispatcher
  closure at upstream `:3762`, including the DP redispatch at `:3797`.
  Combine the hole guard with upstream's `_allow_microbatching` call at
  `:4038`; ordinary steps retain that upstream decision. The caller passes
  the segment view to attention construction (`gpu_model_runner.py:4208`
  in the port).

Checks performed:

```bash
PYTHONPYCACHEPREFIX=/tmp/vllm-port-775f1ebad3-pycache /home/asd/LLMServingSim/.wt/testenv/bin/python -I -m py_compile vllm/v1/attention/backend.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/hole_segments.py
git diff --check HEAD -- vllm/v1/attention/backend.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/hole_segments.py
```

Results: both PASS (exit 0, no diagnostics). No conflict markers remain in
these Python files. Used the brief's existing host test environment; no
packages were installed and no project modules or downloaded sources were
imported or executed. Behavioral tests, independent verification, model
evaluation, GPU jobs and simulator runs were not performed; verification is
assigned to the caller. No tests were modified. Syntax/whitespace checks do
not establish runtime correctness or backend compatibility. Git's unmerged
index entries remain for the master to stage and continue.

Questions: none for this commit. The adaptations follow the original segment
contract and v0.32.0 field/dispatch contracts; no new unsupported hole
semantics were selected.

## f935c3463b — Pluggable GPU block eviction with the session-state rate policy

Status: adapted; ready for caller review, not independently verified. Original
commit: f935c3463b7436432d16bc2516378f6ce70cc026.
This section covers the stopped cherry-pick on `port/v0.32.0/holes-rate`;
HEAD remains `3214ce8f8622432bdc6eb6f8b6e3096a714b4c54` until the master
stages the resolution and continues the cherry-pick. AI assistance was used.
Addendum 1 governs this call; no git command that writes was run.

Read / assumptions:

- Read the full task brief and Addendum 1, preceding port sections, original
  `git show f935c3463b`, the three conflicted files and the new `eviction.py`.
  The branch, HEAD and stopped paths match this call's premises.
- Rule-source provenance is `43ca3a4a99c3171401b04fa388d20f95aff6c043`;
  actual rule read revision is main `57fee9f3b14efcaf734141f2cd1ddec5c9cdcfdb`
  in `/home/asd/LLMServingSim`. Read the task-worker selector/card and Codex
  own rules, never-break section, implementation duties/pitfalls, applicable
  INTERFACE I.10–I.14/I.19, RECORDS I.8 and FORMATS report fields. The newer
  decisions retain master-only git writes and explicitly sequence the original
  single-group core/rate port before a separately designed multi-group extension.
- The adaptations below follow the original policy hooks and upstream block
  lifetimes. Their behavioral equivalence is inferred, not established by the
  syntax checks. No new eviction algorithm or hybrid semantics were selected.

Preserved fork behavior:

- LRU remains the default and installs no policy. Preserve `eviction.py`
  unchanged: configuration/environment overrides, empirical session rates,
  chunk-position recompute costs, owner tracking, whole-group eviction,
  uncached-first allocation, lazy heap versions and rescore cadence.
- Attach the policy only with prefix caching, one KV cache group, no connector
  and no DCP/PCP, as in the original commit. Retain the warning when hole-aware
  lookup is disabled. Arrival, first sampled token and request finish feed the
  session tracker; freeing a request records its block-table ownership.
- Keep allocation/touch/free/uncache/reset hooks and existing cache metrics,
  reference counts and reuse notifications.

Adaptations (upstream references refer to tag `v0.32.0`):

- Preserve `Callable` and the upstream reuse watchers in
  `vllm/v1/core/block_pool.py:666`–`:727`. Policy allocation still passes through
  `_notify_reuse` before blocks are handed out. Register newly free blocks from
  `unpin_blocks` (`:708`) with `on_free`; otherwise those blocks would count as
  free in the queue but remain absent from the policy's allocation candidates.
  This path is already upstream; no additional `holes/cp` commit is required
  for this resolution.
- Upstream `_remove_cached_block_hashes` at `block_pool.py:590` clears the
  primary and partial hashes together. In `_maybe_evict_cached_block` (`:729`),
  call `on_uncache` after successful removal and retain
  `_emit_block_removed_events`, including upstream group IDs and pool medium.
  Do not restore the fork's obsolete single-hash removal/event code.
- Preserve upstream `free_blocks` at `block_pool.py:774`: prepend uncached
  blocks, append cached blocks and route foreign blocks to their owning pool.
  Collect only this pool's newly free non-null blocks, in input order, for the
  policy hook after queue insertion. Thus policy-off queue behavior and
  cross-pool reference accounting remain upstream, while the policy receives
  the fork's free-notification ordering.
- Place `on_request_free` immediately after the request block table is popped
  in `vllm/v1/core/single_type_kv_cache_manager.py:562`. Keep physical freeing
  in the existing caller / `free` at `:581`, avoiding a double free and retaining
  coordinator-mediated deferred frees and previous hole-bookkeeping cleanup.
- Retain v0.32.0 speculative metrics and connector arrival handling in
  `vllm/v1/core/sched/scheduler.py:2616`, adding the policy arrival hook alongside
  them. Preserve stale-output handling at `:2166`, auxiliary-output cleanup and
  the upstream tuple return contract at `_free_request` (`:2714`). Add the fork's
  prefill/finish notifications without replacing these paths.

Checks performed:

```bash
PYTHONPYCACHEPREFIX=/tmp/vllm-port-f935c3463b-pycache /home/asd/LLMServingSim/.wt/testenv/bin/python -I -m py_compile vllm/v1/core/block_pool.py vllm/v1/core/eviction.py vllm/v1/core/sched/scheduler.py vllm/v1/core/single_type_kv_cache_manager.py
git diff --check HEAD -- vllm/v1/core/block_pool.py vllm/v1/core/eviction.py vllm/v1/core/sched/scheduler.py vllm/v1/core/single_type_kv_cache_manager.py
rg -n '^(<<<<<<<|=======|>>>>>>>)' vllm/v1/core/block_pool.py vllm/v1/core/eviction.py vllm/v1/core/sched/scheduler.py vllm/v1/core/single_type_kv_cache_manager.py
```

Verified results: compilation and whitespace check PASS (exit 0, no diagnostics);
marker search exit 1 with no matches. Used the existing host test environment
in isolated mode; no packages were installed or project modules imported.
No tests were modified. Behavioral tests, model evaluations, GPU jobs,
simulator runs and independent verification were not performed; the caller
owns verification. Syntax/whitespace checks do not establish runtime correctness.
Git's unmerged index entries remain for the master to stage and continue.

Questions: none for this commit. Multi-group/hybrid extensions remain a separate
follow-up under the current decision; this resolution preserves the original
single-group policy restriction.
