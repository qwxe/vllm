# vLLM v0.32.0 fork port

Status: ready for caller review; runtime correctness is unverified. AI assistance
was used. Base: v0.32.0 (`5df63a7203cd`). This record covers the original
single-group core/kernel/rate port and the Addendum 2 round-1 working-tree fixes.
The caller owns commits, branch propagation and independent verification.

## Commit inventory

References prefixed `upstream` are file:line at tag v0.32.0. New files have no
upstream counterpart. Port hashes identify the existing commits before the
round-1 working-tree changes; the caller will commit these fixes separately.
Historical test/benchmark numbers in the carried design documents describe the
original fork, not this port.

| Original commit and subject | Port commit | Status and adaptation |
|---|---|---|
| `72946a4c8e` Add hole config and the chunk planner | `63ca7eb8f3` | Ported as-is: new `vllm/v1/core/holes.py`, default off. |
| `cbb2490b6d` KV cache manager: lookup past misses, in-place hole blocks | `0b2326af6d` | Adapted to the coordinator, admission, worker-zeroing and free lifecycle; details below. Round 1 moves hole cache stats to admission. |
| `1e0690c75a` KV connector API: per-block hit mask (design + stub) | `188090acee` | Ported as-is: separate `SupportsBlockHitMask` interface; existing connector APIs retained. |
| `ba53aee742` Scheduler: per-request hole ranges and hole chunking | `03deb36689` | Adapted to scheduling/input budgets, context reservation, statistics and in-flight accounting; details below. Round 1 forces V1 at config selection. |
| `775f1ebad3` Model runner: explicit positions and per-segment attention | `54ba7364d2` | Adapted to common metadata and graph dispatch; details below. Round 1 removes the obsolete XD-RoPE attribute access. |
| `f8099d814b` holes: tests and design notes for the hole-aware core | `cfebdf06a5` | Initially as-is; adapted in round 1 for the manager lookup triple, scheduler block size, admission allocation and emitted PrefillStats. Every existing assertion retains its behavioral meaning. |
| `18fb65769f` holes: record test results from the worktree | `05d40181da` | Ported as-is: historical results only, not evidence for v0.32.0. |
| `65f27846f1` holes: add SPDX headers and drop unused locals in the GPU e2e | `4b0c9694a6` | Ported as-is. |
| `8ccc311181` Add MSA attention backend: hole-aware position-explicit Triton work-list kernel + planner | `b2e4624467` | Initially as-is; adapted in round 1 for exact device lengths, packed KV layout, async speculative decode rejection and explicit non-causal/R-SWA rejection; details below. |
| `87817010b7` MSA: fp32 support, dtype-aware launch config | `98403f3d86` | Ported as-is: dtype support and launch configuration retained. |
| `d5da837013` Add KERNEL-IMPL.md (MSA backend design, CP API, tests, 4050 benchmarks) | `eb0134f7cb` | Ported as-is: historical design/results; CP API text does not enable CP in this port. |
| `b773a64da9` KERNEL-IMPL.md: final test counts | `e7730237c9` | Ported as-is: historical counts only. |
| `425a6ae813` MSA: drop the machine path from the synced-file headers and a dead n_tiles | `e96faaa2d1` | Ported as-is. |
| `f935c3463b` Add pluggable GPU block eviction with the session-state rate policy | `efeb5fb32d` | Adapted to upstream block hashes, queue ordering, cross-pool freeing and scheduler lifecycle. Round 1 prefers native Request.session_id; details below. |
| `c6d20e9c4d` Test the rate eviction policy against the simulator's SessRatePool | `123ef20a5d` | Initially as-is; round 1 captures emitted PrefillStats instead of removed Request.num_cached_tokens; hit-count expectations retained. |
| `bf6afedd94` Add the rate-vs-LRU GPU end-to-end on a scaled codex session workload | `e82773f38d` | Ported as-is; no GPU run performed. |
| `32c44343b9` Add RATE-IMPL.md: rate eviction design as built, tests and 4050 results | `efb85622ae` | Ported as-is: historical results only. |
| `e4aff3cb88` holes/rate: dataset as a flag, run-dependent e2e results, SPDX and unused import | `922f4a9f3a` | Ported as-is: dataset flag and historical-result qualifications retained. |

`3214ce8f86` merges the ported kernel into rate, replacing the original CP-mediated
kernel/core integration. `83909ffe00` adds this port record; it is not an original
fork patch. Core and kernel each originate at v0.32.0.

## Core adaptations

### cbb2490b6d — lookup and allocation

- Preserve one exact FullAttentionSpec/FullAttentionManager group, caching on,
  no DCP/PCP, no EAGLE. Upstream `vllm/v1/core/kv_cache_coordinator.py:559`
  establishes block size and `:565` validates hash alignment. Keep hybrid
  partial-prefix lookup and EAGLE setup (`:572`) unchanged.
- Keep the ordinary three-result lookup and the connector lookup API. The new
  hole lookup returns blocks, frontier, recompute ranges and load ranges.
  Preserve minimum reuse runs and final-token recomputation.
- Replace removed `can_fit_full_sequence` with hole block accounting in both
  `allocate_slots(full_sequence_must_fit=True)` (upstream
  `vllm/v1/core/kv_cache_manager.py:522`) and per-step admission (`:561`),
  retaining upstream watermark/reservations (`:573`). Accept hole-only work
  in the zero-token validation (`:474`).
- Track new hole blocks through the worker-zeroing predicate (upstream
  `vllm/v1/core/single_type_kv_cache_manager.py:115`). Clear pending holes in
  `pop_blocks_for_free` (`:576`), including coordinator-mediated frees.
  Cache only the scheduled hole pieces; keep ordinary partial-hit/CoW logic.
- Round 1: hole lookup no longer records prefix stats. The scheduler records
  one query at successful admission, matching upstream
  `vllm/v1/core/sched/scheduler.py:1475`; local hits exclude recompute and
  connector-load ranges. Retried or rejected lookups add no query.

### ba53aee742 — scheduling

- Retain scheduler/hash block sizes, watermark and Mamba checkpoint arguments
  at upstream `vllm/v1/core/sched/scheduler.py:328`. Gate on `not use_eagle`,
  including when EAGLE block drop is disabled (`:315`).
- Preserve running-request cadence, DP/encoder guards (`:680`), effective
  long-prefill threshold (`:645`) and sampled-token context reservation
  (`:152`). Bound the planner by scheduler and worker-input budgets minus
  drafting slots; reserve metadata entries for extra segments before further
  admissions (`:923`). Only tail work advances the request frontier.
- Keep ordinary `_get_local_prefix_cache_hit` (`:558`) and deferred queue
  handling (`:904`). Keep unknown connector masks deferred, async masks as
  misses, synchronous range loads and per-request in-flight accounting.
- Use PrefillStats for local/external cache hits; keep v0.32.0 output and
  speculative metrics. Preemption clears pending holes.
- Round 1: `VllmConfig.use_v2_model_runner` (upstream
  `vllm/config/vllm.py:703`) reads the same HoleConfig as the scheduler.
  Enabled holes force V1, including an explicit V2 environment override, with
  an info-once log. HiSparse/watermarking require V2 (`:704`, `:712`), so their
  combination with enabled holes raises an explicit configuration error.
  With holes off, upstream runner selection remains unchanged.

### 775f1ebad3 — runner and segment metadata

- Preserve every upstream common metadata field (upstream
  `vllm/v1/attention/backend.py:415-474`), adding an optional hole_segments.
  Use segment ends for seq_lens_cpu_upper_bound; clear derived caches.
  Computed-token counts derive from exact device lengths/query boundaries
  (`:486`); removed CPU fields are not restored.
- Repeat row metadata with the block table: prefill/causal flags, encoder/DCP
  fields, request indices, multimodal ranges, R-SWA lengths and ReplaySSM
  origins. Positions/slot mappings retain token order. In particular, the
  added req_idx fallback is `seg.seg_req_idx.copy()` when cm.req_idx is None;
  the encoder_seq_lens selection is also added, though downstream group
  handling overwrites it. These are listed for r1 m2, not new hybrid support.
- Keep upstream metadata construction (upstream
  `vllm/v1/worker/gpu_model_runner.py:2397`), then expand segments. Keep logits
  and sampling boundaries at request granularity.
- Cascade was removed (`:531`, `:3724`); exclude FULL graphs in the dispatcher
  (`:3762`), including DP redispatch (`:3797`). Combine the hole guard with
  upstream microbatching decisions (`:4038`).
- Round 1: v0.32.0 retains uses_mrope (`:542`) but has no uses_xdrope_dim or
  XD-RoPE runner path. Reject M-RoPE using the existing attribute; remove the
  obsolete attribute access instead of reintroducing removed upstream state.
  Single-group/no-DCP/no-microbatch hole restrictions remain.

## Kernel adaptations — 8ccc311181, round 1

- Ordinary rows use `cm.seq_lens[:num_reqs].cpu()` for exact lengths. Upstream
  `vllm/v1/attention/backend.py:444` calls the CPU copy an upper bound under
  async speculative decode. Do not use it or restore removed _seq_lens_cpu.
  Hole rows continue to use their explicit segment ends.
- Reject async scheduling combined with positive speculative-token count in
  the MSA metadata builder, matching the runner's async-spec predicate
  (upstream `vllm/v1/worker/gpu_model_runner.py:683`). This follows the binding
  r1 B2 decision and makes the host synchronization explicit.
- Read the inherited packed cache `[blocks, kv_heads, block_size, 2*head_size]`
  with `transpose(1, 2).split(head_size, dim=-1)`, matching upstream
  `vllm/v1/attention/backends/triton_attn.py:795`. The resulting K/V views are
  `[blocks, block_size, kv_heads, head_size]`; MSA already passes their strides
  to its kernels. Do not unbind the head dimension.
- Override supports_non_causal and supports_rswa to False, instead of
  inheriting upstream True (`triton_attn.py:327`, `:335`). Also reject
  non-causal row metadata and non-None R-SWA lengths in build. The original
  causal kernel remains unchanged.

## Rate adaptations — f935c3463b

- Default LRU installs no policy. Preserve the original rate/group scoring,
  ownership, lazy versions, session tracker and uncached-first allocation.
  Preserve prefix caching/single group/no connector/no DCP/PCP restrictions.
- Preserve reuse watchers (upstream `vllm/v1/core/block_pool.py:666-727`) and
  register unpin_blocks frees (`:708`) with the policy. Remove primary and
  partial hashes through upstream `_remove_cached_block_hashes` (`:590`),
  then call on_uncache and retain removal events/group IDs (`:729`).
- Keep upstream free_blocks (`:774`): prepend uncached blocks, append cached
  blocks and route foreign blocks to their owning pool. Notify only this
  pool's newly free non-null blocks after insertion, in original input order.
- Register request ownership when its table is popped (upstream
  `vllm/v1/core/single_type_kv_cache_manager.py:562`); leave physical freeing
  to its existing caller (`:581`). Preserve scheduler stale-output handling
  (`vllm/v1/core/sched/scheduler.py:2166`), arrivals (`:2616`), cleanup and the
  _free_request tuple contract (`:2714`).
- Round 1: session_id_of uses Request.session_id when non-None, including an
  empty string; otherwise use sampling_params.extra_args["session_id"].
  This follows the r1 m5 decision and upstream `vllm/v1/request.py:197`.

## CP carried and dropped work

CP commits carried: **none**. Rate uses the core scheduler and block-pool hooks;
its original CP ancestry adds no required rate dependency. v0.32.0 already has
unpin_blocks/reuse watchers and cross-pool freeing. MSA comes directly from
holes/kernel. CP implementations, placement options and benchmarks remain out
of scope, as do server/MLA and the separately approved multi-group extension.

The following original CP-only commits are dropped from this port:

| Original | Reason |
|---|---|
| `f2d9739ebf` | CP schemes/reference local attention. |
| `11ea11ec22` | CP PCP slot mapping/tests. |
| `16b5e6d816` | CP tiled PyTorch attention benchmark. |
| `97a4ffbd75` | CP_HOLES backend glue. |
| `9e5a2a31b2` | CP gloo glue tests. |
| `4b6110d182` | Multi-GPU CP checks/benchmarks. |
| `ed75704555` | CP benchmark layout fixes. |
| `1af1bfb6ca` | CP partial-state combine. |
| `f1998e3f8f` | CP plan index device copies. |
| `90745e4518` | CP implementation/server plan documentation. |
| `2144ce6f33` | CP merge of core; core is carried directly. |
| `7946f9a935` | CP consumption of core segments. |
| `17056ca2cf` | CP fork/core documentation. |
| `755188b3c9` | CP KV placement option. |
| `81a19c121e` | CP sharded dense layers. |
| `1119d5affd` | CP local-attention selection/checks. |
| `2692b859e7` | CP-specific planner budgets. |
| `a796e94b71` | CP KV-slot validity. |
| `87f6bfee39` | CP merge of kernel; kernel is carried directly. |
| `148d29dde4` | CP local MSA attention. |
| `2c8adc24df` | CP MSA benchmarks. |
| `ade5d6490f` | CP placement/sharding documentation. |
| `f9373161d5` | CP merge of kernel cleanup; cleanup carried directly. |
| `f5725b5ef1` | CP merge of core cleanup; cleanup carried directly. |
| `831801da62` | CP-only SPDX/import cleanup. |
| `053addf171` | CP merge into rate; replaced by direct core/kernel integration. |

Non-qwxe upstream commits in the old ancestry are not re-applied: `b92312dfd7`,
`12449f9492`, `69b8bd4b33`, `bcc0fdd0f3`, `268bed9cf3`, `0ee3b7fc3d`,
`1dbbafd3f3`, `29982d48b3`, `3a30a1a6a8`, `c284a6671c`, `cfad6a509c`,
`8adcf8c40a`, `2a69949bda`. They are upstream/superseded, per the brief;
v0.32.0 remains authoritative.

## Round-1 finding disposition

| Finding | Working-tree disposition |
|---|---|
| Sol 1 / Opus B2 | Exact device lengths; async speculative decode explicitly rejected. |
| Sol 2 / Opus B1 | Packed KV split matching v0.32.0. |
| Opus B3 | Removed obsolete uses_xdrope_dim access; retained M-RoPE rejection. |
| Sol 3 / Opus B4 | Tests use scheduler_block_size, lookup triple, full_sequence_must_fit allocation and emitted PrefillStats. No assertion deleted, loosened or skipped. |
| Sol 4 | Non-causal and R-SWA capability flags false, metadata explicitly rejected. |
| Opus M1 | Logged forced V1 at config selection; V2-only features conflict explicitly. |
| Opus M2 | Complete inventory, CP disposition, check limits and Q-list in this record; stale stopped-cherry-pick statements removed. |
| m1 | Hole cache statistics moved from lookup to admission, excluding holes and external loads. |
| m2 | Added req_idx fallback and encoder_seq_lens selection listed above; no code change. |
| m3 | Record only: planner tail is capped by min(num_tokens, max_model_len - num_sampled_tokens_per_step); budget by min(token_budget, input_budget - draft_slots). The _reserve_prefill_lookahead miscount is a no-op because hole lookahead is zero. |
| m4 | Record only: rate still treats hash-less unpin_blocks frees as first candidates; upstream uses these as last resort for HiSparse. HiSparse is excluded by the original hole gate and now explicitly conflicts with forced V1. No policy algorithm change. |
| m5 | Native session_id first, legacy fallback only when unset. |
| m6 | Record only: existing subjects lack the requested '(ported to v0.32.0)' suffix except the merge. No Git history write is authorized; caller owns history disposition. |
| m7 | Record only: MSA is the fork hole-aware backend; upstream CUTLASS_MSA/TRITON_MSA refer to MiniMax sparse attention. Enum names do not collide; no rename/interface change. |

## Checks and could-not-check

The only execution in this call is isolated host-interpreter py_compile on all
touched Python files; no project modules or downloaded sources are imported or
executed. The existing host test environment is
`/home/asd/LLMServingSim/.wt/testenv`; no packages are installed. Exact command and
results are recorded below after completion. Read-only `git diff --check` is
also run. Syntax and whitespace results do not establish runtime correctness.

Tests adapted: manager/scheduler API calls, emitted cache statistics capture in
holes_utils.step and the rate trace, and the fake mask connector's upstream
on_new_request no-op hook. Original cache-hit counts, segment ordering,
frontiers, admission success and no-hole identity assertions are retained.
Added regressions cover lookup statistics versus admission, forced V1 despite
explicit V2, exact MSA lengths despite optimistic bounds, K/V views with 1/2/4
heads, and native-session precedence/fallback. These tests are unexecuted here.

Not verified: CPU behavioral tests/imports, MSA GPU numerics, kernel performance,
DP graph redispatch, PIECEWISE padding, per-segment FA/Triton builders, manager
versus MSA block size, prompt-embed positions, connector load execution,
preemption/runtime scheduling, model output/evaluations and end-to-end GPU/rate
runs. No simulator, GPU, container, profiler, benchmark, download or agent is
launched. The host lacks a compatible torch/vLLM test installation; the brief
assigns independent verification to the caller. No baseline behavioral suite or
quick golden run is claimed. Prior per-commit syntax-check records do not prove
this round's runtime behavior or acceptance.

## Q-list

Open questions: **none for this authorized single-group round**.
The binding decision row 'Fork port r1 answers 2026-10-28' settles forced V1,
exact GPU lengths plus async-spec rejection, and native session precedence.
V2 holes and multi-group/hybrid Mamba/SWA holes remain separately scoped work;
this port retains its original support gate and does not select their semantics.

## Round-1 worker receipt

Status: DONE_WITH_CONCERNS (ready for caller review, not acceptance).

Done: all blocking/major findings and m1/m5 have working-tree fixes; m2 is
listed, and m3/m4/m6/m7 are recorded only, as Addendum 2 directs.
Not done: runtime tests and independent verification; caller-owned Git commits
and propagation to core/kernel were not attempted.

Read: the complete task brief including both addenda and quoted 0.3; both r1
reviews; cited RESEARCH II.1 decisions including the r1 answers and superseded
entries; local AGENTS.md; the task-worker selector/card and CODEX own rules;
INTERFACE I.10, I.12, I.13, I.14 and I.19; IMPLEMENTER duties/pitfalls;
RECORDS I.8 and FORMATS I.24 report fields; affected source/tests and the previous
port record. No newer named handoff is supplied beyond the reviews/decision row.

Assumptions: brief rule-source commit is `43ca3a4a9`; actual main read revision
is `1a2ddcf5b8c99329a9bfdce5888d77c821858a08`, matching the caller's current
revision. Addendum 2 and the current r1 decision row govern over the original
branch-creation instructions. Read-only checks confirmed a clean starting tree,
rate HEAD `83909ffe0030b714641b6ec5a1c00c94efde8f05`, core
`4b0c9694a6772768496c1c7828cea52ff045ffda`, kernel
`e96faaa2d16366e333ca25405fa473e2d1a00e19`. No Git write was run.
The explicit error for holes plus HiSparse/watermarking follows their conflicting
V1/V2 requirements; silently selecting either runner would violate one of them.
The test helper optionally captures emitted PrefillStats and SchedulerStats
because update_from_output transfers/resets them. This is test-only plumbing;
incorrect capture would cause the cache-hit regressions to fail. Replacing the
obsolete XD-RoPE check follows its complete removal upstream, not new position
semantics. No other behavior/scope decision was selected.

Verified (syntax/whitespace only): the following exact compilation command
returned exit 0 with no diagnostics, including after the final test changes:

```bash
PYTHONPYCACHEPREFIX=/tmp/vllm-port-r1-pycache /home/asd/LLMServingSim/.wt/testenv/bin/python -I -m py_compile tests/v1/holes/holes_utils.py tests/v1/holes/test_holes_kv.py tests/v1/holes/test_holes_segments.py tests/v1/holes/test_rate_eviction.py vllm/config/vllm.py vllm/v1/attention/backends/msa_attn.py vllm/v1/core/eviction.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/sched/scheduler.py vllm/v1/worker/gpu_model_runner.py
git diff --check
```

`git diff --check` also returned exit 0 without diagnostics. Filesystem checks
found neither torch nor vLLM in the host testenv site-packages. No project import
was attempted. Behavioral equivalence is inferred from the adaptations, not
verified by these checks.

Test changes: existing assertions preserved through the API adaptations described
above; new regressions additionally exercise explicit MSA non-causal/R-SWA and
async-spec rejection. No test was deleted, loosened or skipped. None was run.

Files changed (all under /home/asd/vllm-port):

- PORT-v0.32.0.md
- tests/v1/holes/holes_utils.py
- tests/v1/holes/test_holes_kv.py
- tests/v1/holes/test_holes_segments.py
- tests/v1/holes/test_rate_eviction.py
- vllm/config/vllm.py
- vllm/v1/attention/backends/msa_attn.py
- vllm/v1/core/eviction.py
- vllm/v1/core/kv_cache_manager.py
- vllm/v1/core/sched/scheduler.py
- vllm/v1/worker/gpu_model_runner.py

Escalations: none. Rule use: phase-1 not active. Not verified: all runtime and
independent-review items in the could-not-check section. There are no commits
from this call; every fix remains unstaged in the working tree for the caller.
