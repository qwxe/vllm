# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes-v2 stage 1-2: the L2 (CPU OffloadingConnector) side of holes.

CPU only: a real Scheduler and OffloadingConnector over the mock spec and mock
manager of ``tests/v1/kv_connector/unit/offloading_connector/utils.py``.
"""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import RequestRunner


def _runner(holes: dict | None, **kw) -> RequestRunner:
    add = {"holes": holes} if holes is not None else None
    return RequestRunner(
        block_size=16, num_gpu_blocks=400, additional_config=add, **kw
    )


@pytest.mark.parametrize(
    "holes,expected",
    [
        (None, 1),  # holes not configured: min_reuse_blocks default 1
        ({"enabled": True}, 1),
        ({"enabled": True, "min_reuse_blocks": 3}, 3),
    ],
)
def test_min_reuse_reaches_manager(holes, expected):
    """Stage 1 (builder-r4 residual): the scheduler-side connector hands the
    hole config's ``min_reuse_blocks`` and the group kinds to the manager once,
    right after the manager is built."""
    r = _runner(holes)
    r.manager.set_group_kinds.assert_called_once_with(["full"], expected)


# ------------------------------------------------------------------ L commit 2: session and prefill-done signalling

from tests.v1.kv_connector.unit.offloading_connector.utils import (  # noqa: E402
    generate_store_output,
)
from vllm import SamplingParams  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (  # noqa: E402
    _create_req_context,
)

from . import holes_utils as U  # noqa: E402


def _step(runner, token=7, sample=True):
    out = runner.scheduler.schedule()
    mro = U.fake_output(out, token)
    if not sample:
        mro.sampled_token_ids = [[] for _ in mro.req_ids]
    runner.scheduler.update_from_output(out, mro)
    return out


def _runner_with_request(prompt_len=40, holes=None, **kw):
    r = _runner(holes, async_scheduling=False, **kw)
    r.manager.prepare_store.side_effect = lambda keys, ctx: generate_store_output(keys)
    r.scheduler.add_request(U.make_request("q", list(range(1, 1 + prompt_len))))
    return r


def test_req_context_carries_the_session_id():
    sp = SamplingParams(max_tokens=2, extra_args={"session_id": "sess-1"})
    req = U.make_request("a", [1, 2, 3])
    req.sampling_params = sp
    assert _create_req_context(req).session_id == "sess-1"
    assert _create_req_context(U.make_request("b", [1, 2, 3])).session_id is None


def test_t10_prefill_done_fires_once_with_the_first_sampled_token():
    r = _runner_with_request(prompt_len=1200)
    _step(r, sample=False)  # a prefill chunk with no sampled token
    assert r.manager.on_prefill_done.call_count == 0
    _step(r)
    assert r.manager.on_prefill_done.call_count == 1
    assert r.manager.on_prefill_done.call_args.args[0].req_id == "q"
    _step(r)  # decoding: not signalled again
    assert r.manager.on_prefill_done.call_count == 1


def test_t10_no_signal_for_a_request_that_samples_nothing_yet():
    r = _runner_with_request(prompt_len=2500)
    for _ in range(2):  # 1000 + 1000 of 2500 tokens: still prefilling
        _step(r, sample=False)
    assert r.manager.on_prefill_done.call_count == 0


# ------------------------------------------------------------------ L commit 3: activation gate (T11)

from vllm.distributed.kv_transfer.kv_connector.v1.hole_mask import (  # noqa: E402
    SupportsBlockHitMask,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (  # noqa: E402
    OffloadingConnector,
)


class _MaskOnly(SupportsBlockHitMask):
    """A mask-capable connector that is not the CPU offload connector."""

    def get_block_hit_mask(self, request, local_hit):
        return None, False

    def update_state_after_alloc_with_ranges(self, request, blocks, load_ranges):
        return None


class _L2Fake(OffloadingConnector, SupportsBlockHitMask):
    """An OffloadingConnector that carries the mask API (as after commit 4)."""

    def __init__(self):  # no real init: only isinstance matters here
        pass

    def get_block_hit_mask(self, request, local_hit):
        return None, False

    def update_state_after_alloc_with_ranges(self, request, blocks, load_ranges):
        return None


class _NoMask:
    pass


def connector_allows_holes(connector, extra):
    from vllm.distributed.kv_transfer.kv_connector.v1.hole_mask import (
        connector_allows_holes as gate,
    )

    return gate(connector, extra)


@pytest.mark.parametrize(
    "connector,extra,allowed",
    [
        (None, None, True),  # no connector: the existing path, unchanged
        (_MaskOnly(), None, True),  # other mask-capable connectors: unchanged
        (_NoMask(), None, False),
        (_L2Fake(), {"eviction_policy": "rate"}, True),
        (_L2Fake(), {"eviction_policy": "lru"}, False),
        (_L2Fake(), {"eviction_policy": "arc"}, False),
        (_L2Fake(), {}, False),  # absent = lru: holes + LRU is not a configuration
        (_L2Fake(), None, False),
    ],
)
def test_t11_activation_table(connector, extra, allowed):
    ok, reason = connector_allows_holes(connector, extra)
    assert ok is allowed
    assert (reason is None) == allowed


def test_t11_multiconnector_keeps_holes_off():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    mc = object.__new__(MultiConnector)  # not mask-capable in this stage [dep MC]
    assert connector_allows_holes(mc, {"eviction_policy": "rate"})[0] is False


@pytest.mark.parametrize("policy", [None, "lru", "arc", "rate"])
def test_t11_scheduler_with_the_offload_connector_stays_off_until_the_mask_api(policy):
    """Real scheduler + real OffloadingConnector over the mock spec: the
    connector has no mask API yet in this stage, so holes stay off for every
    policy (commit 4 turns them on for rate only)."""
    extra = {"eviction_policy": policy} if policy else None
    r = _runner({"enabled": True}, extra_config_overrides=extra)
    assert r.scheduler.holes_enabled is False


# ------------------------------------------------------------------ L commit 4: mask API, async park/resume, store cap (T12, T15)

from tests.v1.kv_connector.unit.offloading_connector.utils import (  # noqa: E402
    MockLoadStoreSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (  # noqa: E402
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.base import (  # noqa: E402
    LookupResult,
    make_offload_key,
)
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402

BS = U.BS
L2 = {"eviction_policy": "rate"}


def _l2_runner(holes=None, **kw):
    cfg = {"enabled": True, "test_hooks": True}
    cfg.update(holes or {})
    r = _runner(cfg, extra_config_overrides=L2, async_scheduling=False, **kw)
    assert r.scheduler.holes_enabled
    r.offered = []

    def store(keys, ctx):
        keys = list(keys)
        r.offered.append(keys)
        return generate_store_output(keys)

    r.manager.prepare_store.side_effect = store
    r.l2 = set()
    r.manager.lookup.side_effect = lambda k, ctx: (
        LookupResult.HIT if k in r.l2 else LookupResult.MISS
    )
    return r


def _key(req, i):
    return make_offload_key(req.block_hashes[i], 0)


def _estep(r, complete=True):
    """One engine step: scheduler, worker connector, model runner output."""
    s = r.scheduler
    out = s.schedule()
    r._update_gpu_blocks()
    md = out.kv_connector_metadata
    r.worker_connector.handle_preemptions(md)
    r.worker_connector.bind_connector_metadata(md)
    r.worker_connector.start_load_kv(r._dummy_ctx)
    if complete:
        r.offloading_spec.complete_transfers()
    r.worker_connector.wait_for_save()
    fs, fr = r.worker_connector.get_finished(out.finished_req_ids)
    wm = r.worker_connector.build_connector_worker_meta() or OffloadingWorkerMetadata()
    r.worker_connector.clear_connector_metadata()
    mro = U.fake_output(out)
    for i, rid in enumerate(mro.req_ids):  # the runner discards partial prefills
        q = s.requests[rid]
        if q.num_computed_tokens < q.num_tokens or q.pending_hole_ranges:
            mro.sampled_token_ids[i] = []
    mro.kv_connector_output = KVConnectorOutput(
        finished_sending=fs, finished_recving=fr, kv_connector_worker_meta=wm
    )
    s.update_from_output(out, mro)
    return out


def _fill_gpu_cache(r, prompt):
    a = U.make_request("a", prompt)
    r.scheduler.add_request(a)
    while "a" in r.scheduler.requests:
        _estep(r)
    return a


PROMPT = list(range(1, 1 + 4 * BS + 5))  # 4 full blocks + 5 tokens


def _park_setup(holes=None):
    r = _l2_runner(holes)
    a = _fill_gpu_cache(r, PROMPT)
    probe = U.make_request("probe", PROMPT)
    r.scheduler.holes_evict_token_range(probe, 0, 3 * BS)  # GPU keeps block 3 only
    r.l2 = {_key(a, 0), _key(a, 2), _key(a, 3)}
    r.scheduler.kv_cache_manager.make_prefix_cache_stats()
    b = U.make_request("b", PROMPT)
    r.scheduler.add_request(b)
    return r, a, b


def test_connector_has_the_mask_api_and_needs_single_block_chunks_with_holes():
    r = _l2_runner()
    assert isinstance(r.scheduler_connector, SupportsBlockHitMask)
    with pytest.raises(ValueError):
        _runner({"enabled": True}, blocks_per_chunk=2)
    # holes not configured: unchanged, any chunking
    _runner(None, blocks_per_chunk=2)
    # the check does not depend on the policy being effective
    with pytest.raises(ValueError):
        _runner({"enabled": True}, blocks_per_chunk=2, extra_config_overrides={})


def test_t12_mask_and_async_park_keep_hole_and_load_ranges():
    """L2 holds {0,2,3}, the GPU holds {3}: mask [T,F,T,local]; the request
    parks with pending_hole_ranges and pending_load_ranges; nothing is cached
    for the not-yet-loaded blocks."""
    r, a, b = _park_setup()
    mask, is_async = r.scheduler_connector.get_block_hit_mask(
        b, [False, False, False, True]
    )
    assert is_async is True and mask[:3] == [True, False, True]
    _estep(r, complete=False)
    assert b.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    assert b.num_computed_tokens == 4 * BS
    assert b.pending_hole_ranges == [(BS, 2 * BS)]
    assert b.pending_load_ranges == [(0, BS), (2 * BS, 3 * BS)]
    blocks = r.scheduler.kv_cache_manager.get_blocks("b").blocks[0]
    assert [x.block_hash is not None for x in blocks[:4]] == [False, False, False, True]
    # one load job: keys 0 and 2 into the allocated blocks 0 and 2
    ((src, dst),) = r.offloading_spec.handler.transfer_specs.values()
    assert isinstance(src, MockLoadStoreSpec)
    assert src.offload_keys == [_key(a, 0), _key(a, 2)]
    assert list(dst.block_ids) == [blocks[0].block_id, blocks[2].block_id]


def test_t12_completion_caches_only_loaded_ranges_and_failure_makes_holes():
    r, a, b = _park_setup()
    s = r.scheduler
    _estep(r, complete=False)
    blocks = s.kv_cache_manager.get_blocks("b").blocks[0]
    # block 0's load fails: its range becomes a hole, the frontier is unchanged
    s._update_requests_with_invalid_blocks([b], {blocks[0].block_id}, {}, evict_blocks=False)
    assert b.num_computed_tokens == 4 * BS
    assert b.pending_hole_ranges == [(0, 2 * BS)]
    assert b.pending_load_ranges == [(2 * BS, 3 * BS)]
    s.finished_recving_kv_req_ids.add("b")
    s._update_waiting_for_remote_kv(b)
    assert [x.block_hash is not None for x in blocks[:4]] == [False, False, True, True]
    assert b.pending_load_ranges == []


def test_t12_resume_plans_the_hole_without_a_second_lookup():
    r, a, b = _park_setup()
    s = r.scheduler
    _estep(r, complete=False)
    _estep(r, complete=True)  # the load finishes; b is promoted next step
    n_lookup = r.manager.lookup.call_count
    out = _estep(r)
    assert out.scheduled_hole_ranges == {"b": [(BS, 2 * BS)]}
    assert out.num_scheduled_tokens["b"] == BS + 5  # hole piece + tail
    assert r.manager.lookup.call_count == n_lookup  # no new lookup on resume
    assert b.pending_hole_ranges == [] and b.pending_load_ranges == []
    assert b.num_output_tokens == 1
    blocks = s.kv_cache_manager.get_blocks("b").blocks[0]
    assert all(x.block_hash is not None for x in blocks[:4])
    # one prefix-cache record for the admission, not two
    assert s.kv_cache_manager.prefix_cache_stats.requests == 1


def _store_cap_check(r, req_id="b"):
    """Offers of each step must stay below the contiguous computed range."""
    s = r.scheduler
    req = s.requests[req_id]
    n = len(r.offered)
    out = _estep(r)
    remaining = req.pending_hole_ranges
    cap_tokens = remaining[0][0] if remaining else req.num_computed_tokens
    for keys in r.offered[n:]:
        for k in keys:
            idx = [i for i in range(len(req.block_hashes)) if _key(req, i) == k]
            assert idx and (idx[0] + 1) * BS <= cap_tokens, (idx, cap_tokens)
    return out


def test_t15_store_offer_never_runs_past_an_uncomputed_hole():
    r = _l2_runner({"step_budget": BS, "min_piece": BS})
    a = _fill_gpu_cache(r, PROMPT)
    probe = U.make_request("probe", PROMPT)
    r.scheduler.holes_evict_token_range(probe, BS, 3 * BS)  # GPU keeps 0 and 3
    b = U.make_request("b", PROMPT)
    r.scheduler.add_request(b)
    r.offered.clear()
    hole_keys = {_key(b, 1), _key(b, 2)}
    steps = 0
    while "b" in r.scheduler.requests and steps < 12 and b.num_output_tokens < 2:
        _store_cap_check(r)
        steps += 1
    assert steps >= 2
    # after step 1 the second hole block must not have been offered yet
    flat = [k for keys in r.offered for k in keys]
    assert hole_keys <= set(flat)  # both hole blocks are offered in the end


def test_t15_abort_mid_hole_offers_nothing_past_the_first_pending_hole():
    r = _l2_runner({"step_budget": BS, "min_piece": BS})
    a = _fill_gpu_cache(r, PROMPT)
    probe = U.make_request("probe", PROMPT)
    r.scheduler.holes_evict_token_range(probe, BS, 3 * BS)
    b = U.make_request("b", PROMPT)
    r.scheduler.add_request(b)
    r.offered.clear()
    _store_cap_check(r)  # step 1: first hole piece scheduled, second pending
    assert b.pending_hole_ranges, "the test needs a hole left after step 1"
    first_pending = b.pending_hole_ranges[0][0]
    r.scheduler.finish_requests("b", RequestStatus.FINISHED_ABORTED)
    r.offered.clear()
    _estep(r)
    for keys in r.offered:
        for k in keys:
            idx = [i for i in range(len(b.block_hashes)) if _key(b, i) == k][0]
            assert (idx + 1) * BS <= first_pending
