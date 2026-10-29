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
