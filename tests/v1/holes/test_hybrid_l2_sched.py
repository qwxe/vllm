# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/M stage 4 (M note 7.2): scheduler side of hybrid + L2. CPU only.

A real Scheduler and OffloadingConnector over the mock spec/manager (see
``test_hybrid_replay._hyb_runner``). The gate itself stays on (``test_hooks``
bypasses it here); these tests cover the admission, park, resume and failure
paths.
"""

from tests.v1.holes import holes_utils as U
from tests.v1.holes.test_hybrid_replay import (
    BS,
    PROMPT,
    _evict_gpu,
    _hyb_runner,
    _prime_runner,
)
from tests.v1.holes.test_l2_holes import _estep, _key
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockLoadStoreSpec,
)
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.request import RequestStatus


def _l2_hybrid(l2_pages=(2,), evict=(2, 4)):
    """Hybrid (full+swa) runner primed with PROMPT; the GPU lost the Full
    pages ``evict``; L2 holds the pages ``l2_pages`` of request "a"."""
    r = _hyb_runner()
    a = _prime_runner(r)
    r.l2 = {_key(a, p) for p in l2_pages}
    r.manager.lookup.side_effect = lambda k, ctx: (
        LookupResult.HIT if k in r.l2 else LookupResult.MISS
    )
    b = U.make_request("b", PROMPT)
    for p in evict:
        _evict_gpu(r, b, p, 0)
    r.scheduler.add_request(b)
    return r, a, b


def _loads(r):
    return [
        sd
        for sd in r.offloading_spec.handler.transfer_specs.values()
        if isinstance(sd[0], MockLoadStoreSpec)
    ]


def test_admission_parks_the_request_on_an_l2_full_page_load():
    r, a, b = _l2_hybrid()
    out = _estep(r, complete=False)
    s = r.scheduler
    assert b.status is RequestStatus.WAITING_FOR_REMOTE_KVS
    assert "b" not in out.num_scheduled_tokens  # the plan is not started
    assert b.hybrid_plan is None
    assert b.hybrid_pending_loads == {0: [(2 * BS, 3 * BS)]}
    assert b.prefill_stats.num_external_cached_tokens == BS
    ((src, dst),) = _loads(r)
    assert src.offload_keys == [_key(a, 2)]
    blocks = s.kv_cache_manager.get_blocks("b").blocks[0]
    assert list(dst.block_ids) == [blocks[2].block_id]
    # the loaded page is not published before the load completes
    assert blocks[2].block_hash is None


def test_resume_after_the_load_replans_from_the_loaded_residency():
    r, a, b = _l2_hybrid()
    _estep(r, complete=False)
    _estep(r, complete=True)
    n_loads = len(_loads(r))
    outs = [_estep(r) for _ in range(3)]
    first = next(o for o in outs if "b" in o.num_scheduled_tokens)
    # page 2 is now a GPU hit; only page 4 is replayed
    assert first.scheduled_hole_ranges["b"] == [(4 * BS, 5 * BS)]
    assert first.num_scheduled_tokens["b"] == BS
    assert b.hybrid_pending_loads == {}
    assert len(_loads(r)) == n_loads  # nothing loaded twice
    m = r.scheduler.kv_cache_manager
    assert m.block_pool.get_cached_block(b.block_hashes[2], [0]) is not None


def test_failed_load_is_not_published_and_not_asked_again():
    r, a, b = _l2_hybrid()
    s = r.scheduler
    _estep(r, complete=False)
    s.recompute_kv_load_failures = True
    s.kv_holding_waiting.add_request(b)
    s._handle_failed_recving({"b"})
    assert "b" in s.failed_recving_kv_req_ids
    s.finished_recving_kv_req_ids.add("b")
    s._update_waiting_for_remote_kv(b)
    m = s.kv_cache_manager
    assert m.block_pool.get_cached_block(b.block_hashes[2], [0]) is None
    assert b.hybrid_pending_loads == {}
    assert b.num_computed_tokens == 0
    n_loads = len(_loads(r))
    b.status = RequestStatus.WAITING
    outs = [_estep(r) for _ in range(3)]
    first = next(o for o in outs if "b" in o.num_scheduled_tokens)
    # the failed page 2 became a hole (replayed), page 4 too
    assert (2 * BS, 3 * BS) in first.scheduled_hole_ranges["b"] or any(
        s_ <= 2 * BS < e_ for s_, e_ in first.scheduled_hole_ranges["b"]
    )
    assert len(_loads(r)) == n_loads


def test_no_l2_hit_leaves_admission_as_before():
    r, a, b = _l2_hybrid(l2_pages=())
    out = _estep(r)
    assert out.scheduled_hole_ranges["b"][0] == (2 * BS, 3 * BS)
    assert b.hybrid_pending_loads == {}
    assert _loads(r) == []
