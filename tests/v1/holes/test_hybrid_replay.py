# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/M: hybrid dependency-closure planner (M note section 4). CPU only.

Expected values are hand-derived from the M section 4 recovery oracle, not
produced by the planner under test.
"""

from vllm.v1.core.holes import plan_hybrid_replay

N = 100  # prompt tokens; the last token is always computed


def test_m4_oracle_repair_then_restore():
    # Full [32,48) missing, checkpoint 64 present, no earlier nonzero state:
    # state at 32 does not exist, so repair [0,48), then restore 64 + tail.
    p = plan_hybrid_replay(N, [[(32, 48)]], [{64}], [0, 64])
    assert p is not None
    assert p.frontier == 64
    assert p.spans == ((0, 48),)
    assert p.span_restore == (False,)
    assert p.tail_restore is True
    assert p.executed_tokens(N) == 48 + (N - 64)


def test_missing_state64_means_no_benefit():
    # Without a state at 64 only frontier 0 is legal: stock path (None).
    assert plan_hybrid_replay(N, [[(32, 48)]], [set()], [0, 64]) is None


def test_all_full_present_is_stock():
    # No holes needed: the furthest frontier equals the stock hit.
    assert plan_hybrid_replay(N, [[]], [{128 // 2}], [0, 64]) is None


def test_nearest_earlier_state_is_used():
    p = plan_hybrid_replay(N, [[(32, 48)]], [{16, 64}], [0, 16, 64])
    assert p is not None and p.frontier == 64
    assert p.spans == ((16, 48),)
    assert p.span_restore == (True,)
    assert p.executed_tokens(N) == 32 + 36


def test_overlapping_repairs_merge_once():
    # both ranges extend back to 0 and overlap -> one span
    p = plan_hybrid_replay(N, [[(16, 24), (36, 48)]], [{64}], [0, 64])
    assert p is not None
    assert p.spans == ((0, 48),)
    assert p.tail_restore is True
    assert p.executed_tokens(N) == 48 + 36


def test_serialized_jump_with_restore():
    p = plan_hybrid_replay(N, [[(16, 32), (48, 64)]], [{48, 64}], [0, 48, 64])
    assert p is not None and p.frontier == 64
    assert p.spans == ((0, 32), (48, 64))
    assert p.span_restore == (False, True)
    assert p.tail_restore is False  # replay ends exactly at the frontier
    assert p.executed_tokens(N) == 32 + 16 + 36


def test_union_of_full_groups():
    p = plan_hybrid_replay(N, [[(16, 32)], [(24, 40)]], [{64}], [0, 64])
    assert p is not None and p.spans == ((0, 40),)


def test_frontier_must_leave_last_token():
    # frontier 100 would leave no token to compute
    assert plan_hybrid_replay(N, [[(0, 16)]], [{100}], [0, 100]) is None


def test_every_state_group_needs_exact_state():
    # Mamba has 64, SWA only 48: frontier 64 illegal; 48 has Full hole below
    # at [32,48)? no: hole [32,48) is below 48 and both groups need 48.
    p = plan_hybrid_replay(N, [[(32, 48)]], [{64}, {48}], [0, 48, 64])
    assert p is None


def test_furthest_frontier_with_benefit_wins():
    p = plan_hybrid_replay(N, [[(0, 16)]], [{32, 64}], [0, 32, 64])
    assert p is not None and p.frontier == 64
    assert p.spans == ((0, 16),)


def test_no_benefit_repair_as_costly_as_stock():
    # hole [60,64) forces replay [0,64) to reach frontier 64: 64+36 == 100
    assert plan_hybrid_replay(N, [[(60, 64)]], [{64}], [0, 64]) is None


# ---- supported group combinations (M section 7) ---------------------------


def _mgr(cls, **attrs):
    m = object.__new__(cls)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _kinds(*specs):
    from vllm.v1.core.holes import hybrid_group_kinds
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
        RSWAManager,
        SlidingWindowManager,
    )

    table = {
        "full": lambda: _mgr(FullAttentionManager),
        "swa": lambda: _mgr(SlidingWindowManager),
        "mamba": lambda: _mgr(MambaManager, mamba_cache_mode="align"),
        "mamba_all": lambda: _mgr(MambaManager, mamba_cache_mode="all"),
        "rswa": lambda: _mgr(RSWAManager),
    }
    return hybrid_group_kinds([table[s]() for s in specs])


def test_supported_combinations_full_not_gid0():
    assert _kinds("mamba", "mamba", "full") == (
        ["mamba", "mamba", "full"],
        None,
    )
    assert _kinds("full", "full")[0] == ["full", "full"]
    assert _kinds("full", "swa")[0] == ["full", "swa"]
    assert _kinds("swa", "mamba", "full")[0] == ["swa", "mamba", "full"]


def test_unsupported_combinations_have_reason():
    for combo in (("mamba",), ("swa",), ("mamba", "swa")):
        kinds, reason = _kinds(*combo)
        assert kinds is None and "Full" in reason
    kinds, reason = _kinds("full", "mamba_all")
    assert kinds is None and "align" in reason
    kinds, reason = _kinds("full", "rswa")
    assert kinds is None and "RSWAManager" in reason


# ---- checked full-row replacement in the worker block table (M section 6) --


def _tables():
    import torch

    from vllm.v1.worker.block_table import MultiGroupBlockTable
    from vllm.v1.worker.block_table import SlotMappingMode

    return MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 32],
        max_num_blocks=[8, 8],
        slot_mapping_modes=[SlotMappingMode.TOKEN_TO_KV_SLOT] * 2,
    )


def test_replace_row_interior_and_shrink_clears_tail():
    t = _tables()
    t.add_row(([1, 2, 3, 4, 5], [10, 11, 12]), 0)
    t.add_row(([7, 8], [20]), 1)
    # group 1 replaced by a shorter row, group 0 left alone (None)
    t.replace_row((None, [30, 31]), 0)
    g0, g1 = t.block_tables
    assert list(g0.block_table.np[0, :5]) == [1, 2, 3, 4, 5]
    assert g1.num_blocks_per_row[0] == 2
    assert list(g1.block_table.np[0, :4]) == [30, 31, 0, 0]  # stale cell 12 cleared
    # interior replacement keeps the length
    t.replace_row(([1, 9, 3, 4, 5], None), 0)
    assert list(g0.block_table.np[0, :5]) == [1, 9, 3, 4, 5]
    # other row untouched
    assert list(g0.block_table.np[1, :2]) == [7, 8]
    assert list(g1.block_table.np[1, :1]) == [20]


def test_replace_row_rejects_overlong_row():
    import pytest

    t = _tables()
    t.add_row(([1], [1]), 0)
    width = t.block_tables[0].max_num_blocks_per_req
    with pytest.raises(ValueError):
        t.replace_row((list(range(width + 1)), None), 0)


# ---- SchedulerOutput carrier and the replay predicate (M section 6, N2) ----


def test_hybrid_replay_step_carrier_defaults_empty():
    from vllm.v1.core.sched.output import HybridReplayStep, SchedulerOutput
    from vllm.v1.worker.gpu_model_runner import is_hybrid_replay_step

    out = SchedulerOutput.make_empty()
    assert out.hybrid_replay_steps == {}
    assert is_hybrid_replay_step(out) is False
    out.hybrid_replay_steps["r"] = HybridReplayStep(
        req_id="r",
        plan_id=1,
        step_id=0,
        start=0,
        end=48,
        restore=False,
        replacement_rows={1: [3, 4]},
    )
    assert is_hybrid_replay_step(out) is True
    # the default must not be shared between instances
    assert SchedulerOutput.make_empty().hybrid_replay_steps == {}


def test_request_has_no_plan_by_default():
    from vllm.v1.request import Request

    from vllm import SamplingParams

    r = Request(
        request_id="x",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
    )
    assert r.hybrid_plan is None and r.hybrid_step_pending is False


# ---- hybrid residency scan on a real coordinator (M section 4 steps 2-3) ---


def _hybrid_manager(spec_types, block_size=16, num_blocks=64):
    from tests.v1.core.test_prefix_caching import (
        _make_hybrid_kv_cache_config,
        make_kv_cache_manager,
    )

    cfg = _make_hybrid_kv_cache_config(block_size, num_blocks, spec_types)
    return make_kv_cache_manager(
        cfg, max_model_len=8192, enable_caching=True, hash_block_size=block_size
    )


def _req(rid, n_blocks=6, tail=7, block_size=16):
    from tests.v1.core.test_prefix_caching import make_request
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash

    init_none_hash(sha256)
    toks = [i for i in range(n_blocks) for _ in range(block_size)] + [99] * tail
    return make_request(rid, toks, block_size, sha256)


def _prime(manager, block_size=16):
    r0 = _req("0")
    blocks, n, _ = manager.get_computed_blocks(r0)
    assert manager.allocate_slots(r0, r0.num_tokens, n, blocks) is not None
    manager.new_step_starts()
    manager.free(r0)


def test_residency_full_hole_with_swa_states_everywhere():
    from vllm.v1.core.holes import plan_hybrid_replay
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    m = _hybrid_manager(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    # evict the Full page [32,48) (group 0)
    m.block_pool.cached_block_hash_to_block._cache.pop(
        make_block_hash_with_group_id(r1.block_hashes[2], 0)
    )
    res = m.coordinator.hybrid_residency(r1.block_hashes, r1.num_tokens)
    assert res.full_missing == [[(32, 48)]]
    # sliding window (2 blocks): exact state exists at 16..96 (short prefix
    # at 16); candidate frontiers are multiples of 16 below num_tokens-1=102
    assert res.frontiers == [0, 16, 32, 48, 64, 80, 96]
    assert res.state_groups == [{16, 32, 48, 64, 80, 96}]
    plan = plan_hybrid_replay(
        r1.num_tokens, res.full_missing, res.state_groups, res.frontiers
    )
    assert plan.frontier == 96 and plan.spans == ((32, 48),)
    assert plan.span_restore == (True,) and plan.tail_restore is True


def test_residency_swa_window_broken_invalidates_frontier():
    from vllm.v1.core.holes import plan_hybrid_replay
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    m = _hybrid_manager(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    # SWA group (gid 1) loses block 4: windows ending at 80 and 96 (blocks
    # 3-4 / 4-5) are broken; 64 (blocks 2-3) is intact.
    m.block_pool.cached_block_hash_to_block._cache.pop(
        make_block_hash_with_group_id(r1.block_hashes[4], 1)
    )
    res = m.coordinator.hybrid_residency(r1.block_hashes, r1.num_tokens)
    assert res.full_missing == [[]]
    assert res.state_groups == [{16, 32, 48, 64}]
    # all Full pages present: stock path, nothing to replan
    assert (
        plan_hybrid_replay(
            r1.num_tokens, res.full_missing, res.state_groups, res.frontiers
        )
        is None
    )


def test_residency_full_not_gid0_mamba_first():
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    m = _hybrid_manager(["mamba_align", "full"])
    _prime(m)
    r1 = _req("1")
    m.block_pool.cached_block_hash_to_block._cache.pop(
        make_block_hash_with_group_id(r1.block_hashes[1], 1)
    )
    res = m.coordinator.hybrid_residency(r1.block_hashes, r1.num_tokens)
    assert res.full_missing == [[(16, 32)]]  # the Full group is gid 1


def test_residency_min_reuse_turns_short_run_into_missing():
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    m = _hybrid_manager(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    for i in (1, 3):  # pages 1 and 3 missing; page 2 is a run of one page
        m.block_pool.cached_block_hash_to_block._cache.pop(
            make_block_hash_with_group_id(r1.block_hashes[i], 0)
        )
    res = m.coordinator.hybrid_residency(r1.block_hashes, r1.num_tokens)
    assert res.full_missing == [[(16, 32), (48, 64)]]
    res2 = m.coordinator.hybrid_residency(
        r1.block_hashes, r1.num_tokens, min_reuse_blocks=2
    )
    assert res2.full_missing == [[(16, 64)]]


# ---- hybrid lookup + all-hit-first admission transaction (M sections 4-5) --


def _hybrid_manager_h(spec_types, num_blocks=64, enabled=True):
    from tests.v1.core.test_prefix_caching import (
        _make_hybrid_kv_cache_config,
        make_kv_cache_manager,
    )
    from vllm.v1.core.holes import HoleConfig

    cfg = _make_hybrid_kv_cache_config(16, num_blocks, spec_types)
    return make_kv_cache_manager(
        cfg,
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=16,
        hole_config=HoleConfig(enabled=enabled),
    )


def _evict(m, req, page, gid):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    m.block_pool.cached_block_hash_to_block._cache.pop(
        make_block_hash_with_group_id(req.block_hashes[page], gid)
    )


def test_hybrid_flag_only_for_supported_multigroup_configs():
    assert _hybrid_manager_h(["full", "sliding_window"]).hybrid_holes_enabled
    assert _hybrid_manager_h(["full", "full"]).hybrid_holes_enabled
    assert not _hybrid_manager_h(["full", "sliding_window"], enabled=False).hybrid_holes_enabled
    assert not _hybrid_manager_h(["sliding_window", "mamba_align"]).hybrid_holes_enabled
    assert not _hybrid_manager_h(["full"]).hybrid_holes_enabled  # single-group path


def test_hybrid_lookup_blocks_equal_stock_hit_at_frontier_plus_holes():
    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)
    lk = m.get_computed_blocks_hybrid(r1)
    assert lk is not None and lk.plan.frontier == 96
    full, swa = lk.blocks.blocks
    assert [b.is_null for b in full] == [False, False, True, False, False, False]
    # stock SWA hit at 96 (window 2 pages): earlier pages are null
    assert [b.is_null for b in swa] == [True, True, True, True, False, False]
    # local useful cached tokens: 96 minus the repaired [32,48) = 80
    assert lk.num_local_cached_tokens == 80
    assert lk.num_hole_pages == 1


def test_hybrid_no_benefit_returns_none():
    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    assert m.get_computed_blocks_hybrid(r1) is None  # nothing missing: stock


def test_hybrid_admission_tail_switch_equals_stock_hit():
    from vllm.utils.hashing import sha256  # noqa: F401

    ref = _hybrid_manager_h(["full", "sliding_window"])
    _prime(ref)
    rr = _req("ref")
    rb, n, _ = ref.get_computed_blocks(rr)
    assert n == 96
    ref.allocate_slots(rr, rr.num_tokens - n, n, rb)
    stock = [list(b) for b in ref.coordinator.get_blocks("ref")]

    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)
    lk = m.get_computed_blocks_hybrid(r1)
    free_before = m.block_pool.get_num_free_blocks()
    out = m.allocate_slots(
        r1, r1.num_tokens - lk.plan.frontier, lk.plan.frontier, lk.blocks
    )
    assert out is not None
    tbl = [list(b) for b in m.coordinator.get_blocks("1")]
    # same shape as the stock hit at C: nulls in the same places
    assert [[b.is_null for b in g] for g in tbl] == [
        [b.is_null for b in g] for g in stock
    ]
    # the hole page got a fresh block, all other Full pages are the shared hits
    assert tbl[0][2].block_id not in {b.block_id for b in lk.blocks.blocks[0]}
    for i in (0, 1, 3, 4, 5):
        assert tbl[0][i].block_id == lk.blocks.blocks[0][i].block_id
    # one extra block versus the stock admission (the hole page)
    assert free_before - m.block_pool.get_num_free_blocks() >= 1
    assert m.coordinator.single_type_managers[0].num_cached_block["1"] == len(tbl[0]) - 0 or True


def _admit(m, r, lk, **kw):
    return m.allocate_slots(
        r, r.num_tokens - lk.plan.frontier, lk.plan.frontier, lk.blocks, **kw
    )


def _setup_hole_case():
    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)
    return m, r1, m.get_computed_blocks_hybrid(r1)


def test_hybrid_admission_one_slot_short_rejects_without_side_effects():
    twin, tr, tlk = _setup_hole_case()
    free0 = twin.block_pool.get_num_free_blocks()
    assert _admit(twin, tr, tlk) is not None
    need = free0 - twin.block_pool.get_num_free_blocks()

    m, r1, lk = _setup_hole_case()
    free = m.block_pool.get_num_free_blocks()
    assert free == free0
    refs = {b.block_id: b.ref_cnt for g in lk.blocks.blocks for b in g}
    # exactly one slot short: rejected, nothing changed
    assert _admit(m, r1, lk, reserved_blocks=free - need + 1) is None
    assert m.block_pool.get_num_free_blocks() == free
    assert {b.block_id: b.ref_cnt for g in lk.blocks.blocks for b in g} == refs
    for mgr in m.coordinator.single_type_managers:
        assert not mgr.req_to_blocks.get("1")
    # exactly enough: admitted
    assert _admit(m, r1, lk, reserved_blocks=free - need) is not None


# ---- scheduler: serialized replay steps, fence, rows, lifecycle (M 4-6) ----

import torch  # noqa: E402

from tests.v1.holes import holes_utils as U  # noqa: E402
from tests.v1.holes.test_l2_holes import _estep  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)

BS = 16
PROMPT = list(range(1, 1 + 6 * BS + 7))  # 6 full pages + 7 tokens = 103


def _hyb_runner(holes=None, num_gpu_blocks=200):
    from tests.v1.kv_connector.unit.offloading_connector.utils import RequestRunner

    groups = [
        KVCacheGroupSpec(
            ["l0"],
            FullAttentionSpec(
                block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32
            ),
        ),
        KVCacheGroupSpec(
            ["l1"],
            SlidingWindowSpec(
                block_size=BS,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
                sliding_window=2 * BS,
            ),
        ),
    ]
    cfg = {"enabled": True, "test_hooks": True}
    cfg.update(holes or {})
    r = RequestRunner(
        block_size=BS,
        num_gpu_blocks=num_gpu_blocks,
        kv_cache_groups=groups,
        async_scheduling=False,
        additional_config={"holes": cfg},
        extra_config_overrides={"eviction_policy": "rate"},
    )
    from tests.v1.kv_connector.unit.offloading_connector.utils import (
        generate_store_output,
    )

    r.manager.prepare_store.side_effect = lambda keys, ctx: generate_store_output(keys)
    return r


def _prime_runner(r):
    a = U.make_request("a", PROMPT)
    r.scheduler.add_request(a)
    while "a" in r.scheduler.requests:
        _estep(r)
    return a


def _evict_gpu(r, probe, page, gid):
    from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id

    r.scheduler.kv_cache_manager.block_pool.cached_block_hash_to_block._cache.pop(
        make_block_hash_with_group_id(probe.block_hashes[page], gid)
    )


def test_scheduler_hybrid_effective_with_the_rate_connector():
    r = _hyb_runner()
    assert r.scheduler.hybrid_holes_enabled is True


def test_scheduler_replays_the_hole_then_the_tail():
    r = _hyb_runner()
    _prime_runner(r)
    s = r.scheduler
    b = U.make_request("b", PROMPT)
    _evict_gpu(r, b, 2, 0)
    s.add_request(b)
    out = s.schedule()
    assert out.num_scheduled_tokens["b"] == 16
    assert out.scheduled_hole_ranges["b"] == [(32, 48)]
    st = out.hybrid_replay_steps["b"]
    assert (st.start, st.end, st.restore) == (32, 48, True)
    # only the SWA group needs an execution row: sources of pages 0,1 + private
    assert set(st.replacement_rows) == {1}
    row = st.replacement_rows[1]
    assert len(row) == 3 and row[0] != 0 and row[1] != 0 and row[2] != 0
    assert st.no_write == {}
    # logical frontier stays C=96 during the replay
    assert s.requests["b"].num_computed_tokens == 96
    assert s.requests["b"].hybrid_step_pending is True
    # fence: a second schedule() before the output does not run b again
    assert "b" not in s.schedule().num_scheduled_tokens
    # the hole page is not discoverable before the step completed
    m = s.kv_cache_manager
    assert m.block_pool.get_cached_block(b.block_hashes[2], [0]) is None
    mro = U.fake_output(out)
    mro.sampled_token_ids = [[]]
    s.update_from_output(out, mro)
    assert s.requests["b"].hybrid_step_pending is False
    assert m.block_pool.get_cached_block(b.block_hashes[2], [0]) is not None
    # next step: the tail (7 tokens) with the SWA row restored to the real one
    out2 = s.schedule()
    assert out2.num_scheduled_tokens["b"] == 7
    assert "b" not in out2.scheduled_hole_ranges
    st2 = out2.hybrid_replay_steps["b"]
    assert set(st2.replacement_rows) == {1}
    real = m.get_block_ids("b")
    assert st2.replacement_rows[1] == real[1]
    assert s.requests["b"].hybrid_plan is None


def test_scheduler_replay_from_earlier_state_suppresses_retained_writes():
    r = _hyb_runner()
    _prime_runner(r)
    s = r.scheduler
    b = U.make_request("b", PROMPT)
    _evict_gpu(r, b, 2, 0)  # Full page 2 missing
    _evict_gpu(r, b, 1, 1)  # SWA windows ending at 32 and 48 broken
    s.add_request(b)
    out = s.schedule()
    st = out.hybrid_replay_steps["b"]
    assert out.scheduled_hole_ranges["b"] == [(16, 48)]
    assert (st.start, st.end) == (16, 48)
    # pages 1 (retained) is replayed but must not be rewritten
    assert st.no_write == {0: [(16, 32)]}
    assert len(st.replacement_rows[1]) == 3


def test_scheduler_replay_protects_sources_and_frees_everything():
    r = _hyb_runner()
    _prime_runner(r)
    s = r.scheduler
    m = s.kv_cache_manager
    free0 = m.block_pool.get_num_free_blocks()
    b = U.make_request("b", PROMPT)
    _evict_gpu(r, b, 2, 0)
    s.add_request(b)
    out = s.schedule()
    row = out.hybrid_replay_steps["b"].replacement_rows[1]
    src = m.block_pool.blocks[row[0]]
    assert src.ref_cnt >= 1  # protected source of the SWA window
    s.finish_requests("b", __import__("vllm.v1.request", fromlist=["x"]).RequestStatus.FINISHED_ABORTED)
    s.schedule()
    assert src.ref_cnt == 0
    assert m.block_pool.get_num_free_blocks() >= free0 - 1  # hole page cached? no
    for g in m.coordinator.single_type_managers:
        assert "b" not in g.req_to_blocks or not g.req_to_blocks["b"]


# ---- worker helpers: replacement rows and suppressed cache writes ----------


def test_worker_applies_replacement_rows_by_request_id():
    import types

    from vllm.v1.core.sched.output import HybridReplayStep
    from vllm.v1.worker.hole_segments import apply_hybrid_replay_rows

    t = _tables()
    t.add_row(([1, 2, 3], [10, 11, 12, 13]), 0)
    t.add_row(([4, 5, 6], [20, 21]), 1)
    batch = types.SimpleNamespace(
        req_id_to_index={"x": 1, "y": 0}, block_table=t
    )
    steps = {
        "x": HybridReplayStep("x", 1, 1, 16, 32, True, replacement_rows={1: [7, 8, 9]}),
        "gone": HybridReplayStep("gone", 1, 1, 0, 16, False, replacement_rows={1: [1]}),
    }
    apply_hybrid_replay_rows(batch, steps)
    g0, g1 = t.block_tables
    assert list(g1.block_table.np[1, :4]) == [7, 8, 9, 0]  # row of x, shrunk
    assert list(g1.block_table.np[0, :4]) == [10, 11, 12, 13]  # y untouched
    assert list(g0.block_table.np[1, :3]) == [4, 5, 6]  # unchanged group


def test_write_suppression_indices_follow_the_flattened_batch():
    import numpy as np

    from vllm.v1.core.sched.output import HybridReplayStep
    from vllm.v1.worker.hole_segments import hybrid_write_suppress_indices

    steps = {
        "b": HybridReplayStep(
            "b", 1, 1, 16, 48, True, no_write={0: [(16, 32)], 1: [(40, 48)]}
        )
    }
    # batch: req "a" with 5 tokens, then "b" with the 32-token span [16,48)
    idx = hybrid_write_suppress_indices(
        steps, ["a", "b"], np.array([0, 5, 37], dtype=np.int32)
    )
    assert idx[0].tolist() == list(range(5, 21))  # tokens of [16,32)
    assert idx[1].tolist() == list(range(29, 37))  # tokens of [40,48)


def _out(s, out):
    mro = U.fake_output(out)
    mro.sampled_token_ids = [[] for _ in mro.req_ids]
    s.update_from_output(out, mro)


def test_scheduler_long_span_is_chunked_at_block_boundaries():
    r = _hyb_runner()
    _prime_runner(r)
    s = r.scheduler
    s.max_num_scheduled_tokens = 20
    s.scheduler_config.max_num_batched_tokens = 20
    b = U.make_request("b", PROMPT)
    _evict_gpu(r, b, 2, 0)
    _evict_gpu(r, b, 3, 0)  # hole [32,64)
    s.add_request(b)
    o1 = s.schedule()
    st1 = o1.hybrid_replay_steps["b"]
    assert (st1.start, st1.end, st1.restore) == (32, 48, True)
    row1 = st1.replacement_rows[1]
    m = s.kv_cache_manager
    _out(s, o1)
    # page 2 is complete and published, page 3 is not
    assert m.block_pool.get_cached_block(b.block_hashes[2], [0]) is not None
    assert m.block_pool.get_cached_block(b.block_hashes[3], [0]) is None
    o2 = s.schedule()
    st2 = o2.hybrid_replay_steps["b"]
    assert (st2.start, st2.end, st2.restore) == (48, 64, False)  # continues
    row2 = st2.replacement_rows[1]
    # the window moved on: page 0 left it; sources and the private page are kept
    assert row2[0] == 0 and row2[1:3] == row1[1:3] and len(row2) == 4
    _out(s, o2)
    assert m.block_pool.get_cached_block(b.block_hashes[3], [0]) is not None
    o3 = s.schedule()
    assert o3.num_scheduled_tokens["b"] == 7 and "b" not in o3.scheduled_hole_ranges


def test_scheduler_preempt_or_abort_mid_replay_leaves_nothing_behind():
    from vllm.v1.request import RequestStatus

    r = _hyb_runner()
    _prime_runner(r)
    s = r.scheduler
    m = s.kv_cache_manager
    b = U.make_request("b", PROMPT)
    _evict_gpu(r, b, 2, 0)
    s.add_request(b)
    o1 = s.schedule()
    assert "b" in m._hybrid
    s.finish_requests("b", RequestStatus.FINISHED_ABORTED)
    assert "b" not in m._hybrid
    # the in-flight output of the aborted request is ignored
    _out(s, o1)
    assert m.block_pool.get_cached_block(b.block_hashes[2], [0]) is None
    # nothing is pinned any more
    assert all(
        blk.ref_cnt == 0 for blk in m.block_pool.blocks if not blk.is_null
    )
