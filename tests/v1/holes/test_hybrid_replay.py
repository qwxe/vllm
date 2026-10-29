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
