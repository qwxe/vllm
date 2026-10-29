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
