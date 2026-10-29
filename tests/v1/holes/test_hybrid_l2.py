# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/M stage 4 (M note 7.2): L2 sources of the hybrid lookup. CPU only."""

from tests.v1.holes.test_hybrid_replay import (  # noqa: F401
    _evict,
    _hybrid_manager_h,
    _prime,
    _req,
)


# ---- stage 4: L2 Full pages as hybrid sources (M 7.2 rules 1-2, lookup level) ----


def _ext(n, pages):
    return [i in pages for i in range(n)]


def test_l2_full_page_below_the_frontier_is_a_load_not_a_replay():
    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)  # Full page 2 lost on the GPU, held in L2
    _evict(m, r1, 4, 0)  # Full page 4 lost everywhere
    lk = m.get_computed_blocks_hybrid(r1, ext_hit={0: _ext(6, {2})})
    assert lk is not None
    # only page 4 (tokens 64-80) needs a whole-model replay; page 2 is loaded
    assert lk.plan.spans[-1] == (64, 80)
    assert all(not (s <= 32 < e) for s, e in lk.plan.spans)
    assert lk.load_ranges == {0: [(32, 48)]}
    assert lk.blocks.blocks[0][2].is_null  # a fresh block in place at admission
    assert lk.num_hole_pages == 1  # only page 4 is a hole; page 2 is not
    # without the L2 page both pages are holes
    lk0 = m.get_computed_blocks_hybrid(r1)
    assert lk0.load_ranges == {}
    assert lk0.num_hole_pages == 2


def test_l2_page_covering_every_hole_leaves_the_stock_path():
    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)
    # the stock lookup (with the connector's prefix hit) is at least as good
    assert m.get_computed_blocks_hybrid(r1, ext_hit={0: _ext(6, {2})}) is None


def test_l2_hole_accounting_excludes_loaded_pages_from_replay_writes():
    m = _hybrid_manager_h(["full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)
    _evict(m, r1, 4, 0)
    ext = {0: _ext(6, {2})}
    lk = m.get_computed_blocks_hybrid(r1, ext_hit=ext)
    m.protect_hybrid_sources(r1, lk)
    m.acquire_hybrid(r1, lk)
    # the loaded page is retained for the replay's purposes (never rewritten)
    assert m._hybrid["1"].holes == {0: [(64, 80)]}


def test_l2_joint_full_groups_need_every_group_covered():
    m = _hybrid_manager_h(["full", "full", "sliding_window"])
    _prime(m)
    r1 = _req("1")
    _evict(m, r1, 2, 0)
    _evict(m, r1, 2, 1)
    _evict(m, r1, 4, 0)
    _evict(m, r1, 4, 1)
    # page 2 in L2 for both groups, page 4 only for group 0: still a hole
    lk = m.get_computed_blocks_hybrid(
        r1, ext_hit={0: _ext(6, {2, 4}), 1: _ext(6, {2})}
    )
    assert lk.plan.spans[-1] == (64, 80)
    assert lk.load_ranges == {0: [(32, 48)], 1: [(32, 48)]}


# ---- stage 4: the store offer of a hybrid request never covers an unfinished replay ----


def test_completed_replay_prefix_of_a_plan():
    from vllm.v1.core.holes import HybridReplayPlan, hybrid_completed_prefix

    plan = HybridReplayPlan(
        frontier=96, spans=((32, 48), (64, 96)), span_restore=(True, True),
        tail_restore=True,
    )
    assert hybrid_completed_prefix(plan, []) == 32  # nothing replayed yet
    assert hybrid_completed_prefix(plan, [(32, 48)]) == 64  # retained gap passes
    assert hybrid_completed_prefix(plan, [(32, 48), (64, 80)]) == 80  # partial span
    assert hybrid_completed_prefix(plan, [(32, 48), (64, 96)]) == 96
    assert hybrid_completed_prefix(plan, [(64, 96)]) == 32  # span 0 still open

