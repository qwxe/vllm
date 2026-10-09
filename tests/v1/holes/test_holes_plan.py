# SPDX-License-Identifier: Apache-2.0
"""holes/core: chunk planner (KERNEL-ALGO 5.2) and range helpers. CPU only."""

from vllm.v1.core.holes import (
    HoleConfig,
    apply_min_reuse,
    hits_to_holes,
    plan_step,
    subtract_ranges,
)

CFG = HoleConfig(enabled=True)


def test_hits_to_holes():
    assert hits_to_holes([True, False, False, True, False, True], 16) == [
        (16, 48),
        (64, 80),
    ]
    assert hits_to_holes([], 16) == []
    assert hits_to_holes([False, True], 16) == [(0, 16)]


def test_apply_min_reuse():
    assert apply_min_reuse([True, False, True, False, False], 1) == [True, False, True]
    # A 1-block run after a hole is dropped with min 2; the leading run stays.
    assert apply_min_reuse([True, False, True, False, True, True], 2) == [
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    assert apply_min_reuse([True, False, True], 2) == [True]
    assert apply_min_reuse([False, False], 1) == []


def test_subtract_ranges():
    assert subtract_ranges([(0, 100), (200, 300)], [(0, 50)]) == [(50, 100), (200, 300)]
    assert subtract_ranges([(0, 100)], [(0, 100)]) == []
    assert subtract_ranges([(0, 100)], [(20, 40)]) == [(0, 20), (40, 100)]


def test_plan_whole_fits():
    # hole + tail fit the budget: all in one step, holes first.
    pieces, n_tail = plan_step([(32, 96)], 160, 300, 2048, 2048, CFG)
    assert pieces == [(32, 96)] and n_tail == 140


def test_plan_front_first_and_granule_cut():
    # big hole: cut at 128-multiples from the hole start; no tail this step.
    pieces, n_tail = plan_step([(1024, 9024)], 20000, 21000, 2048, 2048, CFG)
    assert pieces == [(1024, 1024 + 2048)] and n_tail == 0


def test_plan_tail_cut_at_granule_from_segment_start():
    # hole 1000 tokens, budget 2048: tail gets 1048 -> cut to 1024 (128*8).
    pieces, n_tail = plan_step([(0, 1008)], 4096, 9000, 2048, 2048, CFG)
    assert pieces == [(0, 1008)]
    assert n_tail == 1024 - 0 or n_tail == 1024
    assert n_tail % 128 == 0


def test_plan_no_small_remainder_overshoot():
    # hole of 2048+100: remainder 100 < 256 and overshoot 100 < 256 -> take it
    # all when the hard limit allows.
    pieces, n_tail = plan_step([(0, 2148)], 4096, 5000, 2048, 4096, CFG)
    assert pieces == [(0, 2148)] and n_tail == 0


def test_plan_no_small_remainder_cut_back():
    # same, but the hard limit forbids the overshoot: cut back so that the
    # remainder is >= 256 (cut at a 128 multiple).
    pieces, n_tail = plan_step([(0, 2148)], 4096, 5000, 2048, 2048, CFG)
    (s, e), = pieces
    assert s == 0 and e % 128 == 0 and 2148 - e >= 256 and e <= 2048
    assert e == 1792


def test_plan_short_hole_not_left_behind():
    # hole A fills most of the budget, short hole B (<256) would end the step
    # with nothing of the next segment: B is deferred.
    pieces, n_tail = plan_step([(0, 1920), (4000, 4096)], 8000, 9000, 2048, 2048, CFG)
    assert pieces == [(0, 1920)] and n_tail == 0
    # If B is the only work, it runs (with the tail when it fits).
    pieces, n_tail = plan_step([(4000, 4096)], 8000, 8100, 2048, 2048, CFG)
    assert pieces == [(4000, 4096)] and n_tail == 100


def test_plan_min_mixed_tail():
    # A 1-token tail does not share a step with hole pieces.
    pieces, n_tail = plan_step([(0, 32)], 64, 65, 2048, 2048, CFG)
    assert pieces == [(0, 32)] and n_tail == 0
    pieces, n_tail = plan_step([], 64, 65, 2048, 2048, CFG)
    assert pieces == [] and n_tail == 1


def test_plan_budget_below_granule():
    pieces, n_tail = plan_step([(0, 4096)], 8000, 9000, 100, 100, CFG)
    assert pieces == [] and n_tail == 0


def test_plan_configurable():
    cfg = HoleConfig(enabled=True, chunk_granule=16, min_piece=0)
    pieces, n_tail = plan_step([(0, 4096)], 8000, 9000, 100, 100, cfg)
    assert pieces == [(0, 96)] and n_tail == 0


def test_plan_max_segments():
    holes = [(0, 32), (64, 96), (128, 160)]
    big = [(0, 512), (1024, 1536), (2048, 2560)]
    pieces, n_tail = plan_step(big, 3200, 3300, 2048, 2048, CFG, max_segments=2)
    assert pieces == [(0, 512), (1024, 1536)] and n_tail == 0
    # Short holes: the second one would end the step -> deferred (q_min rule).
    pieces, n_tail = plan_step(holes, 320, 400, 2048, 2048, CFG, max_segments=2)
    assert pieces == [(0, 32)] and n_tail == 0
    pieces, n_tail = plan_step(holes, 320, 400, 2048, 2048, CFG, max_segments=4)
    assert pieces == holes and n_tail == 80


if __name__ == "__main__":
    import sys

    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
