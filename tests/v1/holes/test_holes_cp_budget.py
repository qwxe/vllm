# SPDX-License-Identifier: Apache-2.0
"""holes/cp: per-scheme CP step budgets in the hole chunk planner (CPU)."""

try:
    from . import holes_utils as U
except ImportError:
    import holes_utils as U

from vllm.v1.core.holes import HoleConfig, cp_ring_min_tokens, cp_step_budget


def test_budget_values():
    c = HoleConfig.from_dict
    assert cp_step_budget(c({}), 4) == 0  # no CP scheme: no budget
    for s in ("ulysses", "passq", "allgather"):
        assert cp_step_budget(c({"cp_scheme": s}), 4) == 4 * 2048
    # C_ring = 280e12 / (4 * 16e9) = 4375 -> 4480 (granule 128)
    assert cp_ring_min_tokens(c({"cp_scheme": "ring"})) == 4480
    assert cp_step_budget(c({"cp_scheme": "ring"}), 2) == 2 * 4480
    # a fast link: C_ring = 1400 -> 1408 < 2048
    fast = c({"cp_scheme": "ring", "cp_link_gbps": 50.0})
    assert cp_step_budget(fast, 8) == 8 * 2048
    a = c({"cp_scheme": "adaptive"})
    assert cp_step_budget(a, 2, work=1000, context=100000) == 2 * 2048
    assert cp_step_budget(a, 2, work=50000, context=100000) == 2 * 4480
    assert cp_step_budget(c({"cp_scheme": "passq", "cp_rank_tokens": 1024}),
                          4) == 4096
    try:
        c({"cp_scheme": "zigzag"})
        raise RuntimeError("expected a validation error")
    except AssertionError:
        pass


def _hole_steps(holes_cfg, max_tokens):
    prompt = list(range(1, 1 + 400 * U.BS + 1))  # 6401 tokens
    s = U.make_scheduler(holes_cfg, max_num_batched_tokens=max_tokens)
    a = U.make_request("a", prompt, max_tokens=1)
    s.add_request(a)
    while not a.is_finished():
        U.step(s)
    probe = U.make_request("probe", prompt)
    s.kv_cache_manager.evict_cached_token_range(probe, 16 * U.BS, 316 * U.BS)
    b = U.make_request("b", prompt)
    s.add_request(b)
    outs = []
    while b.num_output_tokens == 0:
        outs.append(U.step(s))
    assert b.num_computed_tokens == 6401
    return [o.num_scheduled_tokens["b"] for o in outs], \
        [o.scheduled_hole_ranges.get("b") for o in outs]


def test_scheduler_cp_budgets():
    # hole [256, 5056) = 4800 tokens, then a 1-token tail (runs alone)
    base = {"enabled": True}
    assert _hole_steps(base, 2048)[0] == [2048, 2048, 704, 1]
    # pass-Q / Ulysses at d = 2: 4096 per step
    for s in ("passq", "ulysses"):
        n, h = _hole_steps({**base, "cp_scheme": s, "cp_size": 2}, 8192)
        assert n == [4096, 704, 1], (s, n)
        assert h[0] == [(256, 4352)] and h[1] == [(4352, 5056)]
    # ring at d = 2: 8960 per step, the whole hole in one step
    n, _ = _hole_steps({**base, "cp_scheme": "ring", "cp_size": 2}, 16384)
    assert n == [4800, 1], n
    # max_num_batched_tokens still caps the CP budget
    n, _ = _hole_steps({**base, "cp_scheme": "ring", "cp_size": 2}, 2048)
    assert n == [2048, 2048, 704, 1], n
    # adaptive: 4800 of 6401 to compute (0.75 >= 0.2) -> ring budget
    n, _ = _hole_steps({**base, "cp_scheme": "adaptive", "cp_size": 2}, 16384)
    assert n == [4800, 1], n


if __name__ == "__main__":
    test_budget_values()
    test_scheduler_cp_budgets()
    print("cp budget tests OK")
