# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/cp: missing-KV ranges (hit-mask misses) from the scheduler to the
CP_HOLES slot validity (CPU)."""

import numpy as np
import torch

try:
    from . import holes_utils as U
except ImportError:
    import holes_utils as U

from vllm.v1.attention.backends.cp_holes import (kv_slot_validity,
                                                 request_segments)
from vllm.v1.worker.hole_segments import build_hole_segments


def test_scheduler_reports_missing_kv():
    prompt = list(range(1, 1 + 400 * U.BS + 1))  # 6401 tokens
    s = U.make_scheduler({"enabled": True})
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
    miss = [o.missing_kv_ranges.get("b") for o in outs]
    # hole [256, 5056) chunked 2048 / 2048 / 704: what is still missing
    # after each step
    assert miss[0] == [(2304, 5056)], miss
    assert miss[1] == [(4352, 5056)], miss
    assert miss[2] is None and miss[3] is None, miss
    # every missing range lies past the step's computed positions
    for o, m in zip(outs, miss):
        for a0, _ in m or ():
            ends = [e for _, e in o.scheduled_hole_ranges.get("b", [])]
            assert all(e <= a0 for e in ends)


def test_segments_carry_missing():
    pos = np.zeros(10, dtype=np.int64)
    hs = build_hole_segments(["x", "y"], {"x": [(16, 20)]},
                             np.array([6, 4]), np.array([64, 8]), pos,
                             missing_ranges={"x": [(20, 48)]})
    assert hs.req_missing_ranges == [[(20, 48)], []]
    hs2 = build_hole_segments(["x"], {"x": [(16, 20)]}, np.array([4]),
                              np.array([64]), np.zeros(4, dtype=np.int64))
    assert hs2.req_missing_ranges is None


def test_slot_validity():
    lp = torch.arange(0, 64)
    lp[5] = -1  # an empty slot
    pr = kv_slot_validity(lp, 40, [(10, 6), (36, 4)], [(48, 60)], 40)
    want = (lp >= 0) & (lp < 40)
    want[10:16] = False
    want[36:40] = False
    assert torch.equal(pr, want)
    try:
        kv_slot_validity(lp, 40, [(10, 6)], [(20, 30)], 40)
        raise RuntimeError("expected ValueError")
    except ValueError:
        pass


def test_request_segments_missing():
    from types import SimpleNamespace
    pos = np.zeros(10, dtype=np.int64)
    hs = build_hole_segments(["x", "y"], {"x": [(16, 20)]},
                             np.array([6, 4]), np.array([64, 8]), pos,
                             missing_ranges={"x": [(80, 96)]})
    cam = SimpleNamespace(hole_segments=hs,
                          block_table_tensor=torch.arange(30).view(3, 10))
    segs, rows, seq, missing = request_segments(cam)
    assert segs == [[(16, 4), (64, 2)], [(8, 4)]]
    assert seq == [66, 12] and missing == [[(80, 96)], []]


if __name__ == "__main__":
    test_scheduler_reports_missing_kv()
    test_segments_carry_missing()
    test_slot_validity()
    test_request_segments_missing()
    print("missing kv tests OK")
