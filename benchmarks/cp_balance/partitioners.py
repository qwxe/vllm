# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row partitioners for one holed prefill step, as row-index arrays.

A step is one request's computed ranges `segs = [(start, length), ...]`. The
computed list is the concatenation of the ranges in position order; row j of
the step's Q tensor holds computed-list token j. Every partitioner returns
`rows[rank]` = ascending int64 row indices.

- contig, zigzag, zigzag_abs, balanced: the simulator's partitioners
  (`serving/core/cp_partition.py` on feat/cp-impl, copied verbatim into
  cp_partition_sim.py). `balanced` is PAPER-ALGORITHM section 4: equal
  sum of c(i) = a + b i per rank, prefix-sum binary search.
- balanced_cal: the same algorithm with (a, b) fitted on this GPU.
- balanced_attn: the same algorithm with a = 0 (pure causal attention work).
- c2e: holes/cp's ring / all-gather rows (COBAL two-ended, cuts on u = 128).
- c2e16: the same with cuts on 16-token units.
- snake: KERNEL.md's proposal, 4d equal-token chunks dealt by cost in snake
  order.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cp_partition_sim as cps  # noqa: E402

SIM_A, SIM_B = cps.DEFAULT_A, cps.DEFAULT_B


def computed_positions(segs):
    segs = sorted((int(s), int(n)) for s, n in segs if n > 0)
    if not segs:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(s, s + n, dtype=np.int64) for s, n in segs])


def _pieces_to_rows(pieces, qpos):
    out = []
    for ps in pieces:
        idx = [np.searchsorted(qpos, s) + np.arange(n) for s, n in ps if n > 0]
        out.append(np.sort(np.concatenate(idx)).astype(np.int64) if idx
                   else np.zeros(0, dtype=np.int64))
    return out


def _sim(method, segs, d, a=SIM_A, b=SIM_B):
    qpos = computed_positions(segs)
    pieces, _ = cps.partition(segs, d, method, a=a, b=b)
    return _pieces_to_rows(pieces, qpos)


def c2e(segs, d, u=128):
    import torch
    from vllm.v1.attention.cp_holes.plan import StepPlan, rows_c2e
    plan = StepPlan(segments=[list(segs)], d=d, u=u)
    return [r.numpy().astype(np.int64) for r in rows_c2e(plan)]


def snake(segs, d, per=4, a=SIM_A, b=SIM_B):
    qpos = computed_positions(segs)
    m = len(qpos)
    k = per * d
    cuts = [m * j // k for j in range(k + 1)]
    cost = [a * (cuts[j + 1] - cuts[j]) + b * float(qpos[cuts[j]:cuts[j + 1]].sum())
            for j in range(k)]
    order = np.argsort(cost)[::-1]
    out = [[] for _ in range(d)]
    for i, j in enumerate(order):
        rnd, pos = divmod(i, d)
        r = pos if rnd % 2 == 0 else d - 1 - pos
        out[r].append(np.arange(cuts[j], cuts[j + 1], dtype=np.int64))
    return [np.sort(np.concatenate(x)) if x else np.zeros(0, dtype=np.int64) for x in out]


def partition(name, segs, d, cal=None):
    """cal: dict with a, b (seconds per token, seconds per token-position) for balanced_cal."""
    if d == 1:
        return [np.arange(len(computed_positions(segs)), dtype=np.int64)]
    if name in ("contig", "zigzag", "zigzag_abs", "balanced"):
        return _sim(name, segs, d)
    if name == "balanced_cal":
        return _sim("balanced", segs, d, a=cal["a"], b=cal["b"])
    if name == "balanced_attn":
        return _sim("balanced", segs, d, a=1e-30, b=1.0)
    if name == "c2e":
        return c2e(segs, d)
    if name == "c2e16":
        return c2e(segs, d, u=16)
    if name == "snake":
        return snake(segs, d)
    raise ValueError(name)


ROW_PARTITIONERS = ("contig", "zigzag", "zigzag_abs", "balanced", "balanced_cal", "balanced_attn", "c2e", "snake")
