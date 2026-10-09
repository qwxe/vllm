# SPDX-License-Identifier: Apache-2.0
"""Step plan, KV placement and row partitioners for CP holed prefills.

A `StepPlan` is the global description of one prefill step. It is built from
scheduler metadata that every CP rank receives identically (in vLLM: the
`SchedulerOutput` plus the request's computed ranges from the hole-aware
scheduler), so every rank derives the same row assignment, buffer sizes and
collective split sizes without exchanging metadata.

Conventions (KERNEL-ALGO.md section 1 and 4.0):
- request `i` of the step computes segments S_i = [s, e) (hole pieces and
  tail chunk). E_step(i) = max e over its segments.
- Every position of [0, E_step(i)) that is not computed in this step is
  resident in the KV cache. Positions >= E_step(i) are never read in this step
  (causal truncation).
- The computed list is the concatenation of all requests' computed positions,
  request by request, each in position order. Global row index = index into
  this list.
- Position-sharded placement is interleaved with unit u (default 128 tokens):
  position p is homed on rank (p // u) % d.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

DEFAULT_INTERLEAVE = 128


def home_rank(pos: torch.Tensor, d: int,
              u: int = DEFAULT_INTERLEAVE) -> torch.Tensor:
    """Rank that holds the KV of absolute position `pos` (interleaved u)."""
    return torch.div(pos, u, rounding_mode="floor") % d


@dataclass
class StepPlan:
    """Global plan of one CP prefill step (identical on every rank)."""

    segments: list[list[tuple[int, int]]]  # per request: [(start, length)]
    d: int
    u: int = DEFAULT_INTERLEAVE
    # Filled by __post_init__.
    qpos: torch.Tensor = field(init=False)
    qreq: torch.Tensor = field(init=False)
    e_step: list[int] = field(init=False)
    req_rows: list[tuple[int, int]] = field(init=False)  # [lo, hi) per req

    def __post_init__(self):
        pos, req, self.e_step, self.req_rows = [], [], [], []
        lo = 0
        for i, segs in enumerate(self.segments):
            segs = sorted((int(s), int(n)) for s, n in segs if n > 0)
            self.segments[i] = segs
            for (s0, n0), (s1, _) in zip(segs, segs[1:]):
                assert s0 + n0 <= s1, f"overlapping segments in req {i}"
            n_i = 0
            for s, n in segs:
                pos.append(torch.arange(s, s + n, dtype=torch.int64))
                n_i += n
            req.append(torch.full((n_i, ), i, dtype=torch.int64))
            self.e_step.append(max((s + n for s, n in segs), default=0))
            self.req_rows.append((lo, lo + n_i))
            lo += n_i
        self.qpos = torch.cat(pos) if pos else torch.empty(0, dtype=torch.int64)
        self.qreq = torch.cat(req) if req else torch.empty(0, dtype=torch.int64)

    @property
    def m(self) -> int:
        return int(self.qpos.numel())

    @property
    def num_reqs(self) -> int:
        return len(self.segments)

    def computed_mask(self, i: int) -> torch.Tensor:
        """Bool [E_step(i)]: True where position is computed this step."""
        msk = torch.zeros(self.e_step[i], dtype=torch.bool)
        for s, n in self.segments[i]:
            msk[s:s + n] = True
        return msk

    def resident_positions(self, i: int) -> torch.Tensor:
        """Resident positions of request i inside [0, E_step(i))."""
        return (~self.computed_mask(i)).nonzero(as_tuple=True)[0]

    def resident_homed(self, i: int, rank: int) -> torch.Tensor:
        p = self.resident_positions(i)
        return p[home_rank(p, self.d, self.u) == rank]

    def num_resident_homed(self, rank: int,
                           reqs: Optional[Sequence[int]] = None) -> int:
        reqs = range(self.num_reqs) if reqs is None else reqs
        return sum(int(self.resident_homed(i, rank).numel()) for i in reqs)

    def ratio(self, i: int) -> float:
        """Computed tokens / E_step for request i (the ring vs pass-Q rule)."""
        lo, hi = self.req_rows[i]
        return (hi - lo) / max(1, self.e_step[i])


# --------------------------------------------------------------- partitioners


def rows_contiguous(rows, d: int, n_mb: int = 1) -> list[list[torch.Tensor]]:
    """Dense-layer rows for Ulysses / pass-Q.

    `rows` is either m (all rows of the step) or an ascending tensor of global
    row indices (a subset of requests). The row list is cut into `n_mb` contiguous micro-batches, and each
    micro-batch into d contiguous pieces; rank r owns piece r of every
    micro-batch. Within a request, every row of micro-batch k precedes every
    row of micro-batch k+1 in position, so a micro-batch's attention only needs
    keys of earlier micro-batches (causality across the overlap pipeline).

    Returns out[rank][mb] = global row indices (int64, ascending).
    """
    if isinstance(rows, int):
        rows = torch.arange(rows, dtype=torch.int64)
    m = int(rows.numel())
    out = [[None] * n_mb for _ in range(d)]
    mb_bounds = [m * k // n_mb for k in range(n_mb + 1)]
    for k in range(n_mb):
        a, b = mb_bounds[k], mb_bounds[k + 1]
        for r in range(d):
            lo = a + (b - a) * r // d
            hi = a + (b - a) * (r + 1) // d
            out[r][k] = rows[lo:hi]
    return out


def _attn_cost_units(pos: torch.Tensor, u: int) -> torch.Tensor:
    """Causal attention work of each u-token unit of a computed list."""
    w = (pos + 1).double()
    n = pos.numel()
    nu = (n + u - 1) // u
    pad = nu * u - n
    if pad:
        w = torch.cat([w, torch.zeros(pad, dtype=w.dtype)])
    return w.view(nu, u).sum(dim=1)


def _two_ended(cost: torch.Tensor, d: int, iters: int = 14
               ) -> list[list[tuple[int, int]]]:
    """COBAL c2e on unit costs: equal units and equal attention per rank.

    Ranks 0..d-2 are filled from the outside in: rank r takes n_r = remaining
    units / remaining ranks, x of them from the deep end and n_r - x from the
    shallow end, x the largest with attention <= T. T is bisected so the last
    rank (the middle that is left) matches. Returns per rank a list of unit
    ranges [a, b).
    """
    nu = int(cost.numel())
    pre = torch.cat([torch.zeros(1, dtype=cost.dtype), cost.cumsum(0)])

    def seg(a, b):
        return float(pre[b] - pre[a]) if b > a else 0.0

    def fill(T):
        out, lo, hi = [], 0, nu
        for r in range(d - 1):
            n_r = -(-(hi - lo) // (d - r))

            def c(x):
                return seg(lo, lo + n_r - x) + seg(hi - x, hi)

            if c(n_r) <= T:
                x = n_r
            elif c(0) > T:
                x = 0
            else:
                a, b = 0, n_r
                while b - a > 1:
                    mid = (a + b) // 2
                    if c(mid) <= T:
                        a = mid
                    else:
                        b = mid
                x = a
            out.append([(lo, lo + n_r - x), (hi - x, hi)])
            lo, hi = lo + n_r - x, hi - x
        out.append([(lo, hi)])
        return out

    if d == 1 or nu == 0:
        return [[(0, nu)]] + [[] for _ in range(d - 1)]
    tot = seg(0, nu)
    lo_t, hi_t = tot / d * 0.8, tot / d * 1.5
    best, bestv = None, math.inf
    for _ in range(iters):
        T = (lo_t + hi_t) / 2
        rp = fill(T)
        A = [sum(seg(a, b) for a, b in p) for p in rp]
        if max(A) < bestv:
            best, bestv = rp, max(A)
        if A[-1] > T:
            lo_t = T
        else:
            hi_t = T
    return best


def rows_of_reqs(plan: StepPlan, reqs: Sequence[int]) -> torch.Tensor:
    """Ascending global row indices of the given requests."""
    parts = [torch.arange(*plan.req_rows[i], dtype=torch.int64)
             for i in sorted(reqs)]
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.int64)


def rows_c2e(plan: StepPlan, reqs: Optional[Sequence[int]] = None
             ) -> list[torch.Tensor]:
    """Ring / all-gather rows: per-request c2e, cuts snapped to plan.u.

    Returns out[rank] = global row indices (ascending).
    """
    d, u = plan.d, plan.u
    reqs = range(plan.num_reqs) if reqs is None else reqs
    per_rank = [[] for _ in range(d)]
    for i in reqs:
        lo, hi = plan.req_rows[i]
        if hi == lo:
            continue
        cost = _attn_cost_units(plan.qpos[lo:hi], u)
        parts = _two_ended(cost, d)
        for r, ranges in enumerate(parts):
            for a, b in ranges:
                a_t, b_t = lo + a * u, min(hi, lo + b * u)
                if b_t > a_t:
                    per_rank[r].append(torch.arange(a_t, b_t,
                                                    dtype=torch.int64))
    out = []
    for r in range(d):
        if per_rank[r]:
            out.append(torch.sort(torch.cat(per_rank[r])).values)
        else:
            out.append(torch.empty(0, dtype=torch.int64))
    return out
