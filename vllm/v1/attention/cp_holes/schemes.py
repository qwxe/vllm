# SPDX-License-Identifier: Apache-2.0
"""Context-parallel schemes for holed prefills (KERNEL-ALGO.md section 4).

Every scheme is built once per step from the global `StepPlan` (identical on
all CP ranks) and then called once per attention layer:

    scheme = make_scheme("passq", plan, comm, hq, hkv, head_dim)
    # rows this rank runs the dense layers on (ascending global row indices)
    scheme.own_rows
    o = scheme.forward(q, k, v, store, scale)

q: [n_own, Hq, D], k / v: [n_own, Hkv, D] are the outputs of qkv_proj (+rope)
for the rank's own rows; o: [n_own, Hq, D] goes to o_proj. `store` is the
rank's local KV of this layer (see kv_store.py). The scheme writes the step's
new K/V into the store wherever the placement homes them, and attends with the
position-explicit local primitive (local_attn.py).

Wire formats: Q / K / V / O travel in the model dtype (bf16 in practice), LSE
in fp32 (decision 4.D). Positions never travel: every rank derives them from
the plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.distributed as dist

from .comm import (all_gather_bytes, all_to_all_bytes, pack_rows,
                   ring_exchange, row_bytes, unpack_rows, wait_all)
from .kv_store import CPKVStore
from .local_attn import (KeySet, LocalAttention, get_local_attention,
                         merge_attn_states)
from .plan import StepPlan, home_rank, rows_c2e, rows_contiguous, rows_of_reqs

PASSQ_RATIO = 0.2  # pass-Q when computed / E_step < this (KERNEL-ALGO 4.6)


@dataclass
class CPComm:
    """The CP process group as seen by the schemes."""

    group: Optional[dist.ProcessGroup]
    rank: int
    world: int

    @staticmethod
    def from_coordinator(gc) -> "CPComm":
        """From a vLLM GroupCoordinator (e.g. get_pcp_group())."""
        grp = gc.device_group if gc.world_size > 1 else None
        return CPComm(grp, gc.rank_in_group, gc.world_size)

    @staticmethod
    def default() -> "CPComm":
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return CPComm(None, 0, 1)
        return CPComm(dist.group.WORLD, dist.get_rank(), dist.get_world_size())


def _reqs_and_limits(plan: StepPlan, rows: torch.Tensor):
    reqs = torch.unique(plan.qreq[rows]).tolist() if rows.numel() else []
    return reqs, [plan.e_step[i] for i in reqs]


class _Scheme:
    name = "base"

    def __init__(self, plan: StepPlan, comm: CPComm, hq: int, hkv: int,
                 head_dim: int, local_attn: Optional[LocalAttention] = None):
        assert plan.d == comm.world, (plan.d, comm.world)
        self.plan, self.comm = plan, comm
        self.hq, self.hkv, self.dh = hq, hkv, head_dim
        self.attn = local_attn or get_local_attention()
        self.own_rows = torch.empty(0, dtype=torch.int64)

    def _to(self, t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return t.to(ref.device, non_blocking=True)


# --------------------------------------------------------------------- Ulysses


def ulysses_heads(d: int, hq: int, hkv: int, r: int):
    """(q head slice, kv head slice) of rank r. d > hkv replicates KV heads."""
    assert hq % d == 0, f"Ulysses needs d | Hq ({d}, {hq})"
    hq_l = hq // d
    if d <= hkv:
        assert hkv % d == 0, f"Ulysses needs d | Hkv or Hkv | d ({d}, {hkv})"
        hkv_l = hkv // d
        kv0 = r * hkv_l
    else:
        assert d % hkv == 0, f"Ulysses needs d | Hkv or Hkv | d ({d}, {hkv})"
        hkv_l = 1
        kv0 = r * hkv // d
    return slice(r * hq_l, (r + 1) * hq_l), slice(kv0, kv0 + hkv_l)


class UlyssesCP(_Scheme):
    """Head-sharded CP (4.1). d <= Hkv: KV heads split; d > Hkv: replicated.

    Rows for the dense layers: `n_mb` micro-batches, each cut into d contiguous
    pieces. Per micro-batch: a2a#1 (Q, K, V of own rows -> heads), write the
    micro-batch's K/V (rank's KV heads, all rows), attention over all rows of
    the micro-batch at Hq/d heads, a2a#2 (O -> row owners). The a2a#1 of
    micro-batch k+1 and the a2a#2 of micro-batch k run while micro-batch k / k+1
    attends.
    """

    name = "ulysses"

    def __init__(self, plan, comm, hq, hkv, head_dim, local_attn=None,
                 n_mb: int = 2, rows: Optional[torch.Tensor] = None):
        super().__init__(plan, comm, hq, hkv, head_dim, local_attn)
        d, r = comm.world, comm.rank
        self.heads = [ulysses_heads(d, hq, hkv, x) for x in range(d)]
        self.hq_l = hq // d
        self.hkv_l = self.heads[0][1].stop - self.heads[0][1].start
        sub = plan.m if rows is None else rows
        self.n_mb = n_mb
        self.mb_rows = rows_contiguous(sub, d, n_mb)
        self.own_rows = torch.cat([self.mb_rows[r][k] for k in range(n_mb)])
        self.cnt = [[int(self.mb_rows[x][k].numel()) for k in range(n_mb)]
                    for x in range(d)]
        self.mb_all = [torch.cat([self.mb_rows[x][k] for x in range(d)])
                       for k in range(n_mb)]
        self.mb_meta = []
        for k in range(n_mb):
            rows_k = self.mb_all[k]
            reqs, lim = _reqs_and_limits(plan, rows_k)
            self.mb_meta.append((plan.qpos[rows_k], plan.qreq[rows_k], reqs,
                                 lim))

    def forward(self, q, k, v, store: CPKVStore, scale: float):
        d, me, dt = self.comm.world, self.comm.rank, q.dtype
        D = self.dh
        spec1 = [((self.hq_l, D), dt), ((self.hkv_l, D), dt),
                 ((self.hkv_l, D), dt)]
        rb1 = sum(row_bytes(*s) for s in spec1)
        rbo = row_bytes((self.hq_l, D), dt)
        offs = [0]
        for k_ in range(self.n_mb):
            offs.append(offs[-1] + self.cnt[me][k_])

        def start1(kb):
            sl = slice(offs[kb], offs[kb + 1])
            qm, km, vm = q[sl], k[sl], v[sl]
            send = [pack_rows([qm[:, qs], km[:, ks], vm[:, ks]]).reshape(-1)
                    for qs, ks in self.heads]
            recv = [self.cnt[x][kb] * rb1 for x in range(d)]
            return all_to_all_bytes(send, recv, self.comm.group,
                                    async_op=True)

        pend2 = []
        nxt = start1(0)
        for kb in range(self.n_mb):
            bufs, w = nxt
            wait_all([w])
            if kb + 1 < self.n_mb:
                nxt = start1(kb + 1)
            qa, ka, va = unpack_rows(torch.cat(bufs), spec1)
            qpos, qreq, reqs, lim = self.mb_meta[kb]
            qpos, qreq = self._to(qpos, q), self._to(qreq, q)
            store.write(KeySet(ka, va, qpos, qreq))
            keys = store.gather(reqs, lim)
            o, _ = self.attn(qa, qpos, qreq, keys, scale)
            o = o.to(dt)
            send, off = [], 0
            for x in range(d):
                n = self.cnt[x][kb]
                send.append(pack_rows([o[off:off + n]]).reshape(-1))
                off += n
            recv = [self.cnt[me][kb] * rbo for _ in range(d)]
            pend2.append(all_to_all_bytes(send, recv, self.comm.group,
                                          async_op=True))
        outs = []
        for bufs, w in pend2:
            wait_all([w])
            outs.append(torch.cat(
                [unpack_rows(b, [((self.hq_l, D), dt)])[0] for b in bufs],
                dim=1))
        return torch.cat(outs) if outs else q.new_empty(0, self.hq, D)


# ---------------------------------------------------------------------- pass-Q


class PassQCP(_Scheme):
    """KV-stationary CP over interleaved placement (4.3).

    Per micro-batch: one all-to-all carries Q of the own rows to every rank
    (the all-gather of Q) and each new K/V row to its home rank only; each rank
    writes the K/V it homes, attends all rows of the micro-batch over its local
    keys (positions are strided, the per-row limit is the causal mask), and a
    second all-to-all returns (O in the model dtype, LSE in fp32) to the row
    owners, which merge the d partials by LSE. This is vLLM DCP's a2a backend
    (`dcp_a2a_lse_reduce`) scattered by rows instead of heads.
    """

    name = "passq"

    def __init__(self, plan, comm, hq, hkv, head_dim, local_attn=None,
                 n_mb: int = 2, rows: Optional[torch.Tensor] = None):
        super().__init__(plan, comm, hq, hkv, head_dim, local_attn)
        d, me = comm.world, comm.rank
        sub = plan.m if rows is None else rows
        self.n_mb = n_mb
        self.mb_rows = rows_contiguous(sub, d, n_mb)
        self.own_rows = torch.cat([self.mb_rows[me][k] for k in range(n_mb)])
        self.cnt = [[int(self.mb_rows[x][k].numel()) for k in range(n_mb)]
                    for x in range(d)]
        self.mb = []
        for k in range(n_mb):
            rows_k = torch.cat([self.mb_rows[x][k] for x in range(d)])
            reqs, lim = _reqs_and_limits(plan, rows_k)
            # my rows of this micro-batch homed on each rank (send side)
            mine = self.mb_rows[me][k]
            hm = home_rank(plan.qpos[mine], d, plan.u)
            send_sel = [(hm == x).nonzero(as_tuple=True)[0] for x in range(d)]
            # rows of each rank homed on me (receive side), in that order
            recv_rows = []
            for x in range(d):
                rx = self.mb_rows[x][k]
                recv_rows.append(rx[home_rank(plan.qpos[rx], d, plan.u) == me])
            rr = torch.cat(recv_rows)
            self.mb.append(dict(qpos=plan.qpos[rows_k], qreq=plan.qreq[rows_k],
                                reqs=reqs, lim=lim, send_sel=send_sel,
                                recv_kv=[int(t.numel()) for t in recv_rows],
                                kv_pos=plan.qpos[rr], kv_req=plan.qreq[rr]))

    def forward(self, q, k, v, store: CPKVStore, scale: float):
        d, me, dt = self.comm.world, self.comm.rank, q.dtype
        Hq, Hkv, D = self.hq, self.hkv, self.dh
        rq = row_bytes((Hq, D), dt)
        rkv = 2 * row_bytes((Hkv, D), dt)
        ro = row_bytes((Hq, D), dt) + row_bytes((Hq, ), torch.float32)
        offs = [0]
        for k_ in range(self.n_mb):
            offs.append(offs[-1] + self.cnt[me][k_])

        def start1(kb):
            sl = slice(offs[kb], offs[kb + 1])
            qm, km, vm = q[sl], k[sl], v[sl]
            qb = pack_rows([qm]).reshape(-1)
            send = []
            for x in range(d):
                sel = self._to(self.mb[kb]["send_sel"][x], q)
                send.append(torch.cat(
                    [qb, pack_rows([km[sel], vm[sel]]).reshape(-1)]))
            recv = [self.cnt[x][kb] * rq + self.mb[kb]["recv_kv"][x] * rkv
                    for x in range(d)]
            return all_to_all_bytes(send, recv, self.comm.group,
                                    async_op=True)

        pend2 = []
        nxt = start1(0)
        for kb in range(self.n_mb):
            bufs, w = nxt
            wait_all([w])
            if kb + 1 < self.n_mb:
                nxt = start1(kb + 1)
            meta = self.mb[kb]
            qs, kvs = [], []
            for x, b in enumerate(bufs):
                nq = self.cnt[x][kb] * rq
                qs.append(b[:nq])
                kvs.append(b[nq:])
            (qa, ) = unpack_rows(torch.cat(qs), [((Hq, D), dt)])
            ka, va = unpack_rows(torch.cat(kvs), [((Hkv, D), dt)] * 2)
            store.write(KeySet(ka, va, self._to(meta["kv_pos"], q),
                               self._to(meta["kv_req"], q)))
            qpos, qreq = self._to(meta["qpos"], q), self._to(meta["qreq"], q)
            keys = store.gather(meta["reqs"], meta["lim"])
            o, lse = self.attn(qa, qpos, qreq, keys, scale)
            o = o.to(dt)
            send, off = [], 0
            for x in range(d):
                n = self.cnt[x][kb]
                send.append(pack_rows([o[off:off + n],
                                       lse[off:off + n]]).reshape(-1))
                off += n
            recv = [self.cnt[me][kb] * ro for _ in range(d)]
            pend2.append(all_to_all_bytes(send, recv, self.comm.group,
                                          async_op=True))
        outs = []
        for bufs, w in pend2:
            wait_all([w])
            os_, ls_ = [], []
            for b in bufs:
                o_p, l_p = unpack_rows(b, [((Hq, D), dt),
                                           ((Hq, ), torch.float32)])
                os_.append(o_p)
                ls_.append(l_p)
            outs.append(merge_attn_states(os_, ls_)[0].to(dt))
        return torch.cat(outs) if outs else q.new_empty(0, Hq, D)


# ------------------------------------------------------------------ ring pass-KV


class RingCP(_Scheme):
    """Ring pass-KV over interleaved placement (4.2).

    Rows: per-request c2e (equal tokens and equal attention, cuts on u).
    Rank r's initial buffer = its resident homed KV of the step's requests,
    truncated at E_step, + the new K/V of its own rows. The buffers rotate d-1
    times (send to r+1, receive from r-1, double-buffered P2P overlapping the
    attention of the current buffer). Each rank keeps from every passing
    buffer the new K/V homed on it (the cache write) and merges its d partial
    states by LSE.
    """

    name = "ring"

    def __init__(self, plan, comm, hq, hkv, head_dim, local_attn=None,
                 reqs: Optional[Sequence[int]] = None,
                 rows_per_rank: Optional[list[torch.Tensor]] = None):
        super().__init__(plan, comm, hq, hkv, head_dim, local_attn)
        d, me = comm.world, comm.rank
        reqs = list(range(plan.num_reqs)) if reqs is None else sorted(reqs)
        self.rows_pr = rows_per_rank or rows_c2e(plan, reqs)
        self.own_rows = self.rows_pr[me]
        self.reqs = reqs
        self.lim = [plan.e_step[i] for i in reqs]
        self.buf_pos, self.buf_req, self.n_res, self.keep = [], [], [], []
        for x in range(d):
            rp = [plan.resident_homed(i, x) for i in reqs]
            res_pos = torch.cat(rp) if rp else torch.empty(0, dtype=torch.int64)
            res_req = torch.cat([torch.full_like(p, i)
                                 for p, i in zip(rp, reqs)]) if rp else \
                torch.empty(0, dtype=torch.int64)
            new_rows = self.rows_pr[x]
            npos, nreq = plan.qpos[new_rows], plan.qreq[new_rows]
            self.buf_pos.append(torch.cat([res_pos, npos]))
            self.buf_req.append(torch.cat([res_req, nreq]))
            self.n_res.append(int(res_pos.numel()))
            # new keys of x's buffer homed on me (written to my cache)
            sel = (home_rank(npos, d, plan.u) == me).nonzero(as_tuple=True)[0]
            self.keep.append(sel + self.n_res[x])
        self.qpos = plan.qpos[self.own_rows]
        self.qreq = plan.qreq[self.own_rows]

    def forward(self, q, k, v, store: CPKVStore, scale: float):
        d, me, dt = self.comm.world, self.comm.rank, q.dtype
        Hkv, D = self.hkv, self.dh
        spec = [((Hkv, D), dt)] * 2
        rkv = sum(row_bytes(*s) for s in spec)
        res = store.gather(self.reqs, self.lim)
        assert res.n == self.n_res[me], (res.n, self.n_res[me])
        buf = pack_rows([torch.cat([res.k.to(dt), k]),
                         torch.cat([res.v.to(dt), v])]).reshape(-1)
        qpos, qreq = self._to(self.qpos, q), self._to(self.qreq, q)
        o_parts, l_parts = [], []
        for t in range(d):
            src = (me - t) % d
            works = []
            if t < d - 1:
                nsrc = (me - t - 1) % d
                recv, works = ring_exchange(
                    buf, int(self.buf_pos[nsrc].numel()) * rkv, me, d,
                    self.comm.group)
            kk, vv = unpack_rows(buf, spec)
            keys = KeySet(kk, vv, self._to(self.buf_pos[src], q),
                          self._to(self.buf_req[src], q))
            o, lse = self.attn(q, qpos, qreq, keys, scale)
            o_parts.append(o)
            l_parts.append(lse)
            store.write(keys.select(self._to(self.keep[src], q)))
            if t < d - 1:
                wait_all(works)
                buf = recv
        return merge_attn_states(o_parts, l_parts)[0].to(dt)


# ---------------------------------------------------------------- all-gather KV


class AllGatherKVCP(_Scheme):
    """All-gather KV over replicated resident KV (AG-R, 4.4).

    Rows: per-request c2e. Only the step's new K/V is all-gathered. The
    attention over the cached keys starts before the all-gather finishes; the
    attention over the new keys follows, and the two are merged by LSE (the
    cascade-attention shape). Every rank writes every new K/V (replicated).
    """

    name = "allgather"

    def __init__(self, plan, comm, hq, hkv, head_dim, local_attn=None,
                 reqs: Optional[Sequence[int]] = None,
                 rows_per_rank: Optional[list[torch.Tensor]] = None):
        super().__init__(plan, comm, hq, hkv, head_dim, local_attn)
        reqs = list(range(plan.num_reqs)) if reqs is None else sorted(reqs)
        self.rows_pr = rows_per_rank or rows_c2e(plan, reqs)
        self.own_rows = self.rows_pr[comm.rank]
        self.reqs, self.lim = reqs, [plan.e_step[i] for i in reqs]
        self.qpos = plan.qpos[self.own_rows]
        self.qreq = plan.qreq[self.own_rows]

    def forward(self, q, k, v, store: CPKVStore, scale: float):
        dt, Hkv, D = q.dtype, self.hkv, self.dh
        spec = [((Hkv, D), dt)] * 2
        rkv = sum(row_bytes(*s) for s in spec)
        nbytes = [int(x.numel()) * rkv for x in self.rows_pr]
        outs, work = all_gather_bytes(pack_rows([k, v]).reshape(-1), nbytes,
                                      self.comm.group, async_op=True)
        qpos, qreq = self._to(self.qpos, q), self._to(self.qreq, q)
        cached = store.gather(self.reqs, self.lim)
        o1, l1 = self.attn(q, qpos, qreq, cached, scale)
        wait_all([work])
        new = []
        for x, b in enumerate(outs):
            kk, vv = unpack_rows(b, spec)
            rx = self.rows_pr[x]
            new.append(KeySet(kk, vv, self._to(self.plan.qpos[rx], q),
                              self._to(self.plan.qreq[rx], q)))
        new = KeySet.cat(new)
        store.write(new)
        o2, l2 = self.attn(q, qpos, qreq, new, scale)
        return merge_attn_states([o1, o2], [l1, l2])[0].to(dt)


# ------------------------------------------------- adaptive ring / pass-Q choice


class AdaptiveCP(_Scheme):
    """Per request-step choice between ring and pass-Q (4.6).

    Requests with computed / E_step < `threshold` run pass-Q, the rest run the
    ring; both share the interleaved placement, so the choice needs no KV
    movement. Own rows are the union of both schemes' rows.
    """

    name = "adaptive"

    def __init__(self, plan, comm, hq, hkv, head_dim, local_attn=None,
                 threshold: float = PASSQ_RATIO, n_mb: int = 1):
        super().__init__(plan, comm, hq, hkv, head_dim, local_attn)
        self.pq_reqs = [i for i in range(plan.num_reqs)
                        if plan.ratio(i) < threshold]
        self.ring_reqs = [i for i in range(plan.num_reqs)
                          if i not in self.pq_reqs]
        self.ring = RingCP(plan, comm, hq, hkv, head_dim, self.attn,
                           reqs=self.ring_reqs) if self.ring_reqs else None
        self.pq = PassQCP(plan, comm, hq, hkv, head_dim, self.attn, n_mb=n_mb,
                          rows=rows_of_reqs(plan, self.pq_reqs)) \
            if self.pq_reqs else None
        parts = [s.own_rows for s in (self.ring, self.pq) if s is not None]
        self.own_rows = torch.sort(torch.cat(parts)).values if parts else \
            torch.empty(0, dtype=torch.int64)
        self.idx = {}
        for s in (self.ring, self.pq):
            if s is not None:
                self.idx[s.name] = torch.searchsorted(self.own_rows,
                                                      s.own_rows)

    def forward(self, q, k, v, store: CPKVStore, scale: float):
        o = q.new_empty(q.shape)
        for s in (self.ring, self.pq):
            if s is None:
                continue
            ix = self._to(self.idx[s.name], q)
            o[ix] = s.forward(q[ix], k[ix], v[ix], store, scale)
        return o


SCHEMES = {
    "ulysses": UlyssesCP,
    "passq": PassQCP,
    "ring": RingCP,
    "allgather": AllGatherKVCP,
    "adaptive": AdaptiveCP,
}


def make_scheme(name: str, plan: StepPlan, comm: CPComm, hq: int, hkv: int,
                head_dim: int, local_attn: Optional[LocalAttention] = None,
                **kw) -> _Scheme:
    return SCHEMES[name](plan, comm, hq, hkv, head_dim, local_attn, **kw)
