# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-rank attention kernel time of CP schemes on holed prefill steps, one GPU.

Each CP rank's local attention (its query rows, the keys it sees under the
scheme) is run through the MSA kernel on this GPU, one rank after another, and
timed. Communication is not modelled. Model shape: Llama-3.1-8B attention, one
layer (Hq 32, Hkv 8, head_dim 128, bf16), block size 16, interleave u = 128.

Schemes (per step, d ranks):
- allgather: rank r attends its rows over all keys [0, E) (all-gather KV; the
  cached and new passes are fused into one call here).
- ring: rank r at ring step t attends its rows over the buffer of rank
  (r - t) mod d = that rank's homed resident KV (interleaved u) + the new KV of
  that rank's rows. d calls per rank.
- passq: every rank attends all rows over its interleaved key blocks (u = 128).
- passq_contig: every rank attends all rows over a contiguous 1/d of [0, E).
- ulysses: every rank attends all rows with Hq/d query and Hkv/d KV heads.

Phases:
  dense   time one Llama-3.1-8B decoder layer's dense ops vs tokens
  calib   time single-call shards for the cost-model fit
  sweep   run the steps of a layouts file

Every timed call is preceded by an L2 flush; all calls of one (step, d) group
run round-robin for --reps rounds and the median is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import partitioners as P  # noqa: E402

from vllm.v1.attention.ops.msa import msa_attention, plan_msa_rows  # noqa: E402

HQ, HKV, DH, BS, U = 32, 8, 128, 16, 128
G = HQ // HKV
DEV = "cuda"
SCALE = 1.0 / math.sqrt(DH)

_flush_buf = None


def flush_l2():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)
    _flush_buf.zero_()


class Ctx:
    def __init__(self, e_max, m_max):
        self.e_max = e_max
        nblk = -(-e_max // BS) + 1
        g = torch.Generator(device=DEV).manual_seed(0)
        self.k = torch.randn(nblk, BS, HKV, DH, device=DEV, dtype=torch.bfloat16, generator=g)
        self.v = torch.randn(nblk, BS, HKV, DH, device=DEV, dtype=torch.bfloat16, generator=g)
        nsb = nblk + 16  # ring scratch: all d buffers together hold <= E keys + padding
        self.ks = torch.zeros(nsb, BS, HKV, DH, device=DEV, dtype=torch.bfloat16)
        self.vs = torch.zeros_like(self.ks)
        self.q = torch.randn(m_max, HQ, DH, device=DEV, dtype=torch.bfloat16, generator=g)
        self.out = torch.empty(m_max, HQ, DH, device=DEV, dtype=torch.bfloat16)
        self.lse = torch.empty(m_max, HQ, device=DEV, dtype=torch.float32)
        self.bt_all = torch.arange(nblk, device=DEV, dtype=torch.int32)[None, :]
        self.kflat = self.k.view(-1, HKV, DH)
        self.vflat = self.v.view(-1, HKV, DH)


def n_runs(p):
    if len(p) == 0:
        return 0
    return int(1 + np.count_nonzero(np.diff(p) != 1))


class Call:
    """One timed kernel launch with a prebuilt plan."""

    def __init__(self, meta, fn, plan_ms):
        self.meta, self.fn, self.plan_ms = meta, fn, plan_ms
        self.times = []


def _plan(qpos, kv_len, hkv, **kw):
    t0 = time.perf_counter()
    plan = plan_msa_rows(qpos, np.zeros(len(qpos), dtype=np.int64), np.asarray([kv_len]), hkv, G,
                         block_size=BS, mode=os.environ.get("MSA_MODE", "auto"), device=DEV,
                         elem_size=2, **kw)
    return plan, (time.perf_counter() - t0) * 1e3


def call_rows_full(ctx, qpos_rows, E, meta, hkv_slice=None, qsel=None):
    """Rows (positions) over all keys [0, E) of the contiguous cache."""
    n = len(qpos_rows)
    if hkv_slice is None:
        h0, h1 = 0, HKV
    else:
        h0, h1 = hkv_slice
    hkv = h1 - h0
    q = ctx.q[:n, h0 * G:h1 * G] if qsel is None else qsel
    kc, vc = ctx.k[:, :, h0:h1], ctx.v[:, :, h0:h1]
    plan, pms = _plan(qpos_rows, E, hkv)
    out = torch.empty(n, hkv * G, DH, device=DEV, dtype=torch.bfloat16) if hkv_slice else ctx.out[:n]
    lse = torch.empty(n, hkv * G, device=DEV, dtype=torch.float32) if hkv_slice else ctx.lse[:n]

    def fn():
        msa_attention(q, kc, vc, ctx.bt_all, plan, softmax_scale=SCALE, out=out, lse=lse)
    meta = dict(meta, n_rows=n, sum_pos=int(qpos_rows.sum()) if n else 0, runs=n_runs(qpos_rows),
                n_keys=int(E), heads=hkv * G)
    c = Call(meta, fn, pms)
    c.out, c.lse = out, lse
    return c


def call_rows_blocks(ctx, qpos_rows, blocks, E, meta):
    """Rows over a subset of physical blocks (paged in place, per-block start positions)."""
    n = len(qpos_rows)
    blocks = np.asarray(blocks, dtype=np.int64)
    starts = blocks * BS
    kv_len = int(np.clip(E - starts, 0, BS).sum())
    nb = len(blocks)
    if nb == 0 or kv_len == 0:
        return None
    kblk = starts[None, :]
    plan, pms = _plan(qpos_rows, kv_len, HKV, kblk_pos=kblk)
    bt = torch.from_numpy(blocks.astype(np.int32))[None, :].to(DEV)
    kp = torch.from_numpy(starts.astype(np.int32))[None, :].to(DEV)
    q, out, lse = ctx.q[:n], ctx.out[:n], ctx.lse[:n]

    def fn():
        msa_attention(q, ctx.k, ctx.v, bt, plan, kblk_pos=kp, softmax_scale=SCALE, out=out, lse=lse)
    meta = dict(meta, n_rows=n, sum_pos=int(qpos_rows.sum()), runs=n_runs(qpos_rows), n_keys=kv_len,
                heads=HQ)
    return Call(meta, fn, pms)


class RingBuffers:
    """The d ring buffers of one step, gathered into the scratch cache."""

    def __init__(self, ctx, qpos, rows, E, d):
        comp = np.zeros(E, dtype=bool)
        comp[qpos] = True
        allpos = np.arange(E, dtype=np.int64)
        resident = allpos[~comp]
        home = (resident // U) % d
        self.bufs = []
        base = 0
        for s in range(d):
            kp = np.sort(np.concatenate([resident[home == s], qpos[rows[s]]]))
            nk = len(kp)
            nb = max(1, -(-nk // BS))
            dst = torch.arange(base * BS, base * BS + nk, device=DEV)
            src = torch.from_numpy(kp).to(DEV)
            ctx.ks.view(-1, HKV, DH)[dst] = ctx.kflat[src]
            ctx.vs.view(-1, HKV, DH)[dst] = ctx.vflat[src]
            bt = torch.arange(base, base + nb, device=DEV, dtype=torch.int32)[None, :]
            tab = np.full((1, nb * BS), 1 << 30, dtype=np.int64)
            tab[0, :nk] = kp
            self.bufs.append((kp, bt, torch.from_numpy(tab.astype(np.int32)).to(DEV)))
            base += nb


def call_ring(ctx, ring, qpos_rows, src, meta, qsel=None):
    kp, bt, tab = ring.bufs[src]
    n = len(qpos_rows)
    if n == 0:
        return None
    plan, pms = _plan(qpos_rows, len(kp), HKV, key_pos=[kp])
    q, out, lse = (ctx.q[:n] if qsel is None else qsel), ctx.out[:n], ctx.lse[:n]

    def fn():
        msa_attention(q, ctx.ks, ctx.vs, bt, plan, kblk_pos=tab, softmax_scale=SCALE, out=out, lse=lse)
    meta = dict(meta, n_rows=n, sum_pos=int(qpos_rows.sum()), runs=n_runs(qpos_rows),
                n_keys=int(len(kp)), n_vis=int(np.searchsorted(kp, qpos_rows, side="right").sum()),
                heads=HQ)
    return Call(meta, fn, pms)


def run_calls(calls, reps):
    calls = [c for c in calls if c is not None]
    # warm-up (compiles every launch config once)
    for c in calls:
        c.fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        evs = []
        for c in calls:
            flush_l2()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            c.fn()
            e.record()
            evs.append((c, s, e))
        torch.cuda.synchronize()
        for c, s, e in evs:
            c.times.append(s.elapsed_time(e) * 1e3)
    return calls


def emit(calls, fh):
    for c in calls:
        t = sorted(c.times)
        rec = dict(c.meta, t_us=float(np.median(t)), t_min=float(t[0]), t_max=float(t[-1]),
                   plan_ms=round(c.plan_ms, 3))
        fh.write(json.dumps(rec) + "\n")
    fh.flush()


# --------------------------------------------------------------------- phases


def step_calls(ctx, step, d, schemes, parts, cal):
    segs = [tuple(x) for x in step["segs"]]
    qpos = P.computed_positions(segs)
    E = int(qpos[-1]) + 1
    m = len(qpos)
    base = dict(layout=step["name"], set=step.get("set", ""), req=step.get("req"), stepi=step.get("stepi", 0),
                d=d, E=E, m=m, nseg=len(segs))
    calls = []
    if d == 1 or "t1" in schemes:
        calls.append(call_rows_full(ctx, qpos, E, dict(base, d=1, scheme="t1", part="-", rank=0)))
    if d == 1:
        yield calls
        return
    ring_keep = []
    for part in parts:
        if not ({"allgather", "ring"} & set(schemes)):
            break
        rows = P.partition(part, segs, d, cal=cal)
        assert sum(len(r) for r in rows) == m
        for r in range(d):
            qr = qpos[rows[r]]
            if "allgather" in schemes and len(qr):
                calls.append(call_rows_full(ctx, qr, E, dict(base, scheme="allgather", part=part, rank=r)))
        if "ring" in schemes:
            ring = RingBuffers(ctx, qpos, rows, E, d)
            ring_keep.append(ring)
            for r in range(d):
                qr = qpos[rows[r]]
                for t in range(d):
                    src = (r - t) % d
                    calls.append(call_ring(ctx, ring, qr, src,
                                           dict(base, scheme="ring", part=part, rank=r, rstep=t, src=src)))
            # one part's ring buffers occupy the scratch; time them before the next part
            yield_now = True
        else:
            yield_now = False
        if yield_now:
            yield calls
            calls = []
    nblk = -(-E // BS)
    blocks = np.arange(nblk)
    if "passq" in schemes:
        for r in range(d):
            mine = blocks[((blocks * BS) // U) % d == r]
            calls.append(call_rows_blocks(ctx, qpos, mine, E, dict(base, scheme="passq", part="interleave", rank=r)))
    if "passq_contig" in schemes:
        cuts = [min(nblk, round(nblk * r / d)) for r in range(d + 1)]
        for r in range(d):
            calls.append(call_rows_blocks(ctx, qpos, blocks[cuts[r]:cuts[r + 1]], E,
                                          dict(base, scheme="passq_contig", part="contig", rank=r)))
    if "ulysses" in schemes and HKV % d == 0:
        hk = HKV // d
        for r in range(d):
            calls.append(call_rows_full(ctx, qpos, E, dict(base, scheme="ulysses", part="heads", rank=r),
                                        hkv_slice=(r * hk, (r + 1) * hk)))
    yield calls


def phase_sweep(args):
    steps = json.load(open(args.layouts))
    if args.only:
        steps = [s for s in steps if any(o in s["name"] for o in args.only.split(","))]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        steps = steps[i::n]
    cal = json.load(open(args.cal)) if args.cal else None
    e_max = max(int(P.computed_positions(s["segs"])[-1]) + 1 for s in steps)
    m_max = max(len(P.computed_positions(s["segs"])) for s in steps)
    ctx = Ctx(e_max, m_max)
    schemes = args.schemes.split(",")
    parts = args.parts.split(",")
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            r = json.loads(line)
            done.add((r["layout"], r.get("req"), r.get("stepi", 0), r["d"]))
    fh = open(args.out, "a")
    t0 = time.time()
    for k, step in enumerate(steps):
        ds = step.get("d_list") or args.d
        for d in ds:
            if (step["name"], step.get("req"), step.get("stepi", 0), d) in done:
                continue
            ncalls = 0
            for calls in step_calls(ctx, step, d, schemes, parts, cal):
                calls = run_calls(calls, args.reps)
                emit(calls, fh)
                ncalls += len(calls)
            print(f"[{time.time() - t0:7.1f}s] step {k + 1}/{len(steps)} {step['name']} d={d} calls={ncalls}",
                  flush=True)
            torch.cuda.empty_cache()


def phase_calib(args):
    """Single-call shards: one or several pieces at varied depths, all keys up to E."""
    rng = np.random.default_rng(1)
    e_max, m_max = 131072, 16384
    ctx = Ctx(e_max, m_max)
    fh = open(args.out, "a")
    calls = []
    # (1) single piece q at depth s
    for qn in (16, 64, 256, 1024, 2048, 4096, 8192):
        for s in (0, 4096, 16384, 32768, 65536, 98304, 120000):
            if s + qn > e_max:
                continue
            qp = np.arange(s, s + qn, dtype=np.int64)
            calls.append(call_rows_full(ctx, qp, s + qn, dict(kind="single", q=qn, s=s)))
    # (2) random multi-piece shards (what partitioners produce): k pieces, total n, inside E
    for _ in range(120):
        E = int(rng.choice([16384, 32768, 65536, 100000, 131072]))
        k = int(rng.choice([1, 2, 3, 4, 8, 16]))
        n = int(rng.choice([256, 1024, 2048, 4096, 8192]))
        n = min(n, E // 2)
        cuts = np.sort(rng.choice(np.arange(1, n), size=k - 1, replace=False)) if k > 1 else []
        lens = np.diff(np.concatenate([[0], cuts, [n]])).astype(np.int64)
        starts = np.sort(rng.choice(np.arange(0, E - n), size=k, replace=False))
        pos, off = [], 0
        for st, ln in zip(starts, lens):
            s0 = int(st) + off
            pos.append(np.arange(s0, s0 + ln))
            off += int(ln)
        qp = np.concatenate(pos).astype(np.int64)
        qp = np.unique(np.clip(qp, 0, E - 1))
        calls.append(call_rows_full(ctx, qp, E, dict(kind="multi", k=k, E=E)))
    for i in range(0, len(calls), 24):
        emit(run_calls(calls[i:i + 24], args.reps), fh)
        print(f"calib {i + 24}/{len(calls)}", flush=True)


def phase_dense(args):
    """One Llama-3.1-8B decoder layer's dense part (no TP): norms, qkv, rope stand-in, o, gate_up, act, down."""
    H, I, QKV = 4096, 14336, 6144
    g = torch.Generator(device=DEV).manual_seed(0)
    w_qkv = torch.randn(H, QKV, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    w_o = torch.randn(H, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    w_gu = torch.randn(H, 2 * I, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    w_d = torch.randn(I, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02
    fh = open(args.out, "a")
    for n in (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        x = torch.randn(n, H, device=DEV, dtype=torch.bfloat16, generator=g)

        def layer():
            h = torch.nn.functional.rms_norm(x, (H,))
            qkv = h @ w_qkv
            a = qkv[:, :H] @ w_o
            h2 = torch.nn.functional.rms_norm(x + a, (H,))
            gu = h2 @ w_gu
            act = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
            return act @ w_d
        c = Call(dict(kind="dense", n=n), layer, 0.0)
        emit(run_calls([c], args.reps), fh)
        print(f"dense n={n} {np.median(c.times):.0f} us", flush=True)
        del x
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["dense", "calib", "sweep"])
    ap.add_argument("--layouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cal")
    ap.add_argument("--d", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--schemes", default="t1,allgather,ring,passq,passq_contig,ulysses")
    ap.add_argument("--parts", default=",".join(P.ROW_PARTITIONERS))
    ap.add_argument("--only", default="")
    ap.add_argument("--shard", default="")
    args = ap.parse_args()
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), sms=torch.cuda.get_device_properties(0).multi_processor_count,
                          torch=torch.__version__, msa_mode=os.environ.get("MSA_MODE", "auto"))), flush=True)
    {"dense": phase_dense, "calib": phase_calib, "sweep": phase_sweep}[args.phase](args)


if __name__ == "__main__":
    main()
