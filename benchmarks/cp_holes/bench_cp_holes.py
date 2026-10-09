# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-GPU correctness check and per-scheme step time of the CP schemes.

Run on the rented node (NCCL, one process per GPU):

    torchrun --nproc-per-node 8 benchmarks/cp_holes/bench_cp_holes.py \
        --check                                   # small shapes vs reference
    torchrun --nproc-per-node 8 benchmarks/cp_holes/bench_cp_holes.py \
        --layouts rate tail deep_small tiny8 --E 65536 --m 16384 \
        --out cp_bench.jsonl

Without torchrun it runs with world size 1 (no collectives; a GPU smoke).

Per scheme and layout it reports, per attention layer (mean over iterations,
max over ranks):
- total: the scheme's forward (packing + collectives + local attention +
  merges + cache writes), CUDA-event timed after a barrier;
- comm: the same forward with a null local attention (zeros), i.e. the
  communication, packing and merge overhead with no attention compute;
- attn: the local attention calls alone (summed), with the real data;
- bytes: bytes this rank sent per layer (from the plan).
The local attention is pluggable (`--local-attn`): `tiled` (PyTorch, default),
`ref`, `msa` (holes/kernel's MsaLocalAttention over gathered keys), or
`module:attr`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    from vllm.v1.attention import cp_holes as cph
    from vllm.v1.attention.cp_holes import schemes as _schemes  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(HERE, "..", "..", "vllm", "v1",
                                    "attention"))
    import cp_holes as cph  # type: ignore

KeySet = cph.local_attn.KeySet

# ------------------------------------------------------------------- layouts


def layout(name: str, E: int, m: int) -> list[list[tuple[int, int]]]:
    """Segments per request for named holed-prefill layouts.

    rate:        [front][hole 7.5K..+15.6K, truncated to m/2][back][tail]
                 (the `rate` hole anatomy, KERNEL-ALGO section 1)
    tail:        cached prefix, a tail chunk of m tokens at E (no hole)
    fresh:       a fresh prefill of m tokens from 0
    deep_small:  one 16-token piece at E/4 + a tail of m - 16 at E
    tiny8:       8 pieces of 64 tokens within 1K of E/4 + the tail
    hole_only:   a hole step that does not reach the tail (m tokens at 7.5K)
    multi:       4 requests of E/4 context, each a tail of m/4
    """
    if name == "rate":
        h = min(15600, m // 2)
        s0 = min(7500, E // 4)
        return [[(s0, h), (E - (m - h), m - h)]]
    if name == "tail":
        return [[(E - m, m)]]
    if name == "fresh":
        return [[(0, m)]]
    if name == "deep_small":
        return [[(E // 4, 16), (E - (m - 16), m - 16)]]
    if name == "tiny8":
        p = [(E // 4 + 128 * i, 64) for i in range(8)]
        return [p + [(E - (m - 512), m - 512)]]
    if name == "hole_only":
        return [[(min(7500, E // 4), m)]]
    if name == "multi":
        return [[(E // 4 - m // 4, m // 4)] for _ in range(4)]
    raise ValueError(name)


# ---------------------------------------------------------- stores, attention


class BenchKVStore:
    """Resident KV per request (contiguous, sorted) + the step's writes.

    reset() drops the writes so every iteration starts from the same state.
    """

    def __init__(self, base: dict[int, KeySet]):
        self.base = base
        self.reset()

    def reset(self):
        self.new: dict[int, list[KeySet]] = {}

    def write(self, keys: KeySet):
        if keys.n == 0:
            return
        for r in torch.unique(keys.kreq).tolist():
            sel = (keys.kreq == r).nonzero(as_tuple=True)[0]
            self.new.setdefault(r, []).append(keys.select(sel))

    def gather(self, reqs, limits):
        parts = []
        for r, lim in zip(reqs, limits):
            b = self.base.get(r)
            if b is not None and b.n:
                # base is sorted: a prefix
                hi = int(torch.searchsorted(b.kpos, torch.tensor(
                    [lim], device=b.kpos.device)).item())
                if hi:
                    parts.append(KeySet(b.k[:hi], b.v[:hi], b.kpos[:hi],
                                        b.kreq[:hi]))
            for ks in self.new.get(r, []):
                parts.append(ks)
        if not parts:
            b = next(iter(self.base.values()))
            return KeySet(b.k[:0], b.v[:0], b.kpos[:0], b.kreq[:0])
        return KeySet.cat(parts)


class NullAttention:

    def __call__(self, q, qpos, qreq, keys, scale):
        return (torch.zeros(q.shape, dtype=torch.float32, device=q.device),
                torch.zeros(q.shape[:2], dtype=torch.float32,
                            device=q.device))


class TimedAttention:
    """Wraps a local attention and accumulates its CUDA time."""

    def __init__(self, inner):
        self.inner, self.events = inner, []

    def __call__(self, *a):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = self.inner(*a)
        e.record()
        self.events.append((s, e))
        return out

    def take_ms(self):
        torch.cuda.synchronize()
        t = sum(s.elapsed_time(e) for s, e in self.events)
        self.events = []
        return t


def get_local_attn(spec: str):
    if spec == "msa":
        from vllm.v1.attention.ops.msa import MsaLocalAttention
        return MsaLocalAttention(mode="auto")
    if spec == "tiled":
        return cph.local_attn.TiledLocalAttention()
    if spec == "ref":
        return cph.RefLocalAttention()
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)()


# --------------------------------------------------------------------- driver


def placement_keep(name, plan, rank, world):
    if name in ("ulysses", "allgather"):
        return lambda p: torch.ones_like(p, dtype=torch.bool)
    return lambda p: cph.home_rank(p, world, plan.u) == rank


def build_store(name, plan, efull, K, V, rank, world, hq, hkv, paged=False,
                u=128):
    keep = placement_keep(name, plan, rank, world)
    if name == "ulysses":
        _, ks = cph.schemes.ulysses_heads(world, hq, hkv, rank)
    else:
        ks = slice(0, hkv)
    base = {}
    for i in range(plan.num_reqs):
        dev = K[i].device
        msk = torch.ones(efull[i], dtype=torch.bool, device=dev)
        for s, n in plan.segments[i]:
            msk[s:s + n] = False
        pos = msk.nonzero(as_tuple=True)[0]
        pos = pos[keep(pos)]
        base[i] = KeySet(K[i][pos][:, ks].contiguous(),
                         V[i][pos][:, ks].contiguous(), pos,
                         torch.full_like(pos, i))
    if paged:
        return PagedBenchStore(base, plan, efull, name, rank, world, u)
    return BenchKVStore(base)


class PagedBenchStore:
    """A vLLM-layout paged cache (block_size = u = interleave) wrapped in
    PagedKVStore with MSA in place: what the CP_HOLES backend runs. reset()
    restores the step-start validity; the per-step cache (tables, MSA plans)
    survives across iterations and layers, as within one step."""

    def __init__(self, base, plan, efull, name, rank, world, u):
        sharded = name not in ("ulysses", "allgather")
        w, r = (world, rank) if sharded else (1, 0)
        some = next(iter(base.values()))
        dev, dt = some.k.device, some.k.dtype
        hkv, dh = some.k.shape[1], some.k.shape[2]
        nblk = [-(-e // (u * w)) for e in efull]
        perm = torch.randperm(sum(nblk) + 1)
        tables, lps, off = [], [], 0
        for e, nb in zip(efull, nblk):
            bt, lp = cph.kv_store.vllm_local_layout(perm[off:off + nb], nb, u,
                                                    r, w, u)
            off += nb
            tables.append(bt.to(dev))
            lps.append(torch.where(lp < e, lp, torch.full_like(lp, -1))
                       .to(dev))
        cache = torch.zeros(2, len(perm), u, hkv, dh, dtype=dt, device=dev)
        pres = [torch.zeros_like(lp, dtype=torch.bool) for lp in lps]
        self.store = cph.kv_store.PagedKVStore(cache, tables, lps, pres,
                                               in_place=True, step_cache={})
        for ks in base.values():
            self.store.write(ks)
        self.present0 = [p.clone() for p in self.store.present]

    def reset(self):
        self.store.present = [p.clone() for p in self.present0]

    def write(self, keys):
        self.store.write(keys)

    def gather(self, reqs, limits):
        return self.store.gather(reqs, limits)

    def attend(self, q, qpos, qreq, reqs, limits, scale, attn, *a, **kw):
        if isinstance(attn, NullAttention):  # comm-only timing
            return attn(q, qpos, qreq, None, scale)
        if isinstance(attn, TimedAttention):  # time the in-place kernel
            st, en = torch.cuda.Event(True), torch.cuda.Event(True)
            st.record()
            out = self.store.attend(q, qpos, qreq, reqs, limits, scale,
                                    attn.inner, *a, **kw)
            en.record()
            attn.events.append((st, en))
            return out
        return self.store.attend(q, qpos, qreq, reqs, limits, scale, attn,
                                 *a, **kw)


def bytes_sent(scheme, plan, rank, world, hq, hkv, dh, elt=2):
    """Bytes this rank puts on the wire per layer (excluding self-copies)."""
    d = world
    if d == 1:
        return 0
    q, kv, o, lse = hq * dh * elt, 2 * hkv * dh * elt, hq * dh * elt, hq * 4
    n_own = int(scheme.own_rows.numel())
    name = scheme.name
    m = int(sum(int(r.numel()) for r in scheme.rows_by_rank))
    if name == "ulysses":
        hq_l, hkv_l = scheme.hq_l, scheme.hkv_l
        return (d - 1) * n_own * (hq_l + 2 * hkv_l) * dh * elt + \
            (m - n_own) * hq_l * dh * elt
    if name == "passq":
        return (d - 1) * n_own * q + n_own * kv * (d - 1) // d + \
            (m - n_own) * (o + lse)
    if name == "ring":
        return sum(int(scheme.buf_pos[(rank - t) % d].numel())
                   for t in range(d - 1)) * kv
    if name == "allgather":
        return (d - 1) * n_own * kv
    if name == "adaptive":
        return sum(bytes_sent(s, plan, rank, world, hq, hkv, dh, elt)
                   for s in (scheme.ring, scheme.pq) if s is not None)
    return 0


def run_case(args, name, lay, rank, world, comm, dev, local_attn):
    segs = layout(lay, args.E, args.m)
    plan = cph.StepPlan(segs, d=world, u=args.u)
    efull = [max(plan.e_step[i], args.E) for i in range(plan.num_reqs)]
    hq, hkv, dh, dt = args.hq, args.hkv, args.dh, getattr(torch, args.dtype)
    g = torch.Generator(device=dev).manual_seed(1234)
    K = [torch.randn(e, hkv, dh, generator=g, device=dev, dtype=dt)
         for e in efull]
    V = [torch.randn(e, hkv, dh, generator=g, device=dev, dtype=dt)
         for e in efull]
    Qall = torch.randn(plan.m, hq, dh, generator=g, device=dev, dtype=dt)
    scale = 1.0 / math.sqrt(dh)
    kw = {}
    if name == "adaptive":
        kw["threshold"] = args.passq_ratio
    if name in ("ulysses", "passq"):
        kw["n_mb"] = args.n_mb
    timed = TimedAttention(local_attn)
    scheme = cph.make_scheme(name, plan, comm, hq, hkv, dh, timed, **kw)
    own = scheme.own_rows.to(dev)
    qpos, qreq = plan.qpos.to(dev), plan.qreq.to(dev)
    q = Qall[own]
    off = torch.tensor([0] + efull[:-1], device=dev).cumsum(0)
    kidx = off[qreq[own]] + qpos[own]
    k, v = torch.cat(K)[kidx], torch.cat(V)[kidx]
    store = build_store(name, plan, efull, K, V, rank, world, hq, hkv,
                        paged=args.store == "paged", u=args.u)
    res = {"scheme": name, "layout": lay, "d": world, "E": args.E,
           "m": plan.m, "own_rows": int(own.numel()),
           "bytes_sent": bytes_sent(scheme, plan, rank, world, hq, hkv, dh,
                                    torch.empty(0, dtype=dt).element_size())}

    if args.check:
        o = scheme.forward(q, k, v, store, scale)
        keys = KeySet(torch.cat(K), torch.cat(V),
                      torch.cat([torch.arange(e, device=dev) for e in efull]),
                      torch.cat([torch.full((e, ), i, device=dev,
                                            dtype=torch.int64)
                                 for i, e in enumerate(efull)]))
        ro, _ = local_attn(Qall[own], qpos[own], qreq[own], keys, scale)
        res["max_err"] = (o.float() - ro).abs().max().item() \
            if own.numel() else 0.0
        store.reset()
        timed.take_ms()

    def timeit(attn_impl):
        scheme.attn = attn_impl
        for sub in (getattr(scheme, "ring", None), getattr(scheme, "pq",
                                                            None)):
            if sub is not None:
                sub.attn = attn_impl
        ts = []
        for it in range(args.warmup + args.iters):
            store.reset()
            if world > 1:
                dist.barrier()
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            for _ in range(args.layers):
                store.reset()
                scheme.forward(q, k, v, store, scale)
            e.record()
            torch.cuda.synchronize()
            if it >= args.warmup:
                ts.append(s.elapsed_time(e) / args.layers)
        return sum(ts) / len(ts)

    res["total_ms"] = timeit(timed)
    res["attn_ms"] = timed.take_ms() / ((args.warmup + args.iters) *
                                        args.layers)
    res["comm_ms"] = timeit(NullAttention())
    scheme.attn = timed
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="+", default=[
        "ulysses", "passq", "ring", "allgather", "adaptive"])
    ap.add_argument("--layouts", nargs="+", default=["rate", "tail",
                                                      "deep_small"])
    ap.add_argument("--E", type=int, default=32768)
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--hq", type=int, default=32)
    ap.add_argument("--hkv", type=int, default=8)
    ap.add_argument("--dh", type=int, default=128)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--u", type=int, default=128)
    ap.add_argument("--n-mb", type=int, default=2)
    ap.add_argument("--passq-ratio", type=float, default=0.2)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--local-attn", default="tiled")
    ap.add_argument("--store", default="dense", choices=["dense", "paged"],
                    help="paged: vLLM-layout paged cache; Ulysses / pass-Q "
                    "attend in place with MSA (the CP_HOLES backend path)")
    ap.add_argument("--check", action="store_true",
                    help="small shapes, compare with reference attention")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.check:
        args.E, args.m = min(args.E, 4096), min(args.m, 1024)
        args.layouts = ["rate", "tail", "deep_small", "tiny8", "hole_only",
                        "multi", "fresh"]
        args.iters, args.warmup, args.layers = 1, 0, 1

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=dev)
        comm = cph.CPComm(dist.group.WORLD, rank, world)
    else:
        comm = cph.CPComm(None, 0, 1)
    local_attn = get_local_attn(args.local_attn)
    rows = []
    for lay in args.layouts:
        for name in args.schemes:
            if name == "ulysses" and (args.hq % world or (
                    args.hkv % world and world % args.hkv)):
                continue
            t0 = time.time()
            r = run_case(args, name, lay, rank, world, comm, dev, local_attn)
            # max over ranks of the timings, sum of bytes
            keys = ["total_ms", "attn_ms", "comm_ms"] + \
                (["max_err"] if args.check else [])
            t = torch.tensor([r[x] for x in keys], device=dev,
                             dtype=torch.float64)
            b = torch.tensor([r["bytes_sent"]], device=dev,
                             dtype=torch.float64)
            if world > 1:
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                dist.all_reduce(b, op=dist.ReduceOp.MAX)
            for x, val in zip(keys, t.tolist()):
                r[x] = val
            r["bytes_sent_max_rank"] = int(b.item())
            r["wall_s"] = round(time.time() - t0, 1)
            rows.append(r)
            if rank == 0:
                msg = (f"{lay:10s} {name:9s} d={world} m={r['m']:6d} "
                       f"total={r['total_ms']:8.3f} ms attn={r['attn_ms']:8.3f}"
                       f" comm={r['comm_ms']:8.3f} MB_sent="
                       f"{r['bytes_sent_max_rank'] / 2**20:8.2f}")
                if args.check:
                    msg += f" max_err={r['max_err']:.2e}"
                print(msg, flush=True)
    if rank == 0 and args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if args.check and rank == 0:
        tol = 3e-2 if args.dtype != "float32" else 1e-4
        bad = [r for r in rows if r["max_err"] > tol]
        print("CHECK", "FAIL" if bad else "PASS", f"(tol {tol})")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
