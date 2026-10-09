# SPDX-License-Identifier: Apache-2.0
"""CPU (gloo) correctness tests of the CP schemes for holed prefills.

Each test spawns `world` CPU processes, builds the same random holed step on
every rank, runs a scheme for several layers and compares every rank's output
rows with single-device reference attention over the full context. It also
checks the cache writes: after the step, each rank's store must hold exactly
the new K/V its placement homes.

Run directly (one world size at a time keeps host memory small):
    python tests/v1/attention/cp_holes/test_cp_gloo.py --world 4
or through pytest (world sizes 1, 2, 4, 8).
"""

from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys
import traceback

try:
    import pytest
except ImportError:  # direct runs need no pytest
    pytest = None
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from vllm.v1.attention.cp_holes import schemes as _s  # noqa: F401
    from vllm.v1.attention import cp_holes as cph
except ImportError:  # standalone staging tree (no vLLM install)
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                         "..", ".."))
    sys.path.insert(0, os.path.join(_root, "vllm", "v1", "attention"))
    import cp_holes as cph  # type: ignore

from_plan = cph.plan
KeySet = cph.local_attn.KeySet
RefKVStore = cph.kv_store.RefKVStore

# --------------------------------------------------------------- layouts


def random_segments(rng: random.Random, kind: str) -> tuple[list, int]:
    """Return (segments, E_full) for one request. E_full >= E_step: resident
    keys past E_step exist and must be ignored (causal truncation)."""
    E = rng.randint(96, 700)
    extra = rng.choice([0, 0, rng.randint(1, 200)])
    if kind == "fresh":
        return [(0, E)], E + extra
    if kind == "tail":  # cached prefix, new tail
        c = rng.randint(1, E - 1)
        return [(c, E - c)], E + extra
    if kind == "hole_only":  # a hole piece computed, the tail not yet
        a = rng.randint(1, E // 2)
        n = rng.randint(1, max(1, E // 3))
        return [(a, n)], E + extra + rng.randint(1, 100)
    if kind == "tiny":  # sub-tile pieces deep in the context
        segs, p = [], rng.randint(10, 60)
        while p < E - 8 and len(segs) < 5:
            n = rng.randint(1, 6)
            segs.append((p, n))
            p += n + rng.randint(5, 120)
        segs.append((p, max(1, E - p)))
        return segs, max(E, p + max(1, E - p)) + extra
    # "holes": several holes + tail
    segs, p = [], rng.randint(0, 80)
    nh = rng.randint(1, 4)
    for _ in range(nh):
        if p >= E - 20:
            break
        n = rng.randint(1, max(1, (E - p) // 3))
        segs.append((p, n))
        p += n + rng.randint(1, max(2, (E - p) // 3))
    if p < E:
        segs.append((p, E - p))
    return segs, max(s + n for s, n in segs) + extra


def build_case(seed: int, d: int, u: int, n_reqs: int):
    rng = random.Random(seed)
    kinds = ["holes", "tiny", "tail", "fresh", "hole_only"]
    segs, efull = [], []
    for i in range(n_reqs):
        s, e = random_segments(rng, kinds[i % len(kinds)] if i < 5 else
                               rng.choice(kinds))
        segs.append(s)
        efull.append(e)
    return from_plan.StepPlan(segs, d=d, u=u), efull


def make_tensors(plan, efull, hq, hkv, dh, layers, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    K = [[torch.randn(e, hkv, dh, generator=g).to(dtype) for e in efull]
         for _ in range(layers)]
    V = [[torch.randn(e, hkv, dh, generator=g).to(dtype) for e in efull]
         for _ in range(layers)]
    Q = [torch.randn(plan.m, hq, dh, generator=g).to(dtype)
         for _ in range(layers)]
    return Q, K, V


def reference(plan, efull, Q, K, V, scale):
    """Single-device attention of all computed rows over the full context."""
    attn = cph.RefLocalAttention()
    outs = []
    for L in range(len(Q)):
        ks = KeySet(torch.cat(K[L]), torch.cat(V[L]),
                    torch.cat([torch.arange(e) for e in efull]),
                    torch.cat([torch.full((e, ), i, dtype=torch.int64)
                               for i, e in enumerate(efull)]))
        o, lse = attn(Q[L], plan.qpos, plan.qreq, ks, scale)
        outs.append((o, lse))
    return outs


def resident_keys(plan, efull, i, L, K, V, keep):
    """Resident keys of request i (everything not computed this step, incl.
    positions past E_step), filtered by keep(pos) -> bool mask."""
    msk = torch.ones(efull[i], dtype=torch.bool)
    for s, n in plan.segments[i]:
        msk[s:s + n] = False
    pos = msk.nonzero(as_tuple=True)[0]
    pos = pos[keep(pos)]
    return KeySet(K[L][i][pos], V[L][i][pos], pos,
                  torch.full_like(pos, i))


# --------------------------------------------------------------- worker


def run_scheme(name, rank, world, cfg):
    hq, hkv, dh, layers, dtype = (cfg["hq"], cfg["hkv"], cfg["dh"],
                                  cfg["layers"], cfg["dtype"])
    plan, efull = build_case(cfg["seed"], world, cfg["u"], cfg["n_reqs"])
    Q, K, V = make_tensors(plan, efull, hq, hkv, dh, layers, dtype,
                           cfg["seed"] + 1)
    scale = 1.0 / math.sqrt(dh)
    comm = cph.CPComm(dist.group.WORLD if world > 1 else None, rank, world)
    kw = dict(cfg.get("kw", {}).get(name, {}))
    scheme = cph.make_scheme(name, plan, comm, hq, hkv, dh, **kw)

    if name == "ulysses":
        _, ks = cph.schemes.ulysses_heads(world, hq, hkv, rank)
        hkv_store = ks.stop - ks.start
    else:
        ks, hkv_store = slice(0, hkv), hkv

    def keep_fn(pos):
        if name in ("ulysses", "allgather"):
            return torch.ones_like(pos, dtype=torch.bool)
        return cph.home_rank(pos, world, plan.u) == rank

    Ks = [[k[:, ks] for k in KL] for KL in K]
    Vs = [[v[:, ks] for v in VL] for VL in V]
    ref = reference(plan, efull, Q, K, V, scale)
    own = scheme.own_rows
    errs = []
    for L in range(layers):
        store = RefKVStore(hkv_store, dh, dtype, torch.device("cpu"))
        for i in range(plan.num_reqs):
            store.write(resident_keys(plan, efull, i, L, Ks, Vs, keep_fn))
        o = scheme.forward(Q[L][own], _rows_kv(plan, K[L], own),
                           _rows_kv(plan, V[L], own), store, scale)
        ro = ref[L][0][own]
        err = (o.float() - ro).abs().max().item() if own.numel() else 0.0
        errs.append(err)
        # cache writes: every computed position this rank homes is present
        for i in range(plan.num_reqs):
            cm = plan.computed_mask(i).nonzero(as_tuple=True)[0]
            cm = cm[keep_fn(cm)]
            got = store.data.get(i)
            want_n = int(cm.numel()) + int(
                resident_keys(plan, efull, i, L, Ks, Vs, keep_fn).n)
            n = 0 if got is None else got.n
            assert n == want_n, f"{name} r{rank} req{i}: {n} != {want_n}"
            if cm.numel():
                sel = torch.searchsorted(got.kpos, cm)
                assert torch.equal(got.kpos[sel], cm)
                assert torch.equal(got.k[sel], Ks[L][i][cm]), "bad K write"
                assert torch.equal(got.v[sel], Vs[L][i][cm]), "bad V write"
    return max(errs), int(own.numel())


def _rows_kv(plan, KL, rows):
    """K or V of the given computed rows, gathered from per-request tensors."""
    if rows.numel() == 0:
        return KL[0][:0]
    return torch.stack([KL[int(plan.qreq[r])][int(plan.qpos[r])]
                        for r in rows.tolist()])


def _worker(rank, world, port, cfg, names, q):
    try:
        torch.set_num_threads(1)
        if world > 1:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(port)
            dist.init_process_group("gloo", rank=rank, world_size=world)
        res = {}
        for n in names:
            err, nrows = run_scheme(n, rank, world, cfg)
            tot = torch.tensor([nrows])
            if world > 1:
                dist.all_reduce(tot)
            res[n] = (err, int(tot))
        q.put((rank, res, None))
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
    except Exception:
        q.put((rank, None, traceback.format_exc()))


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run_world(world, cfg, names):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=_worker, args=(r, world, port, cfg, names, q))
          for r in range(world)]
    for p in ps:
        p.start()
    out = [q.get(timeout=600) for _ in range(world)]
    for p in ps:
        p.join(timeout=60)
    fails = [tb for _, _, tb in out if tb]
    if fails:
        raise AssertionError(fails[0])
    worst = {}
    for _, res, _ in out:
        for n, (e, tot) in res.items():
            worst[n] = (max(worst.get(n, (0, 0))[0], e), tot)
    return worst


ALL = ["ulysses", "passq", "ring", "allgather", "adaptive"]


def cases(world):
    """(cfg, tolerance) pairs for a world size."""
    out = []
    for hkv in sorted({2, 4, 8} & set(range(1, 9))):
        if 8 % world:  # Ulysses needs d | Hq
            continue
        for dtype, tol in ((torch.float32, 2e-5), (torch.bfloat16, 3e-2)):
            out.append((dict(hq=8, hkv=hkv, dh=16, layers=2, dtype=dtype,
                             seed=1000 * world + hkv, u=16, n_reqs=7,
                             kw={"adaptive": {"threshold": 0.35},
                                 "ulysses": {"n_mb": 2},
                                 "passq": {"n_mb": 2}}), tol))
    # interleave unit 128 (the design value) and single micro-batch
    out.append((dict(hq=8, hkv=2, dh=16, layers=1, dtype=torch.float32,
                     seed=77 + world, u=128, n_reqs=5,
                     kw={"ulysses": {"n_mb": 1}, "passq": {"n_mb": 1}}),
                2e-5))
    return out


def _parametrize(fn):
    if pytest is None:
        return fn
    return pytest.mark.parametrize("world", [1, 2, 4, 8])(fn)


@_parametrize
def test_cp_schemes(world):
    for cfg, tol in cases(world):
        worst = run_world(world, cfg, ALL)
        for n, (e, tot) in worst.items():
            plan, _ = build_case(cfg["seed"], world, cfg["u"], cfg["n_reqs"])
            assert tot == plan.m, f"{n}: rows {tot} != {plan.m}"
            assert e <= tol, f"{n} world={world} {cfg['dtype']} " \
                f"hkv={cfg['hkv']}: max err {e} > {tol}"


def test_merge_and_c2e():
    # LSE merge of key splits equals the unsplit attention
    torch.manual_seed(0)
    a = cph.RefLocalAttention(key_chunk=7)
    b = cph.RefLocalAttention(key_chunk=10**9)
    q = torch.randn(20, 4, 8)
    ks = KeySet(torch.randn(50, 2, 8), torch.randn(50, 2, 8),
                torch.randperm(50), torch.zeros(50, dtype=torch.int64))
    qp = torch.randint(-1, 60, (20, ))
    o1, l1 = a(q, qp, torch.zeros(20, dtype=torch.int64), ks, 0.3)
    o2, l2 = b(q, qp, torch.zeros(20, dtype=torch.int64), ks, 0.3)
    assert torch.allclose(o1, o2, atol=1e-5)
    assert torch.equal(torch.isinf(l1), torch.isinf(l2))
    fin = ~torch.isinf(l1)
    assert torch.allclose(l1[fin], l2[fin], atol=1e-5)
    # c2e covers every row exactly once, cuts on u
    plan, _ = build_case(5, 4, 16, 6)
    rows = from_plan.rows_c2e(plan)
    allr = torch.sort(torch.cat(rows)).values
    assert torch.equal(allr, torch.arange(plan.m))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, nargs="+", default=[2])
    ap.add_argument("--schemes", nargs="+", default=ALL)
    args = ap.parse_args()
    test_merge_and_c2e()
    for w in args.world:
        for cfg, tol in cases(w):
            worst = run_world(w, cfg, args.schemes)
            for n, (e, tot) in worst.items():
                ok = "OK " if e <= tol else "BAD"
                print(f"{ok} world={w} {n:9s} hkv={cfg['hkv']} u={cfg['u']} "
                      f"{str(cfg['dtype'])[6:]:8s} rows={tot} "
                      f"max_err={e:.2e} tol={tol:.0e}", flush=True)
