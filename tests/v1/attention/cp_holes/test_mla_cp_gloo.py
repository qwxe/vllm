# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA pass-Q CP (vllm.v1.attention.cp_holes.mla) on `world` processes sharing
one GPU, gloo for the collectives (buffers staged through the host).

Every rank builds the same random holed steps (DeepSeek-V2-Lite dims), holds
only the resident latent it homes, runs the scheme for two layers, and checks
its own rows against a single-device dense reference over the full context,
plus its store: after the step it must hold the latent of every position
< E_step it homes.

    python tests/v1/attention/cp_holes/test_mla_cp_gloo.py --world 1 2 4
"""

from __future__ import annotations

import argparse
import math
import os
import random
import socket
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H, P, R, V, L = 16, 128, 64, 128, 512


def random_step(rng: random.Random):
    """[(segments [(start, n)], E_full)] for 1-2 requests."""
    reqs = []
    for _ in range(rng.choice([1, 2])):
        E = rng.randint(300, 2500)
        kind = rng.choice(["fresh", "tail", "holes", "tiny", "hole_only"])
        if kind == "fresh":
            segs = [(0, E)]
        elif kind == "tail":
            c = rng.randint(1, E - 1)
            segs = [(c, E - c)]
        elif kind == "hole_only":
            a = rng.randint(1, E // 2)
            segs = [(a, rng.randint(1, E // 3))]
        elif kind == "tiny":
            segs, p = [], rng.randint(10, 200)
            while p < E - 40 and len(segs) < 6:
                n = rng.randint(1, 8)
                segs.append((p, n))
                p += n + rng.randint(30, 400)
            segs.append((p, max(1, E - p)))
        else:
            segs, p = [], rng.randint(0, 200)
            for _ in range(rng.randint(1, 4)):
                if p >= E - 40:
                    break
                n = rng.randint(1, max(1, (E - p) // 3))
                segs.append((p, n))
                p += n + rng.randint(1, max(2, (E - p) // 3))
            if p < E:
                segs.append((p, E - p))
        e_full = max(s + n for s, n in segs) + rng.choice([0, 0, rng.randint(1, 100)])
        reqs.append((segs, e_full))
    return reqs


def worker(rank, world, port, seeds, u, q_out):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world)
        from vllm.v1.attention.cp_holes.mla import MlaLatentStore, MlaPassQCP
        from vllm.v1.attention.cp_holes.plan import StepPlan, home_rank
        from vllm.v1.attention.cp_holes.schemes import CPComm
        from vllm.v1.attention.ops.msa.api_mla import mla_reference

        dev = torch.device("cuda")
        dt = torch.bfloat16
        comm = CPComm.default()
        n_ok = n_all = 0
        worst = 0.0
        for seed in seeds:
            rng = random.Random(seed)
            reqs = random_step(rng)
            plan = StepPlan([list(s) for s, _ in reqs], world, u)
            g = torch.Generator().manual_seed(seed)
            lat_full = [torch.randn(e, L + R, generator=g).to(dt).to(dev) for _, e in reqs]
            w = (torch.randn(L, H * (P + V), generator=g) / math.sqrt(L)).to(dt).to(dev)
            scheme = MlaPassQCP(plan, comm, H, device=dev)
            store = MlaLatentStore([e for _, e in reqs], L + R, dt, dev)
            for i in range(plan.num_reqs):
                p = plan.resident_homed(i, rank).to(dev)
                store.write(lat_full[i][p], p, torch.full_like(p, i))
            own = scheme.own_rows
            new_lat = torch.cat([lat_full[int(plan.qreq[j])][int(plan.qpos[j])][None]
                                 for j in own.tolist()]) if own.numel() else \
                torch.empty(0, L + R, dtype=dt, device=dev)
            scale = 1.0 / math.sqrt(P + R)
            for layer in range(2):
                q_full = torch.randn(plan.m, H, P + R, generator=g).to(dt).to(dev)
                o = scheme.forward(q_full[own.to(dev)], new_lat, store,
                                   lambda x: x @ w, scale)
                # reference over all keys < E_step per request
                kpos, kreq, lats = [], [], []
                for i in range(plan.num_reqs):
                    e = plan.e_step[i]
                    kpos.append(torch.arange(e, device=dev))
                    kreq.append(torch.full((e, ), i, device=dev))
                    lats.append(lat_full[i][:e])
                lat = torch.cat(lats).float()
                kvn = (lat[:, :L] @ w.float()).view(-1, H, P + V)
                qo = q_full[own.to(dev)]
                o_ref, _ = mla_reference(qo[..., :P], qo[..., P:], kvn[..., :P], lat[:, L:],
                                         kvn[..., P:], plan.qpos[own].to(dev),
                                         plan.qreq[own].to(dev), torch.cat(kpos),
                                         torch.cat(kreq), scale)
                err = float((o.float() - o_ref).abs().max()) if own.numel() else 0.0
                worst = max(worst, err)
                n_all += 1
                n_ok += err < 2.5e-2
                if err >= 2.5e-2:
                    print(f"rank {rank} seed {seed} layer {layer} err {err:.3e} reqs {reqs}",
                          flush=True)
            # store check: every homed position < E_step holds the true latent
            for i in range(plan.num_reqs):
                p = torch.arange(plan.e_step[i])
                p = p[home_rank(p, world, u) == rank].to(dev)
                n_all += 1
                ok = bool(torch.equal(store.buf[i][p], lat_full[i][p]))
                n_ok += ok
                if not ok:
                    print(f"rank {rank} seed {seed} store mismatch req {i}", flush=True)
        q_out.put((rank, n_ok, n_all, worst))
        dist.barrier()
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        q_out.put((rank, 0, 1, float("inf")))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run(world: int, n_steps: int, u: int) -> bool:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    seeds = list(range(1000 * world, 1000 * world + n_steps))
    port = free_port()
    ps = [ctx.Process(target=worker, args=(r, world, port, seeds, u, q)) for r in range(world)]
    for p in ps:
        p.start()
    res = [q.get() for _ in range(world)]
    for p in ps:
        p.join()
    ok = sum(r[1] for r in res)
    tot = sum(r[2] for r in res)
    worst = max(r[3] for r in res)
    print(f"world={world} u={u}: {ok}/{tot} checks pass, worst |dO| {worst:.3e}", flush=True)
    return ok == tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--u", type=int, nargs="+", default=[128, 16])
    args = ap.parse_args()
    ok = True
    for w in args.world:
        for u in args.u:
            ok &= run(w, args.steps, u)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
