# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-step time of MLA CP (MSA_MLA_CP) on d GPUs, for the server calibration.

Each rank is one process on its own GPU (NCCL PCP group), running vLLM's
DeepSeek-V2 modules (one decoder layer, dummy weights by default) through the
backend exactly as a PCP worker does in the replicated dense mode (see
tests/v1/attention/cp_holes/test_mla_cp_model_gloo.py, whose MiniRunner this
reuses). The latent cache of [0, E) is filled with random values, then each
layout's step is built (holes/core segment metadata, PCP slot mapping, the
backend's builder) and run; rank 0 writes one JSON line per (d, E, layout):
median step time (host build + forward, synchronised) and build time.

    python benchmarks/cp_holes/bench_mla_cp.py --model deepseek-ai/DeepSeek-V2-Lite \
        --world 2 --E 32768 131072 --out mla_cp.jsonl
    # no-CP baseline: --world 1 --backend MSA_MLA

Layouts (m = computed tokens of the step): tail (one 2048 tail at E), rate (a
2048 hole at E / 8 + 2048 tail), deep16 (16 x 16-token holes + 2048 tail),
holeonly4 (4 x 16-token holes, absorbed path), decode8 (8 requests decoding
at E). The model has one layer: layer 0 of DeepSeek-V2-Lite is dense, so the
time is attention + projections + dense MLP of one layer, with no MoE.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import torch
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "v1", "attention", "cp_holes"))


def layouts(E: int):
    t = 2048
    out = {
        "tail": [("r0", [(E - t, t)])],
        "rate": [("r0", [(E // 8, t), (E - t, t)])],
        "deep16": [("r0", [(E // 4 + k * (E // 40), 16) for k in range(16)] + [(E - t, t)])],
        "holeonly4": [("r0", [(E // 2 + k * 4096, 16) for k in range(4)])],
        "decode8": [(f"d{i}", [(E - 1, 1)]) for i in range(8)],
    }
    return out


def worker(rank, world, args, q):
    try:
        os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), MASTER_ADDR="127.0.0.1",
                          MASTER_PORT=str(args.port),
                          LOCAL_RANK=str(rank % torch.cuda.device_count()))
        from test_mla_cp_model_gloo import MiniRunner

        Emax = max(args.E)
        args.max_model_len = Emax + 64
        reqs = ["r0"] + [f"d{i}" for i in range(8)]
        per_req = -(-(Emax + 64) // (args.u * world))
        args.num_blocks = per_req * len(reqs) + 2
        run = MiniRunner(args, rank, world)
        for r in reqs:
            run.alloc(r, Emax + 64)
        g = torch.Generator(device=run.dev).manual_seed(1)
        for layer in run.layers.values():
            layer.kv_cache.normal_(0, 0.5, generator=g)
        seqs = {r: [100 + (i % 1000) for i in range(Emax + 64)] for r in reqs}
        bt = run.builder.build
        tb: list[float] = []

        def timed_build(*a, **kw):
            t0 = time.perf_counter()
            md = bt(*a, **kw)
            tb.append(time.perf_counter() - t0)
            return md

        run.builder.build = timed_build
        for E in args.E:
            for name, entries in layouts(E).items():
                if args.layouts and name not in args.layouts:
                    continue
                ts = []
                tb.clear()
                for it in range(args.warmup + args.iters):
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                    t0 = time.perf_counter()
                    run.step(entries, seqs)
                    torch.cuda.synchronize()
                    if it >= args.warmup:
                        ts.append(time.perf_counter() - t0)
                m = sum(n for _, segs in entries for _, n in segs)
                row = dict(backend=args.backend, d=world, E=E, layout=name, m=m, u=args.u,
                           step_ms=1e3 * statistics.median(ts),
                           build_ms=1e3 * statistics.median(tb[args.warmup:]),
                           dtype=args.dtype, gpu=torch.cuda.get_device_name(run.dev))
                if rank == 0:
                    print(json.dumps(row), flush=True)
                    if args.out:
                        with open(args.out, "a") as f:
                            f.write(json.dumps(row) + "\n")
        q.put((rank, None))
    except Exception:  # noqa: BLE001
        q.put((rank, traceback.format_exc()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--backend", default="MSA_MLA_CP")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--u", type=int, default=128, help="block size = interleave size")
    ap.add_argument("--E", type=int, nargs="+", default=[32768, 131072])
    ap.add_argument("--layouts", nargs="*", default=None)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--load-format", default="dummy")
    ap.add_argument("--hf-overrides", default='{"num_hidden_layers": 1}')
    ap.add_argument("--dist-backend", default="nccl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from test_mla_cp_model_gloo import free_port

    args.port = free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, args.world, args, q)) for r in range(args.world)]
    for p in ps:
        p.start()
    res = [q.get() for _ in range(args.world)]
    for p in ps:
        p.join()
    bad = [tb for _, tb in res if tb]
    for tb in bad:
        print(tb, flush=True)
    print("RESULT", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
