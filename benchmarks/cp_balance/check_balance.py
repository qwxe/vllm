# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that the per-rank calls of bench_balance.py compose to the full attention.

For each scheme, the rank partials are LSE-merged per row and compared with the
single-call (t1) output of the step.
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_balance as bb  # noqa: E402
import partitioners as P  # noqa: E402


def run(c, ctx, n, heads):
    c.fn()
    torch.cuda.synchronize()
    if heads == bb.HQ:
        return ctx.out[:n].float().clone(), ctx.lse[:n].clone()
    raise RuntimeError


def merge(parts, m, H):
    o = torch.zeros(m, H, bb.DH, device=bb.DEV)
    lse = torch.full((m, H), float("-inf"), device=bb.DEV)
    for rows, po, pl in parts:
        r = torch.from_numpy(rows).to(bb.DEV)
        a_o, a_l = o[r], lse[r]
        mx = torch.maximum(a_l, pl)
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        wa, wb = torch.exp(a_l - mx), torch.exp(pl - mx)
        s = wa + wb
        o[r] = (a_o * wa[..., None] + po * wb[..., None]) / s.clamp_min(1e-30)[..., None]
        lse[r] = mx + torch.log(s)
    return o, lse


def main():
    segs = [(1000, 300), (5000, 77), (9000, 1500), (20000 - 700, 700)]
    qpos = P.computed_positions(segs)
    m, E = len(qpos), int(qpos[-1]) + 1
    ctx = bb.Ctx(E, m)
    ref_c = bb.call_rows_full(ctx, qpos, E, {})
    ref, _ = run(ref_c, ctx, m, bb.HQ)
    allrows = np.arange(m)
    worst = 0.0
    for d in (2, 4):
        for part in P.ROW_PARTITIONERS:
            rows = P.partition(part, segs, d, cal={"a": 1e-5, "b": 1e-9})
            parts = []
            for r in range(d):
                c = bb.call_rows_full(ctx, qpos[rows[r]], E, {}, qsel=ctx.q[torch.from_numpy(rows[r]).to(bb.DEV)].contiguous())
                o, l = run(c, ctx, len(rows[r]), bb.HQ)
                parts.append((rows[r], o, l))
            o, _ = merge(parts, m, bb.HQ)
            e1 = (o - ref).abs().max().item()
            ring = bb.RingBuffers(ctx, qpos, rows, E, d)
            parts = []
            for r in range(d):
                for t in range(d):
                    c = bb.call_ring(ctx, ring, qpos[rows[r]], (r - t) % d, {}, qsel=ctx.q[torch.from_numpy(rows[r]).to(bb.DEV)].contiguous())
                    if c is None:
                        continue
                    o, l = run(c, ctx, len(rows[r]), bb.HQ)
                    parts.append((rows[r], o, l))
            o, _ = merge(parts, m, bb.HQ)
            e2 = (o - ref).abs().max().item()
            print(f"d={d} {part:13s} allgather {e1:.2e} ring {e2:.2e}")
            worst = max(worst, e1, e2)
        nblk = -(-E // bb.BS)
        blocks = np.arange(nblk)
        for sch in ("passq", "passq_contig"):
            parts = []
            for r in range(d):
                if sch == "passq":
                    mine = blocks[((blocks * bb.BS) // bb.U) % d == r]
                else:
                    cuts = [min(nblk, round(nblk * x / d)) for x in range(d + 1)]
                    mine = blocks[cuts[r]:cuts[r + 1]]
                c = bb.call_rows_blocks(ctx, qpos, mine, E, {})
                o, l = run(c, ctx, m, bb.HQ)
                parts.append((allrows, o, l))
            o, _ = merge(parts, m, bb.HQ)
            e3 = (o - ref).abs().max().item()
            print(f"d={d} {sch:13s} {e3:.2e}")
            worst = max(worst, e3)
        # ulysses: head slices concatenated
        hk = bb.HKV // d
        outs = []
        for r in range(d):
            c = bb.call_rows_full(ctx, qpos, E, {}, hkv_slice=(r * hk, (r + 1) * hk))
            c.fn()
            torch.cuda.synchronize()
            outs.append(c.out.float().clone())
        o = torch.cat(outs, dim=1)
        e4 = (o - ref).abs().max().item()
        print(f"d={d} ulysses       {e4:.2e}")
        worst = max(worst, e4)
    print("CHECK", "PASS" if worst < 3e-2 else "FAIL", worst)


if __name__ == "__main__":
    main()
