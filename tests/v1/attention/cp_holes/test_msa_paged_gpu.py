# SPDX-License-Identifier: Apache-2.0
"""GPU: CP_HOLES local attention through MSA.

1. `AutoLocalAttention` (MSA on CUDA) vs `RefLocalAttention` on gathered
   holed key sets (random positions, several requests, empty rows).
2. `PagedKVStore.attend` in place (MSA paged mode, per-block start
   positions) vs gather + reference, on vLLM's PCP layout for a rank r of d
   (strided blocks: pass-Q / ring) and on the plain layout (Ulysses /
   all-gather), with the step's keys written first; the cached plan is reused
   for a second call (second "layer").

    gpurun env PYTHONPATH=<wt> <venv>/bin/python test_msa_paged_gpu.py
"""

import math

import torch

from vllm.v1.attention.cp_holes import RefLocalAttention
from vllm.v1.attention.cp_holes.kv_store import (PagedKVStore,
                                                 vllm_local_layout)
from vllm.v1.attention.cp_holes.local_attn import AutoLocalAttention, KeySet
from vllm.v1.attention.cp_holes.plan import StepPlan, home_rank


def _err(a, b):
    fin = ~torch.isinf(b[1])
    eo = (a[0].float() - b[0].float()).abs().max().item()
    el = (a[1][fin] - b[1][fin]).abs().max().item() if fin.any() else 0.0
    assert torch.equal(torch.isinf(a[1]), torch.isinf(b[1]))
    return max(eo, el)


def test_auto_vs_ref(dt, tol):
    g = torch.Generator(device="cuda").manual_seed(0)
    ref, auto = RefLocalAttention(), AutoLocalAttention()
    worst = 0.0
    for trial in range(4):
        n, rows, hq, hkv, d = 3000, 400, 8, 2, 64
        q = torch.randn(rows, hq, d, generator=g, device="cuda").to(dt)
        ks = KeySet(torch.randn(n, hkv, d, generator=g, device="cuda").to(dt),
                    torch.randn(n, hkv, d, generator=g, device="cuda").to(dt),
                    torch.randperm(5000, device="cuda")[:n],
                    torch.randint(0, 3, (n, ), device="cuda"))
        qpos = torch.randint(-1, 5200, (rows, ), device="cuda")
        qreq = torch.randint(0, 4, (rows, ), device="cuda")  # req 3: no keys
        worst = max(worst, _err(auto(q, qpos, qreq, ks, 0.125),
                                ref(q, qpos, qreq, ks, 0.125)))
    assert worst < tol, worst
    return worst


def test_paged_in_place(dt, tol):
    torch.manual_seed(1)
    dev = torch.device("cuda")
    hq, hkv, d, bs = 8, 2, 64, 16
    segs = [[(300, 40), (900, 200)], [(0, 500)], [(1200, 7), (2000, 300)]]
    worst = 0.0
    for world, rank in ((4, 1), (2, 0), (1, 0)):
        plan = StepPlan([list(s) for s in segs], d=world, u=bs)
        nblk = [-(-plan.e_step[i] // (bs * world)) for i in range(3)]
        perm = torch.randperm(sum(nblk) + 5)
        cache = torch.zeros(2, len(perm), bs, hkv, d, dtype=dt, device=dev)
        tables, lps, present, off = [], [], [], 0
        K, V = [], []
        for i in range(3):
            e = plan.e_step[i]
            row = perm[off:off + nblk[i]]
            off += nblk[i]
            bt, lp = vllm_local_layout(row, nblk[i], bs, rank, world, bs)
            lp = torch.where(lp < e, lp, torch.full_like(lp, -1))
            tables.append(bt.to(dev))
            lps.append(lp.to(dev))
            present.append(torch.zeros_like(lp, dtype=torch.bool).to(dev))
            K.append(torch.randn(e, hkv, d, device=dev).to(dt))
            V.append(torch.randn(e, hkv, d, device=dev).to(dt))
        stores = [PagedKVStore(cache, tables, lps, present, in_place=ip,
                               step_cache={}) for ip in (True, False)]
        # write every local position (resident + this step's keys)
        for st in stores[:1]:
            for i in range(3):
                pos = torch.arange(plan.e_step[i], device=dev)
                pos = pos[home_rank(pos, world, bs) == rank]
                st.write(KeySet(K[i][pos], V[i][pos], pos,
                                torch.full_like(pos, i)))
        stores[1].present = [p.clone() for p in stores[0].present]
        qpos, qreq = plan.qpos.to(dev), plan.qreq.to(dev)
        q = torch.randn(plan.m, hq, d, device=dev).to(dt)
        reqs, lim = [0, 1, 2], plan.e_step
        ref = stores[1].attend(q, qpos, qreq, reqs, lim, 0.125,
                               RefLocalAttention())
        for _ in range(2):  # second call reuses the step's MSA plan
            got = stores[0].attend(q, qpos, qreq, reqs, lim, 0.125,
                                   RefLocalAttention(), plan.qpos, plan.qreq,
                                   key="k")
            worst = max(worst, _err(got, ref))
        assert ("msa_plan", "k") in stores[0].step_cache
    assert worst < tol, worst
    return worst


if __name__ == "__main__":
    # fp32 inputs: MSA uses TF32 dots (KERNEL-IMPL.md), hence 5e-3
    for dt, tol in ((torch.float32, 5e-3), (torch.bfloat16, 2e-2)):
        a = test_auto_vs_ref(dt, tol)
        b = test_paged_in_place(dt, tol)
        print(f"{dt}: auto(MSA) vs ref {a:.2e}; paged in-place vs gather "
              f"{b:.2e}")
    print("msa paged tests OK")
