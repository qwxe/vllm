# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness of the MLA variant of MSA (compute and absorbed paths) against a
dense fp32 reference with an explicit position mask.

Builds random holed steps (several requests, 1-16 segments each, pieces of 1 to
thousands of tokens) over a paged MLA latent cache [nb, bs, 576], with random
kv_b_proj weights. Checks, for every planner mode:
* compute path: gather [0, E) -> up-project -> MSA (MHA, qk 128+64, v 128);
* compute path in two context passes (kblk_pos mode) merged by LSE;
* absorbed path: MSA (MQA, qk 512+64, v = latent) over the paged cache, W_UV out.

Run: python tests/v1/attention/msa_mla/test_msa_mla_kernel.py [--quick]
"""

import argparse
import math
import random
import sys

import numpy as np
import torch

from vllm.v1.attention.ops.msa import merge_states
from vllm.v1.attention.ops.msa.api_mla import (ABSORB, COMPUTE, context_index,
                                               mla_reference, msa_mla_attention,
                                               plan_msa_mla)

H, P, R, V, L = 16, 128, 64, 128, 512
BS = 16


def make_case(rng: random.Random, kind: str, n_req: int, max_e: int):
    reqs = []
    for _ in range(n_req):
        E = rng.randint(64, max_e)
        if kind == "fresh":
            segs = [(0, E)]
        elif kind == "tail":
            c = rng.randint(1, E - 1)
            segs = [(c, E - c)]
        elif kind == "tiny":
            segs, p = [], rng.randint(5, max(6, E // 4))
            while p < E - 40 and len(segs) < 15:
                n = rng.randint(1, 12)
                segs.append((p, n))
                p += n + rng.randint(1, max(2, E // 8))
            if rng.random() < 0.5:
                segs.append((E - 1, 1))
        else:  # holes + tail
            segs, p = [], rng.randint(0, E // 3)
            for _ in range(rng.randint(1, 6)):
                if p >= E - 32:
                    break
                n = rng.randint(1, max(1, (E - p) // 4))
                segs.append((p, n))
                p += n + rng.randint(1, max(2, (E - p) // 3))
            if p < E:
                segs.append((p, E - p))
        reqs.append((E, segs))
    return reqs


def run_case(reqs, dtype, mode, device, gen, check_absorb=True, two_pass=False):
    G = len(reqs)
    Es = [E for E, _ in reqs]
    nblk = [-(-E // BS) for E in Es]
    total_blocks = sum(nblk) + 7
    perm = torch.randperm(total_blocks, generator=gen)
    cache = (torch.randn(total_blocks, BS, L + R, generator=gen) * 0.5).to(dtype).to(device)
    bt = torch.zeros(G, max(nblk) + 3, dtype=torch.int32)
    o = 0
    for g in range(G):
        bt[g, : nblk[g]] = perm[o : o + nblk[g]]
        o += nblk[g]
    bt = bt.to(device)
    w_kv = (torch.randn(L, H, P + V, generator=gen) / math.sqrt(L)).to(dtype).to(device)
    W_UK, W_UV = w_kv.split([P, V], dim=-1)  # [L, H, P], [L, H, V]

    segs, qpos, qreq = [], [], []
    row = 0
    for g, (E, ss) in enumerate(reqs):
        for s, n in ss:
            segs.append((g, row, n, s))
            qpos.extend(range(s, s + n))
            qreq.extend([g] * n)
            row += n
    rows = row
    q = (torch.randn(rows, H, P + R, generator=gen)).to(dtype).to(device)
    q_nope, q_pe = q[..., :P], q[..., P:]
    scale = 1.0 / math.sqrt(P + R)

    # reference keys: up-project every request's [0, E) in fp32
    kpos, kreq, lat = [], [], []
    flat = cache.view(-1, L + R)
    for g, E in enumerate(Es):
        pos = torch.arange(E, device=device)
        slots = bt[g, pos // BS].long() * BS + pos % BS
        lat.append(flat[slots])
        kpos.append(pos)
        kreq.append(torch.full_like(pos, g))
    lat = torch.cat(lat).float()
    k_nope_r = torch.einsum("nl,lhp->nhp", lat[:, :L], W_UK.float())
    v_r = torch.einsum("nl,lhv->nhv", lat[:, :L], W_UV.float())
    qpos_t = torch.tensor(qpos, device=device)
    qreq_t = torch.tensor(qreq, device=device)
    o_ref, lse_ref = mla_reference(q_nope, q_pe, k_nope_r, lat[:, L:], v_r, qpos_t, qreq_t,
                                   torch.cat(kpos), torch.cat(kreq), scale)
    errs = {}

    def check(tag, o, lse=None):
        err = (o.float() - o_ref).abs().max().item()
        ok = err < (2.5e-2 if dtype != torch.float32 else 2e-3)
        e2 = 0.0
        if lse is not None:
            fin = torch.isfinite(lse_ref)
            e2 = (lse[fin] - lse_ref[fin]).abs().max().item() if fin.any() else 0.0
            ok = ok and e2 < 5e-3 and bool(torch.isinf(lse[~fin]).all())
        errs[tag] = (ok, err, e2)

    # ---- compute path, one pass
    src, dbt, kv_lens, kbp = context_index(bt, [0] * G, Es, BS)
    ws = flat.index_select(0, src)
    kvn = (ws[:, :L] @ w_kv.view(L, -1)).view(-1, H, P + V)
    nb = ws.shape[0] // BS
    k_nope = kvn.view(nb, BS, H, P + V)[..., :P]
    v_d = kvn.view(nb, BS, H, P + V)[..., P:]
    k_pe = ws.view(nb, BS, 1, L + R)[..., L:].expand(-1, -1, H, -1)
    plan = plan_msa_mla(segs, kv_lens, COMPUTE, H, block_size=BS, mode=mode, device=device,
                        num_rows=rows)
    out, lse = msa_mla_attention(q_nope, q_pe, k_nope, k_pe, v_d, dbt, plan,
                                 softmax_scale=scale, return_lse=True)
    check("compute", out, lse)

    # ---- compute path, two passes (split every request's context at E/2, 16-aligned)
    if two_pass:
        mids = [max(BS, (E // 2) // BS * BS) for E in Es]
        mids = [min(m, E) for m, E in zip(mids, Es)]
        parts = []
        for starts, lens in (([0] * G, mids), (mids, [E - m for E, m in zip(Es, mids)])):
            src, dbt, kv_lens, kbp = context_index(bt, starts, lens, BS)
            ws = flat.index_select(0, src)
            kvn = (ws[:, :L] @ w_kv.view(L, -1)).view(-1, H, P + V)
            nb = ws.shape[0] // BS
            k_nope = kvn.view(nb, BS, H, P + V)[..., :P]
            v_d = kvn.view(nb, BS, H, P + V)[..., P:]
            k_pe = ws.view(nb, BS, 1, L + R)[..., L:].expand(-1, -1, H, -1)
            plan = plan_msa_mla(segs, kv_lens, COMPUTE, H, block_size=BS, mode=mode,
                                device=device, num_rows=rows,
                                kblk_pos=None if kbp is None else kbp)
            kbp_t = None if kbp is None else torch.from_numpy(kbp.astype(np.int32)).to(device)
            o_p = torch.zeros(rows, H, V, device=device, dtype=torch.float32)
            l_p = torch.full((rows, H), float("-inf"), device=device)
            msa_mla_attention(q_nope, q_pe, k_nope, k_pe, v_d, dbt, plan, kblk_pos=kbp_t,
                              softmax_scale=scale, out=o_p, lse=l_p)
            parts.append((o_p, l_p))
        o2, l2 = merge_states(parts[0][0], parts[0][1], parts[1][0], parts[1][1])
        check("compute_2pass", o2, l2)

    # ---- absorbed path over the paged latent
    if check_absorb:
        q_lat = torch.einsum("rhp,lhp->rhl", q_nope.float(), W_UK.float()).to(dtype)
        kv_lens_a = np.asarray(Es, dtype=np.int64)
        plan = plan_msa_mla(segs, kv_lens_a, ABSORB, H, block_size=BS, mode=mode,
                            device=device, num_rows=rows)
        ka = cache.view(-1, BS, 1, L + R)[..., :L]
        kb = cache.view(-1, BS, 1, L + R)[..., L:]
        o_lat, lse = msa_mla_attention(q_lat, q_pe, ka, kb, None, bt, plan,
                                       softmax_scale=scale, return_lse=True)
        o_abs = torch.einsum("rhl,lhv->rhv", o_lat.float(), W_UV.float())
        check("absorb", o_abs, lse)
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda")
    rng = random.Random(0)
    gen = torch.Generator().manual_seed(0)
    n_ok = n_all = 0
    worst = {}
    kinds = ["holes", "tail", "fresh", "tiny"]
    modes = ["a", "b", "c", "bc", "auto"]
    reps = 1 if args.quick else 3
    for dtype in (torch.bfloat16, torch.float16):
        for kind in kinds:
            for mode in modes:
                for rep in range(reps):
                    n_req = rng.choice([1, 1, 2, 3])
                    max_e = rng.choice([300, 2000, 6000])
                    reqs = make_case(rng, kind, n_req, max_e)
                    errs = run_case(reqs, dtype, mode, device, gen,
                                    two_pass=(rep == 0))
                    for tag, (ok, e1, e2) in errs.items():
                        n_all += 1
                        n_ok += ok
                        w = worst.setdefault(tag, [0.0, 0.0])
                        w[0], w[1] = max(w[0], e1), max(w[1], e2)
                        if not ok:
                            print(f"FAIL {dtype} {kind} {mode} {tag} errO={e1:.3e} "
                                  f"errLSE={e2:.3e} reqs={[(E, s[:4]) for E, s in reqs]}",
                                  flush=True)
    # one deep case: 16 tiny pieces + tail at 32K (split-KV engaged)
    reqs = [(32768, [(20000 + 900 * i, 8) for i in range(16)] + [(32768 - 512, 512)])]
    for mode in modes:
        errs = run_case(reqs, torch.bfloat16, mode, device, gen, two_pass=True)
        for tag, (ok, e1, e2) in errs.items():
            n_all += 1
            n_ok += ok
            if not ok:
                print(f"FAIL deep {mode} {tag} errO={e1:.3e} errLSE={e2:.3e}", flush=True)
    for tag, (e1, e2) in worst.items():
        print(f"worst {tag}: |dO|={e1:.3e} |dLSE|={e2:.3e}")
    print(f"RESULT {n_ok}/{n_all} {'PASS' if n_ok == n_all else 'FAIL'}")
    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    sys.exit(main())
