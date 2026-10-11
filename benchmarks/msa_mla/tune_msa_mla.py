# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch-config sweep for the MLA MSA kernel (compute path) and the time split
gather / up-projection / attention. One layer, DeepSeek-V2-Lite dims, bf16.

    python benchmarks/msa_mla/tune_msa_mla.py
"""

import itertools
import math

import numpy as np
import torch
import triton

from vllm.v1.attention.ops.msa.api_mla import COMPUTE, msa_mla_attention
from vllm.v1.attention.ops.msa.planner import plan_msa

H, P, R, V, L, BS = 16, 128, 64, 128, 512, 16


def t_ms(fn, flush, n=7):
    ts = []
    for _ in range(n):
        flush.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def main():
    dev = torch.device("cuda")
    dt = torch.bfloat16
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)
    w = (torch.randn(L, H * (P + V), device=dev) / math.sqrt(L)).to(dt)
    for E, segs_desc in ((32768, [(30720, 2048)]), (16384, [(0, 8192)]),
                         (32768, [(16384 + 900 * i, 16) for i in range(16)] + [(30720, 2048)])):
        nb = E // BS
        cache = torch.randn(nb, BS, L + R, device=dev).to(dt)
        src = torch.arange(E, device=dev)
        bt = torch.arange(nb, device=dev, dtype=torch.int32)[None]
        segs, r = [], 0
        for s, n in segs_desc:
            segs.append((0, r, n, s))
            r += n
        m = r
        q = torch.randn(m, H, P + R, device=dev).to(dt)
        ws = cache.view(-1, L + R).index_select(0, src)
        kvn = (ws[:, :L] @ w).view(nb, BS, H, P + V)
        kpe = ws.view(nb, BS, 1, L + R)[..., L:].expand(-1, -1, H, -1)
        t_g = t_ms(lambda: cache.view(-1, L + R).index_select(0, src), flush)
        t_u = t_ms(lambda: ws[:, :L] @ w, flush)
        print(f"E={E} segs={segs_desc[:2]}.. m={m}: gather {t_g:.2f} ms, up-proj {t_u:.2f} ms")
        res = []
        for rpt, bn, nw, ns in itertools.product((32, 64, 128), (32, 64, 128), (4, 8), (1, 2, 3)):
            smem = (P + R + V) * bn * 2 * ns
            if smem > 99 * 1024:
                continue
            try:
                plan = plan_msa(segs, np.array([E]), H, 1, block_size=BS, mode="auto",
                                device=dev, rows_per_tile=rpt, bn=bn, num_warps=nw,
                                num_stages=ns, num_rows=m)
                f = lambda: msa_mla_attention(q[..., :P], q[..., P:], kvn[..., :P], kpe,
                                              kvn[..., P:], bt, plan,
                                              softmax_scale=1 / math.sqrt(P + R))
                f()
                res.append((t_ms(f, flush), rpt, bn, nw, ns))
            except (triton.runtime.errors.OutOfResources, RuntimeError) as ex:
                res.append((float("inf"), rpt, bn, nw, ns))
        res.sort()
        for x in res[:6]:
            print("   attn %.2f ms  rows_per_tile=%d bn=%d warps=%d stages=%d" % x)
        from vllm.v1.attention.ops.msa.api_mla import plan_msa_mla
        plan = plan_msa_mla(segs, np.array([E]), COMPUTE, H, block_size=BS, device=dev,
                            num_rows=m)
        f = lambda: msa_mla_attention(q[..., :P], q[..., P:], kvn[..., :P], kpe,
                                      kvn[..., P:], bt, plan, softmax_scale=1 / math.sqrt(P + R))
        print(f"   current default: {t_ms(f, flush):.2f} ms {plan.stats}")
        # FA2 reference for the single-segment shapes (stock: v padded to 192)
        try:
            from vllm.vllm_flash_attn import flash_attn_varlen_func
            k = torch.cat([kvn[..., :P], kpe], -1).reshape(E, H, P + R)
            vp = torch.nn.functional.pad(kvn[..., P:].reshape(E, H, V), [0, P + R - V])
            s0, n0 = segs_desc[-1]
            qq = q[-n0:]
            cq = torch.tensor([0, n0], dtype=torch.int32, device=dev)
            ck = torch.tensor([0, s0 + n0], dtype=torch.int32, device=dev)
            g = lambda: flash_attn_varlen_func(qq, k[: s0 + n0], vp[: s0 + n0], max_seqlen_q=n0,
                                               cu_seqlens_q=cq, max_seqlen_k=s0 + n0,
                                               cu_seqlens_k=ck, causal=True, fa_version=2,
                                               softmax_scale=1 / math.sqrt(P + R))
            print(f"   FA2 (last segment alone, bottom-right causal, v padded): {t_ms(g, flush):.2f} ms")
        except Exception as ex:  # noqa: BLE001
            print("   FA2 n/a", ex)


if __name__ == "__main__":
    main()
