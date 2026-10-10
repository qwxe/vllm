# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone single-segment attention times (FA2 varlen vs MSA auto), to sanity-check
the profiler's segmented_attention rows. Llama-3.2-1B heads (32/8, head_dim 64), bf16,
block 16, L2 flushed, median of 11 round-robin rounds."""
import json
import math
import sys

import numpy as np
import torch

from vllm.v1.attention.ops.msa import msa_attention, plan_msa
from vllm.vllm_flash_attn import flash_attn_varlen_func

HQ, HKV, D, BS = 32, 8, 64, 16
fl = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")


def main():
    E = 65536 + 2048
    nb = E // BS + 1
    k = torch.randn(nb, BS, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    bt = torch.arange(nb, device="cuda", dtype=torch.int32)[None, :]
    scale = 1 / math.sqrt(D)
    for qn in (16, 256, 2048):
        for s in (0, 1024, 4096, 16384, 32768, 63488):
            q = torch.randn(qn, HQ, D, device="cuda", dtype=torch.bfloat16)
            out = torch.empty_like(q)
            cu = torch.tensor([0, qn], dtype=torch.int32, device="cuda")
            su = torch.tensor([s + qn], dtype=torch.int32, device="cuda")
            plan = plan_msa([(0, 0, qn, s)], [s + qn], HKV, HQ // HKV, block_size=BS, mode="auto", num_rows=qn)
            fns = {
                "fa2": lambda: flash_attn_varlen_func(q, k, v, qn, cu, s + qn, seqused_k=su, causal=True,
                                                      block_table=bt, softmax_scale=scale, out=out, fa_version=2),
                "msa": lambda: msa_attention(q, k, v, bt, plan, softmax_scale=scale, out=out),
            }
            for f in fns.values():
                f()
            torch.cuda.synchronize()
            ts = {n: [] for n in fns}
            for _ in range(11):
                for n, f in fns.items():
                    fl.zero_()
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record()
                    f()
                    b.record()
                    torch.cuda.synchronize()
                    ts[n].append(a.elapsed_time(b) * 1e3)
            print(json.dumps(dict(q=qn, s=s, **{n + "_us": round(float(np.median(t)), 1) for n, t in ts.items()})),
                  flush=True)


if __name__ == "__main__":
    main()
