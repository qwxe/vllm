# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Synced from the standalone repo /home/asd/vllm-dev-kernels (msa/).
"""Pure-PyTorch reference for position-explicit paged attention (fp32 math)."""

import math

import numpy as np
import torch


def gather_keys(k_cache, v_cache, block_table, req, kv_len, kblk_pos=None):
    bs = k_cache.shape[1]
    nblk = -(-kv_len // bs)
    blks = block_table[req, :nblk].long()
    K = k_cache[blks].reshape(-1, *k_cache.shape[2:])[:kv_len]
    V = v_cache[blks].reshape(-1, *v_cache.shape[2:])[:kv_len]
    if kblk_pos is None:
        kpos = torch.arange(kv_len, device=k_cache.device)
    else:
        starts = kblk_pos[req, :nblk].long()
        kpos = (starts[:, None] + torch.arange(bs, device=k_cache.device)[None, :]).reshape(-1)[:kv_len]
    return K, V, kpos


def reference_attention(q, k_cache, v_cache, block_table, segments, kv_lens, kblk_pos=None,
                        softmax_scale=None, chunk=256):
    """Returns (O fp32 [rows,Hq,D], LSE fp32 [rows,Hq]) for the rows covered by segments;
    other rows are NaN."""
    rows, Hq, D = q.shape
    Hkv = k_cache.shape[2]
    G = Hq // Hkv
    scale = softmax_scale or 1.0 / math.sqrt(D)
    O = torch.full((rows, Hq, D), float("nan"), device=q.device)
    LSE = torch.full((rows, Hq), float("nan"), device=q.device)
    seg = np.asarray(segments).reshape(-1, 4)
    cache = {}
    for req, r0, n, p0 in seg:
        req = int(req)
        if req not in cache:
            cache[req] = gather_keys(k_cache, v_cache, block_table, req, int(kv_lens[req]), kblk_pos)
        K, V, kpos = cache[req]
        Kf = K.float().repeat_interleave(G, dim=1)  # [kv, Hq, D]
        Vf = V.float().repeat_interleave(G, dim=1)
        for c0 in range(0, int(n), chunk):
            c1 = min(int(n), c0 + chunk)
            qq = q[r0 + c0:r0 + c1].float()                          # [c, Hq, D]
            qp = torch.arange(p0 + c0, p0 + c1, device=q.device)
            s = torch.einsum("chd,khd->hck", qq, Kf) * scale         # [Hq, c, kv]
            mask = kpos[None, :] <= qp[:, None]                       # [c, kv]
            s = s.masked_fill(~mask[None], float("-inf"))
            lse = torch.logsumexp(s, dim=-1)                          # [Hq, c]
            p = torch.exp(s - torch.where(torch.isinf(lse), 0, lse)[..., None])
            o = torch.einsum("hck,khd->chd", p, Vf)
            O[r0 + c0:r0 + c1] = o
            LSE[r0 + c0:r0 + c1] = lse.transpose(0, 1)
    return O, LSE
