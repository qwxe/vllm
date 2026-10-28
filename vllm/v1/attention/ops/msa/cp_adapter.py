# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Synced from the standalone MSA kernel repo (its msa/ package).
"""Adapters from MSA to the CP layer's local-attention interface (holes/cp, CP-IMPL.md):

    attend(q[rows, Hq, D], qpos[rows], qreq[rows], keys) -> (O fp32 [rows, Hq, D], LSE fp32 [rows, Hq])
    mask(i, j) = qreq[i] == kreq[j] and kpos[j] <= qpos[i]; a row with no key -> O = 0, LSE = -inf.

* `MsaLocalAttention` implements the `LocalAttention` protocol over a gathered `KeySet`
  (k, v [n, Hkv, D], kpos, kreq; any order). Keys are sorted per request into a scratch paged
  buffer and the kernel reads per-key positions (kpos mode 2). Use it for ring buffers.
* `msa_paged_local` reads a vLLM paged cache in place (no gather): block table + per-block start
  positions (kpos mode 1), e.g. `PagedKVStore` under interleaved placement (u = block_size).
  Every local slot below a row's position must hold valid KV (true under front-first hole order:
  uncomputed hole slots always lie above every query of the step).
"""

import numpy as np
import torch

from .api import msa_attention
from .planner import plan_msa_rows

_PAD_POS = 1 << 30


def _dense_ids(qr, kr):
    reqs = np.unique(np.concatenate([qr, kr]))
    return np.searchsorted(reqs, qr), np.searchsorted(reqs, kr), len(reqs)


class MsaLocalAttention:
    def __init__(self, mode: str = "auto", block_size: int = 16, **plan_kw):
        self.mode, self.bs, self.plan_kw = mode, block_size, plan_kw

    def __call__(self, q, qpos, qreq, keys, scale):
        rows, hq, d = q.shape
        dev = q.device
        o = torch.zeros(rows, hq, d, dtype=torch.float32, device=dev)
        lse = torch.full((rows, hq), float("-inf"), dtype=torch.float32, device=dev)
        if rows == 0 or keys.n == 0:
            return o, lse
        hkv = keys.k.shape[1]
        bs = self.bs
        qp = qpos.cpu().numpy().astype(np.int64)
        qr = qreq.cpu().numpy().astype(np.int64)
        kp = keys.kpos.cpu().numpy().astype(np.int64)
        kr = keys.kreq.cpu().numpy().astype(np.int64)
        qd, kd, R = _dense_ids(qr, kr)
        korder = np.lexsort((kp, kd))
        kd_s, kp_s = kd[korder], kp[korder]
        counts = np.bincount(kd_s, minlength=R)
        nblk = np.maximum(1, -(-counts // bs))
        blk_base = np.concatenate([[0], np.cumsum(nblk)[:-1]])
        kstart = np.concatenate([[0], np.cumsum(counts)[:-1]])
        local = np.arange(len(kd_s)) - kstart[kd_s]
        dest = blk_base[kd_s] * bs + local
        total = int(nblk.sum())
        maxb = int(nblk.max())
        kbuf = torch.zeros(total * bs, hkv, d, dtype=keys.k.dtype, device=dev)
        vbuf = torch.zeros_like(kbuf)
        dest_t = torch.from_numpy(dest).to(dev)
        src_t = torch.from_numpy(korder).to(dev)
        kbuf[dest_t] = keys.k[src_t]
        vbuf[dest_t] = keys.v[src_t].to(kbuf.dtype)
        bt = blk_base[:, None] + np.arange(maxb)[None, :]
        bt = np.minimum(bt, total - 1)
        kpos_tab = np.full((R, maxb * bs), _PAD_POS, dtype=np.int64)
        kpos_tab[kd_s, local] = kp_s
        key_pos = [kp_s[kstart[r]:kstart[r] + counts[r]] for r in range(R)]
        plan = plan_msa_rows(qp, qd, counts, hkv, hq // hkv, block_size=bs, key_pos=key_pos,
                             mode=self.mode, device=dev, elem_size=q.element_size(),
                             qpos_dev=torch.from_numpy(qp.astype(np.int32)).to(dev), **self.plan_kw)
        msa_attention(q, kbuf.view(total, bs, hkv, d), vbuf.view(total, bs, hkv, d),
                      torch.from_numpy(bt.astype(np.int32)).to(dev), plan,
                      kblk_pos=torch.from_numpy(kpos_tab.astype(np.int32)).to(dev),
                      softmax_scale=scale, out=o, lse=lse)
        return o, lse


def msa_paged_local(q, qpos, qreq, k_cache, v_cache, block_table, kblk_pos, kv_lens, scale,
                    mode: str = "auto", out=None, lse=None, kblk_pos_cpu=None, **plan_kw):
    """Paged, in-place local attention. qreq indexes block_table rows; kblk_pos [R, max_blocks]
    int32 holds each local block's start position (None = contiguous, position = local index);
    kv_lens[R] = local keys per request. Returns (O, LSE); O in `out`'s dtype (fp32 default)."""
    rows, hq, d = q.shape
    hkv = k_cache.shape[2]
    if out is None:
        out = torch.zeros(rows, hq, d, dtype=torch.float32, device=q.device)
    if lse is None:
        lse = torch.full((rows, hq), float("-inf"), dtype=torch.float32, device=q.device)
    if kblk_pos is not None and kblk_pos_cpu is None:
        kblk_pos_cpu = kblk_pos.cpu().numpy()
    plan = plan_msa_rows(qpos.cpu().numpy(), qreq.cpu().numpy(), np.asarray(kv_lens), hkv, hq // hkv,
                         block_size=k_cache.shape[1], kblk_pos=kblk_pos_cpu, mode=mode,
                         device=q.device, qpos_dev=qpos.to(torch.int32), elem_size=q.element_size(),
                         **plan_kw)
    msa_attention(q, k_cache, v_cache, block_table, plan, kblk_pos=kblk_pos, softmax_scale=scale,
                  out=out, lse=lse)
    return out, lse
