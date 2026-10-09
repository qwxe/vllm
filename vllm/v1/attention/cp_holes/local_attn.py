# SPDX-License-Identifier: Apache-2.0
"""Position-explicit local attention for context-parallel holed prefills.

Every CP scheme in this package reduces its per-rank work to one primitive
(KERNEL-ALGO.md section 2.1):

    attend(Q[rows, Hq, D], qpos[rows], qreq[rows],
           keys K/V[n, Hkv, D] with kpos[n], kreq[n])
      -> O[rows, Hq, D] (fp32), LSE[rows, Hq] (fp32, natural log)
    mask(i, j) = qreq[i] == kreq[j] and kpos[j] <= qpos[i]

The positions are absolute token positions inside the request, so the same
call serves single-GPU holes, pass-Q (strided local keys, per-row limits) and
ring steps (a received buffer with its own positions). A row that sees no key
returns O = 0 and LSE = -inf, which is the identity of the LSE merge.

`LocalAttention` is the switch point to the hole-aware kernel (branch
holes/kernel). `RefLocalAttention` is the pure-PyTorch reference used by the
tests and as the fallback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import torch


@dataclass
class KeySet:
    """A gathered (contiguous) set of keys with explicit positions.

    k, v: [n, Hkv, D]; kpos: [n] int64 absolute position; kreq: [n] int64
    request index (the step-local request id, the same id space as qreq).
    """

    k: torch.Tensor
    v: torch.Tensor
    kpos: torch.Tensor
    kreq: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.kpos.numel())

    @staticmethod
    def empty(hkv: int, d: int, dtype: torch.dtype,
              device: torch.device) -> "KeySet":
        z = torch.empty(0, hkv, d, dtype=dtype, device=device)
        i = torch.empty(0, dtype=torch.int64, device=device)
        return KeySet(z, z.clone(), i, i.clone())

    @staticmethod
    def cat(sets: Sequence["KeySet"]) -> "KeySet":
        sets = [s for s in sets if s is not None]
        if len(sets) == 1:
            return sets[0]
        return KeySet(torch.cat([s.k for s in sets]),
                      torch.cat([s.v for s in sets]),
                      torch.cat([s.kpos for s in sets]),
                      torch.cat([s.kreq for s in sets]))

    def select(self, idx: torch.Tensor) -> "KeySet":
        return KeySet(self.k[idx], self.v[idx], self.kpos[idx], self.kreq[idx])


class LocalAttention(Protocol):
    """The kernel interface every scheme calls.

    Implementations must accept arbitrary (non-contiguous, unsorted) key
    positions and arbitrary query positions, and return fp32 O and LSE.
    """

    def __call__(self, q: torch.Tensor, qpos: torch.Tensor,
                 qreq: torch.Tensor, keys: KeySet,
                 scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        ...


def merge_attn_states(o_list: Sequence[torch.Tensor],
                      lse_list: Sequence[torch.Tensor]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """LSE merge of partial attention states.

    o_p: [rows, H, D] (any float dtype), lse_p: [rows, H] fp32.
    LSE = logsumexp_p LSE_p; O = sum_p exp(LSE_p - LSE) O_p. Partials with
    LSE = -inf contribute nothing; a row with every partial at -inf stays at
    O = 0, LSE = -inf. O comes back in fp32 (torch path) or in the partials'
    dtype (fused path on CUDA); callers cast.
    """
    assert len(o_list) == len(lse_list) and len(o_list) > 0
    if len(o_list) == 1:
        return o_list[0].float(), lse_list[0].float()
    fused = _fused_combine()
    dh = o_list[0].shape[-1]
    if fused is not None and o_list[0].is_cuda and dh & (dh - 1) == 0 \
            and o_list[0].shape[0] > 0:
        dts = {o.dtype for o in o_list}
        o = torch.stack(o_list if len(dts) == 1 else
                        [x.float() for x in o_list])
        lse = torch.stack([x.float() for x in lse_list])
        return fused(o, lse, return_lse=True)
    return merge_attn_states_torch(o_list, lse_list)


_FUSED = None


def _fused_combine():
    """vLLM's DCP Triton LSE combine (`dcp_lse_combine_triton`), if present.

    It takes [N, B, H, D] partials and [N, B, H] LSEs, handles empty partials
    (LSE = -inf), and returns O in the partials' dtype. Disabled with
    CP_HOLES_TORCH_MERGE=1.
    """
    global _FUSED
    if _FUSED is None:
        _FUSED = False
        import os
        if os.environ.get("CP_HOLES_TORCH_MERGE", "0") != "1":
            try:
                from vllm.v1.attention.ops.dcp_alltoall import (
                    dcp_lse_combine_triton)
                _FUSED = dcp_lse_combine_triton
            except Exception:
                _FUSED = False
    return _FUSED or None


def merge_attn_states_torch(o_list: Sequence[torch.Tensor],
                            lse_list: Sequence[torch.Tensor]
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch LSE merge (the reference for the fused path)."""
    lse = torch.stack([x.float() for x in lse_list])  # [P, rows, H]
    lse_max = lse.max(dim=0).values
    # Rows where every partial is empty: keep -inf and avoid nan.
    safe_max = torch.where(torch.isinf(lse_max),
                           torch.zeros_like(lse_max), lse_max)
    w = torch.exp(lse - safe_max)  # exp(-inf) = 0
    wsum = w.sum(dim=0)
    out_lse = torch.where(wsum > 0, safe_max + torch.log(wsum),
                          torch.full_like(wsum, -math.inf))
    inv = torch.where(wsum > 0, 1.0 / wsum, torch.zeros_like(wsum))
    o = torch.zeros_like(o_list[0], dtype=torch.float32)
    for p, op in enumerate(o_list):
        o += op.float() * (w[p] * inv).unsqueeze(-1)
    return o, out_lse


class RefLocalAttention:
    """Pure-PyTorch reference of the position-explicit primitive.

    Computes per request, in fp32, over key chunks of `key_chunk` keys merged
    by LSE (so memory stays bounded and the merge is exercised). GQA: query
    head h reads KV head h // (Hq // Hkv).
    """

    def __init__(self, key_chunk: int = 4096):
        self.key_chunk = key_chunk

    def __call__(self, q: torch.Tensor, qpos: torch.Tensor,
                 qreq: torch.Tensor, keys: KeySet,
                 scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        rows, hq, dh = q.shape
        o = torch.zeros(rows, hq, dh, dtype=torch.float32, device=q.device)
        lse = torch.full((rows, hq), -math.inf, dtype=torch.float32,
                         device=q.device)
        if rows == 0 or keys.n == 0:
            return o, lse
        hkv = keys.k.shape[1]
        assert hq % hkv == 0, (hq, hkv)
        g = hq // hkv
        for r in torch.unique(qreq).tolist():
            qi = (qreq == r).nonzero(as_tuple=True)[0]
            ki = (keys.kreq == r).nonzero(as_tuple=True)[0]
            if ki.numel() == 0:
                continue
            qq = q[qi].float()  # [nq, Hq, D]
            qp = qpos[qi]
            parts_o, parts_l = [], []
            for c0 in range(0, ki.numel(), self.key_chunk):
                kc = ki[c0:c0 + self.key_chunk]
                kk = keys.k[kc].float().repeat_interleave(g, dim=1)
                vv = keys.v[kc].float().repeat_interleave(g, dim=1)
                kp = keys.kpos[kc]
                s = torch.einsum("qhd,khd->hqk", qq, kk) * scale
                mask = kp.unsqueeze(0) <= qp.unsqueeze(1)  # [nq, nk]
                s = s.masked_fill(~mask.unsqueeze(0), -math.inf)
                m = s.max(dim=-1).values  # [H, nq]
                m_safe = torch.where(torch.isinf(m), torch.zeros_like(m), m)
                p = torch.exp(s - m_safe.unsqueeze(-1))
                l = p.sum(dim=-1)
                oc = torch.einsum("hqk,khd->qhd", p, vv)
                oc = oc / torch.where(l > 0, l, torch.ones_like(l)).t(
                ).unsqueeze(-1)
                lc = torch.where(l > 0, m_safe + torch.log(l),
                                 torch.full_like(l, -math.inf)).t()
                parts_o.append(oc)
                parts_l.append(lc)
            oo, ll = merge_attn_states(parts_o, parts_l)
            o[qi] = oo
            lse[qi] = ll
        return o, lse


_default_local_attn: Optional[LocalAttention] = None


def get_local_attention() -> LocalAttention:
    """The process-wide local attention backend (reference by default)."""
    global _default_local_attn
    if _default_local_attn is None:
        _default_local_attn = RefLocalAttention()
    return _default_local_attn


def set_local_attention(fn: LocalAttention) -> None:
    global _default_local_attn
    _default_local_attn = fn


class TiledLocalAttention:
    """A GPU-usable PyTorch implementation of the primitive.

    Per request, keys are sorted by position, so the keys a query may see
    (kpos <= qpos) are a prefix of the sorted list. Queries are processed in
    tiles of `q_tile` rows (sorted by position); a tile reads the sorted keys
    up to its largest limit in chunks of `k_chunk`. A key chunk entirely at or
    before the tile's smallest query position is a full tile (no mask); only
    the chunks across the boundary are masked (the host-side tile
    classification of KERNEL-ALGO 2.1). Scores use the input dtype for the
    matmuls and fp32 for the softmax. It stands in for the hole-aware kernel
    (branch holes/kernel) in benchmarks; it is not a fused kernel.
    """

    def __init__(self, q_tile: int = 256, k_chunk: int = 4096):
        self.q_tile, self.k_chunk = q_tile, k_chunk

    def __call__(self, q, qpos, qreq, keys: KeySet, scale):
        rows, hq, dh = q.shape
        o = torch.zeros(rows, hq, dh, dtype=torch.float32, device=q.device)
        lse = torch.full((rows, hq), -math.inf, dtype=torch.float32,
                         device=q.device)
        if rows == 0 or keys.n == 0:
            return o, lse
        hkv = keys.k.shape[1]
        g = hq // hkv
        cdt = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else \
            torch.float32
        for r in torch.unique(qreq).tolist():
            qi = (qreq == r).nonzero(as_tuple=True)[0]
            ki = (keys.kreq == r).nonzero(as_tuple=True)[0]
            if ki.numel() == 0:
                continue
            kp_all = keys.kpos[ki]
            kord = torch.argsort(kp_all)
            ki, kp_all = ki[kord], kp_all[kord]
            qord = torch.argsort(qpos[qi])
            qi = qi[qord]
            lim = torch.searchsorted(kp_all, qpos[qi], right=True)
            lim_cpu = lim.cpu()
            qp_cpu = qpos[qi].cpu()
            for t0 in range(0, qi.numel(), self.q_tile):
                tq = qi[t0:t0 + self.q_tile]
                hi = int(lim_cpu[t0:t0 + self.q_tile].max())
                if hi == 0:
                    continue
                qmin = int(qp_cpu[t0])
                qq = q[tq].to(cdt).transpose(0, 1)  # [Hq, nq, D]
                qp = qpos[tq]
                po, pl = [], []
                for c0 in range(0, hi, self.k_chunk):
                    kc = ki[c0:min(hi, c0 + self.k_chunk)]
                    kk = keys.k[kc].to(cdt).transpose(0, 1)  # [Hkv, nk, D]
                    vv = keys.v[kc].to(cdt).transpose(0, 1)
                    if g > 1:
                        kk = kk.repeat_interleave(g, dim=0)
                        vv = vv.repeat_interleave(g, dim=0)
                    s = torch.matmul(qq, kk.transpose(1, 2)).float() * scale
                    kp = keys.kpos[kc]
                    if int(kp.max()) > qmin:  # partial tile: mask
                        mask = kp.unsqueeze(0) <= qp.unsqueeze(1)
                        s = s.masked_fill(~mask.unsqueeze(0), -math.inf)
                    m = s.amax(dim=-1)
                    m_safe = torch.where(torch.isinf(m), torch.zeros_like(m),
                                         m)
                    p = torch.exp(s - m_safe.unsqueeze(-1))
                    l = p.sum(dim=-1)
                    oc = torch.matmul(p.to(cdt), vv).float()
                    oc = oc / torch.where(l > 0, l,
                                          torch.ones_like(l)).unsqueeze(-1)
                    po.append(oc.transpose(0, 1))
                    pl.append(torch.where(l > 0, m_safe + torch.log(l),
                                          torch.full_like(l, -math.inf)).t())
                oo, ll = merge_attn_states(po, pl)
                o[tq] = oo
                lse[tq] = ll
        return o, lse
