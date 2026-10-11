# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA for MLA: plan + launch of the split-qk / separate-V kernel variant.

Two paths (MLA-MSA.md sections 1-2):

    COMPUTE: MHA over up-projected K/V.  qa = q_nope [rows, H, 128], qb = q_pe [rows, H, 64],
             ka = k_nope [nb, bs, H, 128], kb = k_pe [nb, bs, H(stride 0), 64],
             v = v [nb, bs, H, 128]; plan with num_kv_heads = H, G = 1.
    ABSORB:  MQA over the latent.        qa = q_lat [rows, H, 512], qb = q_pe [rows, H, 64],
             ka = cache[..., :512] [nb, bs, 1, 512], kb = cache[..., 512:], v = ka (V_FROM_K);
             plan with num_kv_heads = 1, G = H.

The planner is MSA's `plan_msa` / `plan_msa_rows` (same tiles, items, split-KV,
key layouts); only the launch configuration differs (the MLA dims are wider).
"""

import math

import numpy as np
import torch

from .kernel import _msa_merge
from .kernel_mla import _msa_mla_fwd
from .planner import MsaPlan, _next_pow2, plan_msa, plan_msa_rows

COMPUTE = "compute"
ABSORB = "absorb"


def mla_launch_config(path: str, m_rows: int, dk: int, dv: int, elem_size: int = 2):
    """(bn, num_warps, num_stages) for the MLA kernel variant.

    Shared memory per stage is (dk + dv) * BN * 2 bytes for the compute path and
    dk * BN * 2 for the absorbed path (V comes from the K tile); SM 8.x has about
    100 KB per CTA, Hopper 228 KB. Tuned on the RTX 4050 (MLA-MSA.md section 6).
    """
    if path == ABSORB:
        return (32, 4, 2) if elem_size <= 2 else (16, 4, 2)
    if elem_size > 2:
        return 32, (4 if m_rows <= 64 else 8), 1
    if m_rows <= 64:
        return 64, 4, 2
    return 32, 8, 2


def _bq_for(max_seg: int, g_pad: int, rows_per_tile: int) -> int:
    bq_cap = max(1, rows_per_tile // g_pad)
    return min(bq_cap, max(max(1, 32 // g_pad), _next_pow2(max_seg)))


def plan_msa_mla(
    segments,
    kv_lens,
    path: str,
    num_heads: int,
    *,
    block_size: int = 16,
    kblk_pos=None,
    mode: str = "auto",
    num_sms: int | None = None,
    device="cuda",
    num_rows: int | None = None,
    rows_per_tile: int | None = None,
    qpos: torch.Tensor | None = None,
    eps: float = 0.05,
    c: float = 2.0,
    dk: int | None = None,
    dv: int | None = None,
    elem_size: int = 2,
) -> MsaPlan:
    """Plan an MLA launch from segments (req, row_start, n_rows, pos_start)."""
    seg = np.asarray(segments, dtype=np.int64).reshape(-1, 4)
    if path == ABSORB:
        hkv, g = 1, num_heads
        rows_per_tile = rows_per_tile or 16
        dk, dv = dk or 576, dv or 512
    else:
        hkv, g = num_heads, 1
        rows_per_tile = rows_per_tile or 64
        dk, dv = dk or 192, dv or 128
    g_pad = _next_pow2(g)
    max_seg = int(seg[:, 2].max()) if seg.shape[0] else 1
    m_rows = _bq_for(max_seg, g_pad, rows_per_tile) * g_pad
    bn, nw, ns = mla_launch_config(path, m_rows, dk, dv, elem_size)
    return plan_msa(seg, kv_lens, hkv, g, block_size=block_size, kblk_pos=kblk_pos,
                    mode=mode, num_sms=num_sms, device=device, rows_per_tile=rows_per_tile,
                    bn=bn, num_warps=nw, num_stages=ns, eps=eps, c=c, qpos=qpos,
                    num_rows=num_rows)


def plan_msa_mla_rows(qpos, qreq, kv_lens, path: str, num_heads: int, *,
                      block_size: int = 16, kblk_pos=None, key_pos=None,
                      mode: str = "auto", num_sms: int | None = None, device="cuda",
                      rows_per_tile: int | None = None, qpos_dev=None,
                      eps: float = 0.05, c: float = 2.0, elem_size: int = 2) -> MsaPlan:
    """Per-row variant (CP callers): rows in any order, any positions."""
    if path == ABSORB:
        hkv, g, rpt, dk, dv = 1, num_heads, rows_per_tile or 16, 576, 512
    else:
        hkv, g, rpt, dk, dv = num_heads, 1, rows_per_tile or 64, 192, 128
    bn, nw, ns = mla_launch_config(path, rpt, dk, dv, elem_size)
    return plan_msa_rows(qpos, qreq, kv_lens, hkv, g, block_size=block_size,
                         kblk_pos=kblk_pos, key_pos=key_pos, mode=mode, num_sms=num_sms,
                         device=device, rows_per_tile=rpt, bn=bn, num_warps=nw,
                         num_stages=ns, eps=eps, c=c, qpos_dev=qpos_dev)


def msa_mla_attention(
    qa: torch.Tensor,             # [rows, Hq, DA]
    qb: torch.Tensor | None,      # [rows, Hq, DB] or None
    ka: torch.Tensor,             # [nb, bs, Hkv, DA]   (any strides, last 1)
    kb: torch.Tensor | None,      # [nb, bs, Hkv or 1, DB]; head stride may be 0
    v: torch.Tensor | None,       # [nb, bs, Hkv, DV]; None = V from K part A
    block_table: torch.Tensor,    # [num_reqs, max_blocks] int32
    plan: MsaPlan,
    kblk_pos: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,   # [rows, Hq, DV]
    lse: torch.Tensor | None = None,   # [rows, Hq] fp32 natural log
    return_lse: bool = False,
):
    """Position-explicit MLA attention (one launch + split merge). Rows the plan
    does not cover are left untouched in `out`; empty rows get O = 0, LSE = -inf."""
    rows, Hq, DA = qa.shape
    Hkv = ka.shape[2]
    G = plan.num_q_per_kv
    assert Hq == Hkv * G, (Hq, Hkv, G)
    assert ka.shape[1] == plan.block_size
    DB = 0 if qb is None else qb.shape[2]
    v_from_k = v is None
    DV = DA if v_from_k else v.shape[3]
    for d in (DA, DV) + ((DB,) if DB else ()):
        assert d & (d - 1) == 0, f"MLA MSA needs power-of-two dims, got {d}"
    assert qa.stride(2) == 1 and ka.stride(3) == 1
    if DB:
        assert qb.stride(2) == 1 and kb.stride(3) == 1
    if not v_from_k:
        assert v.stride(3) == 1
    if plan.kpos_mode:
        assert kblk_pos is not None and kblk_pos.dtype == torch.int32
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(DA + DB)
    if out is None:
        out = torch.empty(rows, Hq, DV, device=qa.device, dtype=qa.dtype)
    assert out.stride(2) == 1
    write_lse = return_lse or lse is not None
    if write_lse and lse is None:
        lse = torch.empty(rows, Hq, device=qa.device, dtype=torch.float32)
    M = plan.m_rows
    if plan.n_slots:
        op = torch.empty(plan.n_slots, M, DV, device=qa.device, dtype=torch.float32)
        lsep = torch.empty(plan.n_slots, M, device=qa.device, dtype=torch.float32)
    else:
        op = lsep = qa
    lse_arg = lse if write_lse else qa
    kp = kblk_pos if kblk_pos is not None else block_table
    qb_ = qb if DB else qa
    kb_ = kb if DB else ka
    v_ = ka if v_from_k else v
    n_items = plan.items.shape[0]
    if n_items:
        _msa_mla_fwd[(n_items,)](
            qa, qb_, ka, kb_, v_, out, lse_arg, op, lsep,
            plan.qpos, plan.tile_rows, plan.items, block_table, kp,
            softmax_scale * 1.4426950408889634,
            qa.stride(0), qa.stride(1), qb_.stride(0), qb_.stride(1),
            out.stride(0), out.stride(1),
            lse.stride(0) if write_lse else 0, lse.stride(1) if write_lse else 0,
            ka.stride(0), ka.stride(1), ka.stride(2),
            kb_.stride(0), kb_.stride(1), kb_.stride(2),
            v_.stride(0), v_.stride(1), v_.stride(2),
            block_table.stride(0), kp.stride(0),
            G=G, G_PAD=plan.g_pad, BQ=plan.bq, DA=DA, DB=DB, DV=DV, BN=plan.bn,
            BLOCK_SIZE=plan.block_size, KPOS_MODE=plan.kpos_mode, WRITE_LSE=write_lse,
            V_FROM_K=v_from_k, num_warps=plan.num_warps, num_stages=plan.num_stages,
        )
    n_groups = plan.groups.shape[0]
    if n_groups:
        _msa_merge[(n_groups,)](
            out, lse_arg, op, lsep, plan.tile_rows, plan.groups,
            out.stride(0), out.stride(1),
            lse.stride(0) if write_lse else 0, lse.stride(1) if write_lse else 0,
            G=G, G_PAD=plan.g_pad, BQ=plan.bq, D=DV, D_PAD=DV, WRITE_LSE=write_lse,
            num_warps=4,
        )
    if write_lse:
        return out, lse
    return out


def context_index(bt_rows: torch.Tensor, starts, lens, cache_block_size: int,
                  page: int = 16):
    """Gather index for a dense context buffer of several requests.

    bt_rows: [G, max_blocks] int32 (GPU) block-table rows of the G requests;
    starts / lens: CPU ints, request g needs positions [starts[g], starts[g] + lens[g]).
    Returns (src_slots [N] int64 on the GPU: cache slot of every dense row, padding rows
    point at slot 0; dense_bt [G, nblk] int32: identity block table of the dense buffer
    viewed as pages of `page`; kv_lens np.int64 [G]; kblk_pos [G, nblk] int32 or None
    when every start is 0 (local index = position)).
    """
    starts = np.asarray(starts, dtype=np.int64)
    lens = np.asarray(lens, dtype=np.int64)
    G = len(lens)
    nblk = -(-lens // page)
    off = np.zeros(G + 1, dtype=np.int64)
    np.cumsum(nblk * page, out=off[1:])
    N = int(off[-1])
    grp = np.repeat(np.arange(G), nblk * page)
    loc = np.arange(N) - off[grp]
    pos = starts[grp] + loc
    valid = loc < lens[grp]
    pos = np.where(valid, pos, 0)
    dev = bt_rows.device
    grp_t = torch.from_numpy(grp).to(dev, non_blocking=True)
    pos_t = torch.from_numpy(pos).to(dev, non_blocking=True)
    blk = bt_rows[grp_t, pos_t // cache_block_size].to(torch.int64)
    src = blk * cache_block_size + pos_t % cache_block_size
    src = torch.where(torch.from_numpy(valid).to(dev, non_blocking=True), src, 0)
    maxb = int(nblk.max()) if G else 0
    dbt = np.zeros((G, max(1, maxb)), dtype=np.int32)
    kbp = None
    if (starts != 0).any():
        kbp = np.zeros((G, max(1, maxb)), dtype=np.int64)
    for g in range(G):
        dbt[g, : nblk[g]] = off[g] // page + np.arange(nblk[g])
        if kbp is not None:
            kbp[g, : nblk[g]] = starts[g] + page * np.arange(nblk[g])
            kbp[g, nblk[g]:] = 1 << 30
    return src, torch.from_numpy(dbt).to(dev, non_blocking=True), lens, kbp


def mla_reference(q_nope, q_pe, k_nope, k_pe, v, qpos, qreq, kpos, kreq, scale,
                  row_chunk: int = 256):
    """Dense fp32 reference over explicit keys (MHA form).

    q_nope [rows, H, P], q_pe [rows, H, R], k_nope [n, H, P], k_pe [n, R],
    v [n, H, V], positions / request ids as int tensors. Mask: same request and
    kpos <= qpos. Returns (O [rows, H, V] fp32, LSE [rows, H] fp32, -inf = empty).
    Runs per request and per chunk of query rows to bound memory.
    """
    rows, H, _ = q_nope.shape
    o = torch.zeros(rows, H, v.shape[-1], device=q_nope.device, dtype=torch.float32)
    lse = torch.full((rows, H), float("-inf"), device=q_nope.device)
    for r in torch.unique(qreq).tolist():
        qi = (qreq == r).nonzero(as_tuple=True)[0]
        ki = (kreq == r).nonzero(as_tuple=True)[0]
        kf = torch.cat([k_nope[ki], k_pe[ki][:, None, :].expand(-1, H, -1)], -1).float()
        vf = v[ki].float()
        kp = kpos[ki]
        for c in range(0, qi.numel(), row_chunk):
            rr = qi[c:c + row_chunk]
            qf = torch.cat([q_nope[rr], q_pe[rr]], -1).float()
            s = torch.einsum("qhd,khd->hqk", qf, kf) * scale
            allowed = kp[None, :] <= qpos[rr][:, None]
            s = s.masked_fill(~allowed[None], float("-inf"))
            ls = torch.logsumexp(s, -1)
            p = torch.nan_to_num(torch.exp(s - torch.where(torch.isinf(ls), 0.0, ls)[..., None]))
            o[rr] = torch.einsum("hqk,khd->qhd", p, vf)
            lse[rr] = ls.transpose(0, 1)
    return o, lse
