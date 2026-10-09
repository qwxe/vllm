# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Synced from the standalone repo /home/asd/vllm-dev-kernels (msa/).
"""Triton kernels for position-explicit multi-segment paged attention (MSA).

Derived from vLLM v0.19.0's `triton_unified_attention` (GQA-packed q tiles over a
paged KV cache in the NHD layout `[num_blocks, block_size, num_kv_heads, head_size]`),
generalised to a host-built work list:

* every query row carries an absolute position (`qpos[row]`);
* every key carries an absolute position, either its local index (contiguous cache)
  or `kblk_pos[req, j] + slot` for local block j (strided / received buffers);
* mask(i, j) = kpos(j) <= qpos(i)  (rows of one tile always belong to one request);
* a work item is (q tile, kv head, local key range [k0, k1)); key tiles below
  `kfull` are visible to every row of the tile and run without the mask.

Item table (int32, ITEM_W columns):
    0 tile   1 kv_head   2 k0   3 k1   4 kfull   5 slot (-1: write O directly)   6 req
Tile table: `tile_rows[tile, BQ]` = row index into Q (or -1 for padding).
Partials (split-KV items): `OP[slot, M, D]` fp32 normalised outputs and `LSEP[slot, M]`
(log2 domain); `_msa_merge` folds each split group back into O / LSE.
"""

import triton
import triton.language as tl

ITEM_W = 8
GROUP_W = 4  # tile, kv_head, first_slot, n_split

LN2 = 0.6931471805599453


@triton.jit
def _attend_tile(
    acc, m_i, l_i, q, qp,
    K, V, block_table_ptr, kblk_pos_ptr, req,
    kstart, k1,
    stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
    stride_bt, stride_kp, kvh, sm_scale_log2,
    offs_n, offs_d, dmask,
    D_PAD: tl.constexpr, BN: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr, KPOS_MODE: tl.constexpr, NEED_DMASK: tl.constexpr,
):
    kidx = kstart + offs_n
    if MASKED:
        kvalid = kidx < k1
        blk_local = tl.where(kvalid, kidx // BLOCK_SIZE, 0)
    else:
        blk_local = kidx // BLOCK_SIZE
    blk = tl.load(block_table_ptr + req * stride_bt + blk_local).to(tl.int64)
    slot = kidx % BLOCK_SIZE
    k_ptrs = (K + blk[None, :] * stride_kb + slot[None, :] * stride_ks
              + kvh * stride_kh + offs_d[:, None])
    v_ptrs = (V + blk[:, None] * stride_vb + slot[:, None] * stride_vs
              + kvh * stride_vh + offs_d[None, :])
    if MASKED:
        if NEED_DMASK:
            k = tl.load(k_ptrs, mask=dmask[:, None] & kvalid[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=dmask[None, :] & kvalid[:, None], other=0.0)
        else:
            k = tl.load(k_ptrs, mask=kvalid[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=kvalid[:, None], other=0.0)
    else:
        if NEED_DMASK:
            k = tl.load(k_ptrs, mask=dmask[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=dmask[None, :], other=0.0)
        else:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)

    s = tl.dot(q, k) * sm_scale_log2
    if MASKED:
        if KPOS_MODE == 1:    # per-block start position
            kp = tl.load(kblk_pos_ptr + req * stride_kp + blk_local) + slot
        elif KPOS_MODE == 2:  # per-key position table [req, local key]
            kp = tl.load(kblk_pos_ptr + req * stride_kp + tl.where(kvalid, kidx, 0))
        else:                 # contiguous: local index = position
            kp = kidx
        allowed = (kp[None, :] <= qp[:, None]) & kvalid[None, :]
        s = tl.where(allowed, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    p = tl.exp2(s - m_use[:, None])
    alpha = tl.exp2(m_i - m_use)
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, m_new, l_i


@triton.jit
def _msa_fwd(
    Q, K, V, O, LSE, OP, LSEP,
    qpos_ptr, tile_rows_ptr, items_ptr, block_table_ptr, kblk_pos_ptr,
    sm_scale_log2,
    stride_qt, stride_qh, stride_ot, stride_oh, stride_lt, stride_lh,
    stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
    stride_bt, stride_kp,
    G: tl.constexpr, G_PAD: tl.constexpr, BQ: tl.constexpr,
    D: tl.constexpr, D_PAD: tl.constexpr, BN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, KPOS_MODE: tl.constexpr, WRITE_LSE: tl.constexpr,
):
    M: tl.constexpr = BQ * G_PAD
    NEED_DMASK: tl.constexpr = D != D_PAD
    pid = tl.program_id(0)
    it = items_ptr + pid * 8
    tile = tl.load(it + 0)
    kvh = tl.load(it + 1)
    k0 = tl.load(it + 2)
    k1 = tl.load(it + 3)
    kfull = tl.load(it + 4)
    oslot = tl.load(it + 5)
    req = tl.load(it + 6)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D_PAD)
    dmask = offs_d < D
    tok = offs_m // G_PAD
    hh = offs_m % G_PAD
    row = tl.load(tile_rows_ptr + tile * BQ + tok)
    row_ok = (row >= 0) & (hh < G)
    row_safe = tl.where(row >= 0, row, 0).to(tl.int64)
    qh = kvh * G + tl.where(hh < G, hh, 0)
    qp = tl.load(qpos_ptr + row_safe, mask=row >= 0, other=-1)
    q = tl.load(Q + row_safe[:, None] * stride_qt + qh[:, None] * stride_qh + offs_d[None, :],
                mask=row_ok[:, None] & dmask[None, :], other=0.0)

    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D_PAD], dtype=tl.float32)

    # unmasked key tiles: [k0, k0 + n_full * BN) lie below kfull and k1
    kf = tl.minimum(kfull, k1)
    n_full = tl.maximum(kf - k0, 0) // BN
    for t in range(0, n_full):
        acc, m_i, l_i = _attend_tile(
            acc, m_i, l_i, q, qp, K, V, block_table_ptr, kblk_pos_ptr, req,
            k0 + t * BN, k1,
            stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
            stride_bt, stride_kp, kvh, sm_scale_log2, offs_n, offs_d, dmask,
            D_PAD, BN, BLOCK_SIZE, False, KPOS_MODE, NEED_DMASK)
    for kstart in range(k0 + n_full * BN, k1, BN):
        acc, m_i, l_i = _attend_tile(
            acc, m_i, l_i, q, qp, K, V, block_table_ptr, kblk_pos_ptr, req,
            kstart, k1,
            stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
            stride_bt, stride_kp, kvh, sm_scale_log2, offs_n, offs_d, dmask,
            D_PAD, BN, BLOCK_SIZE, True, KPOS_MODE, NEED_DMASK)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    lse2 = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(l_safe))
    if oslot < 0:
        o_ptrs = O + row_safe[:, None] * stride_ot + qh[:, None] * stride_oh + offs_d[None, :]
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=row_ok[:, None] & dmask[None, :])
        if WRITE_LSE:
            tl.store(LSE + row_safe * stride_lt + qh * stride_lh, lse2 * 0.6931471805599453,
                     mask=row_ok)
    else:
        base = oslot.to(tl.int64) * M
        tl.store(OP + (base + offs_m)[:, None] * D_PAD + offs_d[None, :], out)
        tl.store(LSEP + base + offs_m, lse2)


@triton.jit
def _msa_merge(
    O, LSE, OP, LSEP, tile_rows_ptr, groups_ptr,
    stride_ot, stride_oh, stride_lt, stride_lh,
    G: tl.constexpr, G_PAD: tl.constexpr, BQ: tl.constexpr,
    D: tl.constexpr, D_PAD: tl.constexpr, WRITE_LSE: tl.constexpr,
):
    M: tl.constexpr = BQ * G_PAD
    pid = tl.program_id(0)
    gp = groups_ptr + pid * 4
    tile = tl.load(gp + 0)
    kvh = tl.load(gp + 1)
    first = tl.load(gp + 2)
    nsplit = tl.load(gp + 3)
    offs_m = tl.arange(0, M)
    offs_d = tl.arange(0, D_PAD)
    dmask = offs_d < D
    tok = offs_m // G_PAD
    hh = offs_m % G_PAD
    row = tl.load(tile_rows_ptr + tile * BQ + tok)
    row_ok = (row >= 0) & (hh < G)
    row_safe = tl.where(row >= 0, row, 0).to(tl.int64)
    qh = kvh * G + tl.where(hh < G, hh, 0)

    m = tl.full([M], float("-inf"), dtype=tl.float32)
    w = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, D_PAD], dtype=tl.float32)
    for s in range(0, nsplit):
        base = (first + s).to(tl.int64) * M
        lse_s = tl.load(LSEP + base + offs_m)
        o_s = tl.load(OP + (base + offs_m)[:, None] * D_PAD + offs_d[None, :])
        m_new = tl.maximum(m, lse_s)
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp2(m - m_use)
        b = tl.exp2(lse_s - m_use)
        w = w * a + b
        acc = acc * a[:, None] + o_s * b[:, None]
        m = m_new
    w_safe = tl.where(w == 0.0, 1.0, w)
    out = acc / w_safe[:, None]
    o_ptrs = O + row_safe[:, None] * stride_ot + qh[:, None] * stride_oh + offs_d[None, :]
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=row_ok[:, None] & dmask[None, :])
    if WRITE_LSE:
        lse2 = tl.where(w == 0.0, float("-inf"), m + tl.log2(w_safe))
        tl.store(LSE + row_safe * stride_lt + qh * stride_lh, lse2 * 0.6931471805599453,
                 mask=row_ok)


@triton.jit
def _merge_states(
    O, LSE, OA, LSEA, OB, LSEB, n_rows, H,
    stride_ot, stride_oh, stride_at, stride_ah, stride_bt, stride_bh,
    stride_lt, stride_la, stride_lb,
    D: tl.constexpr, D_PAD: tl.constexpr, BR: tl.constexpr,
):
    """Merge two partial attention states (O, natural-log LSE), rows x heads."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    r = pid * BR + tl.arange(0, BR)
    rm = r < n_rows
    offs_d = tl.arange(0, D_PAD)
    dm = offs_d < D
    la = tl.load(LSEA + r * stride_la + h, mask=rm, other=float("-inf"))
    lb = tl.load(LSEB + r * stride_lb + h, mask=rm, other=float("-inf"))
    mx = tl.maximum(la, lb)
    mu = tl.where(mx == float("-inf"), 0.0, mx)
    wa = tl.exp(la - mu)
    wb = tl.exp(lb - mu)
    ws = wa + wb
    wsafe = tl.where(ws == 0.0, 1.0, ws)
    oa = tl.load(OA + r[:, None] * stride_at + h * stride_ah + offs_d[None, :],
                 mask=rm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    ob = tl.load(OB + r[:, None] * stride_bt + h * stride_bh + offs_d[None, :],
                 mask=rm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    o = (oa * wa[:, None] + ob * wb[:, None]) / wsafe[:, None]
    tl.store(O + r[:, None] * stride_ot + h * stride_oh + offs_d[None, :],
             o.to(O.dtype.element_ty), mask=rm[:, None] & dm[None, :])
    lse = tl.where(ws == 0.0, float("-inf"), mu + tl.log(wsafe))
    tl.store(LSE + r * stride_lt + h, lse, mask=rm)
