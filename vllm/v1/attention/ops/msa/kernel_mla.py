# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA kernel variant for MLA: split qk dims, separate V dim (Triton).

Same work list, tile table, key positions (KPOS_MODE 0/1/2), split-KV slots and
output conventions as `kernel._msa_fwd`; only the dot products change:

* qk = part A (dim DA) + optional part B (dim DB, 0 = absent), each read from its
  own pointer with its own strides. Part B of K may have head stride 0 (MLA's
  k_pe is shared by all heads).
* V has its own dim DV and pointer; V_FROM_K reuses the loaded K part-A tile
  (absorbed MLA: V is the latent itself, DV == DA).

Two MLA uses (MLA-MSA.md section 2):
* compute path: MHA, Hkv = Hq, DA = 128 (k_nope), DB = 64 (k_pe), DV = 128, over a
  dense up-projected buffer viewed as pages;
* absorbed path: MQA, Hkv = 1, DA = 512 (latent), DB = 64 (k_pe), V_FROM_K, over
  vLLM's paged MLA cache [num_blocks, block_size, 576].

All of DA, DB, DV must be powers of two (MLA's are). Split-KV partials are
written as OP[slot, M, DV] / LSEP[slot, M] and folded by `kernel._msa_merge`
with D = D_PAD = DV.
"""

import triton
import triton.language as tl


@triton.jit
def _attend_tile_mla(
    acc, m_i, l_i, qa, qb, qp,
    KA, KB, V, block_table_ptr, kblk_pos_ptr, req,
    kstart, k1,
    s_kab, s_kas, s_kah, s_kbb, s_kbs, s_kbh, s_vb, s_vs, s_vh,
    stride_bt, stride_kp, kvh, sm_scale_log2,
    offs_n, offs_da, offs_db, offs_dv,
    DB: tl.constexpr, BN: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr, KPOS_MODE: tl.constexpr, V_FROM_K: tl.constexpr,
):
    kidx = kstart + offs_n
    kvalid = kidx < k1
    if MASKED:
        blk_local = tl.where(kvalid, kidx // BLOCK_SIZE, 0)
    else:
        blk_local = kidx // BLOCK_SIZE
    blk = tl.load(block_table_ptr + req * stride_bt + blk_local).to(tl.int64)
    slot = kidx % BLOCK_SIZE
    ka_ptrs = (KA + blk[None, :] * s_kab + slot[None, :] * s_kas + kvh * s_kah
               + offs_da[:, None])
    if MASKED:
        ka = tl.load(ka_ptrs, mask=kvalid[None, :], other=0.0)
    else:
        ka = tl.load(ka_ptrs)
    s = tl.dot(qa, ka)
    if DB > 0:
        kb_ptrs = (KB + blk[None, :] * s_kbb + slot[None, :] * s_kbs + kvh * s_kbh
                   + offs_db[:, None])
        if MASKED:
            kb = tl.load(kb_ptrs, mask=kvalid[None, :], other=0.0)
        else:
            kb = tl.load(kb_ptrs)
        s = tl.dot(qb, kb, s)
    s = s * sm_scale_log2
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
    if V_FROM_K:
        v = tl.trans(ka)
    else:
        v_ptrs = (V + blk[:, None] * s_vb + slot[:, None] * s_vs + kvh * s_vh
                  + offs_dv[None, :])
        if MASKED:
            v = tl.load(v_ptrs, mask=kvalid[:, None], other=0.0)
        else:
            v = tl.load(v_ptrs)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, m_new, l_i


@triton.jit
def _msa_mla_fwd(
    QA, QB, KA, KB, V, O, LSE, OP, LSEP,
    qpos_ptr, tile_rows_ptr, items_ptr, block_table_ptr, kblk_pos_ptr,
    sm_scale_log2,
    s_qat, s_qah, s_qbt, s_qbh, stride_ot, stride_oh, stride_lt, stride_lh,
    s_kab, s_kas, s_kah, s_kbb, s_kbs, s_kbh, s_vb, s_vs, s_vh,
    stride_bt, stride_kp,
    G: tl.constexpr, G_PAD: tl.constexpr, BQ: tl.constexpr,
    DA: tl.constexpr, DB: tl.constexpr, DV: tl.constexpr, BN: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, KPOS_MODE: tl.constexpr, WRITE_LSE: tl.constexpr,
    V_FROM_K: tl.constexpr,
):
    M: tl.constexpr = BQ * G_PAD
    DBP: tl.constexpr = DB if DB > 0 else 16
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
    offs_da = tl.arange(0, DA)
    offs_db = tl.arange(0, DBP)
    offs_dv = tl.arange(0, DV)
    tok = offs_m // G_PAD
    hh = offs_m % G_PAD
    row = tl.load(tile_rows_ptr + tile * BQ + tok)
    row_ok = (row >= 0) & (hh < G)
    row_safe = tl.where(row >= 0, row, 0).to(tl.int64)
    qh = kvh * G + tl.where(hh < G, hh, 0)
    qp = tl.load(qpos_ptr + row_safe, mask=row >= 0, other=-1)
    qa = tl.load(QA + row_safe[:, None] * s_qat + qh[:, None] * s_qah + offs_da[None, :],
                 mask=row_ok[:, None], other=0.0)
    if DB > 0:
        qb = tl.load(QB + row_safe[:, None] * s_qbt + qh[:, None] * s_qbh + offs_db[None, :],
                     mask=row_ok[:, None], other=0.0)
    else:
        qb = qa

    m_i = tl.full([M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([M], dtype=tl.float32)
    acc = tl.zeros([M, DV], dtype=tl.float32)

    kf = tl.minimum(kfull, k1)
    n_full = tl.maximum(kf - k0, 0) // BN
    for t in range(0, n_full):
        acc, m_i, l_i = _attend_tile_mla(
            acc, m_i, l_i, qa, qb, qp, KA, KB, V, block_table_ptr, kblk_pos_ptr, req,
            k0 + t * BN, k1,
            s_kab, s_kas, s_kah, s_kbb, s_kbs, s_kbh, s_vb, s_vs, s_vh,
            stride_bt, stride_kp, kvh, sm_scale_log2, offs_n, offs_da, offs_db, offs_dv,
            DB, BN, BLOCK_SIZE, False, KPOS_MODE, V_FROM_K)
    for kstart in range(k0 + n_full * BN, k1, BN):
        acc, m_i, l_i = _attend_tile_mla(
            acc, m_i, l_i, qa, qb, qp, KA, KB, V, block_table_ptr, kblk_pos_ptr, req,
            kstart, k1,
            s_kab, s_kas, s_kah, s_kbb, s_kbs, s_kbh, s_vb, s_vs, s_vh,
            stride_bt, stride_kp, kvh, sm_scale_log2, offs_n, offs_da, offs_db, offs_dv,
            DB, BN, BLOCK_SIZE, True, KPOS_MODE, V_FROM_K)

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / l_safe[:, None]
    lse2 = tl.where(l_i == 0.0, float("-inf"), m_i + tl.log2(l_safe))
    if oslot < 0:
        o_ptrs = O + row_safe[:, None] * stride_ot + qh[:, None] * stride_oh + offs_dv[None, :]
        tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=row_ok[:, None])
        if WRITE_LSE:
            tl.store(LSE + row_safe * stride_lt + qh * stride_lh, lse2 * 0.6931471805599453,
                     mask=row_ok)
    else:
        base = oslot.to(tl.int64) * M
        tl.store(OP + (base + offs_m)[:, None] * DV + offs_dv[None, :], out)
        tl.store(LSEP + base + offs_m, lse2)
