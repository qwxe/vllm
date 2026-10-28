# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Synced from the standalone MSA kernel repo (its msa/ package).
"""Host planner for MSA: segments -> q tiles -> work items (KERNEL-ALGO.md 2.5 / 2.7).

Input is a CPU description of the launch:

* segments: (req, row_start, n_rows, pos_start). Rows [row_start, row_start + n_rows) of Q
  hold positions pos_start .. pos_start + n_rows - 1 of request `req`. A segment is a hole
  piece, a tail chunk, a pass-Q row block, or a ring-step query block -- the planner does
  not care.
* kv_lens[req]: number of LOCAL keys of the request (its block table holds them).
* kblk_pos (optional, CPU int array [num_reqs, max_blocks]): absolute position of slot 0
  of each local block; each block holds block_size consecutive positions, and starts are
  increasing. None = contiguous cache (local key j has position j).

Schedules (`mode`):
    'a'    one tile chain per segment, no split (vLLM varlen with one entry per segment)
    'b'    + position packing of sub-tile pieces of one request
    'c'    + split-KV (chain cap), no packing
    'bc'   packing and split-KV
    'auto' = 'bc' with the selection rule (pack only >= 2 pieces within W_pack; split only
           when the launch would under-fill the GPU), which is what 'bc' does anyway.
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from .kernel import ITEM_W, GROUP_W


def _next_pow2(x: int) -> int:
    return 1 << (max(1, x) - 1).bit_length()


@dataclass
class MsaPlan:
    num_q_per_kv: int
    g_pad: int
    bq: int                      # query tokens per tile
    bn: int                      # keys per KV tile
    block_size: int
    kpos_mode: int               # 0 contiguous, 1 per-block starts, 2 per-key table
    items: torch.Tensor          # [n_items, ITEM_W] int32 (LPT order)
    tile_rows: torch.Tensor      # [n_tiles, bq] int32
    groups: torch.Tensor         # [n_groups, GROUP_W] int32
    qpos: torch.Tensor           # [num_rows] int32
    n_slots: int
    num_warps: int = 8
    num_stages: int = 2
    stats: dict = field(default_factory=dict)

    @property
    def m_rows(self) -> int:
        return self.bq * self.g_pad


def _launch_config(m_rows, elem_size, bn, num_warps, num_stages):
    """Tuned on the RTX 4050 (bench/tune.py): short tiles stream KV (BN 128, 3 stages), full
    tiles are compute-bound (BN 64, 8 warps, 2 stages). fp32 inputs need smaller K/V tiles to
    fit shared memory (~100 KB on SM 8.x)."""
    if bn is None:
        bn = (128 if m_rows <= 64 else 64) if elem_size <= 2 else 32
    if num_warps is None:
        num_warps = 4 if m_rows <= 64 else 8
    if num_stages is None:
        num_stages = (3 if m_rows <= 64 else 2) if elem_size <= 2 else 2
    return bn, num_warps, num_stages


class _KeyLayout:
    def __init__(self, kv_lens, kblk_pos, block_size, key_pos=None):
        self.kv_lens = np.asarray(kv_lens, dtype=np.int64)
        self.kblk_pos = None if kblk_pos is None else np.asarray(kblk_pos, dtype=np.int64)
        self.key_pos = key_pos  # list of sorted per-request key position arrays (mode 2)
        self.bs = block_size
        self.mode = 2 if key_pos is not None else (1 if kblk_pos is not None else 0)

    def count_le(self, req: int, p: np.ndarray) -> np.ndarray:
        """Number of local keys of `req` whose absolute position is <= p (vectorised)."""
        p = np.asarray(p, dtype=np.int64)
        kv = self.kv_lens[req]
        if self.key_pos is not None:
            return np.searchsorted(self.key_pos[req][:kv], p, side="right")
        if self.kblk_pos is None:
            return np.clip(p + 1, 0, kv)
        nblk = -(-kv // self.bs)
        starts = self.kblk_pos[req, :nblk]
        j = np.searchsorted(starts, p, side="right") - 1
        jj = np.maximum(j, 0)
        within = np.minimum(self.bs, p - starts[jj] + 1)
        cnt = jj * self.bs + within
        cnt = np.where(j < 0, 0, cnt)
        return np.clip(cnt, 0, kv)


def plan_msa(
    segments,
    kv_lens,
    num_kv_heads: int,
    num_q_per_kv: int,
    *,
    block_size: int = 16,
    kblk_pos=None,
    mode: str = "auto",
    num_sms: int | None = None,
    device="cuda",
    rows_per_tile: int = 128,
    bn: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.05,
    c: float = 2.0,
    min_split: int | None = None,
    qpos: torch.Tensor | None = None,
    num_rows: int | None = None,
    elem_size: int = 2,
) -> MsaPlan:
    """Build the work list. `rows_per_tile` is M = BQ * g_pad (rows of (token, head))."""
    seg = np.asarray(segments, dtype=np.int64).reshape(-1, 4)
    pack = mode in ("b", "bc", "auto")
    split = mode in ("c", "bc", "auto")
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    g_pad = _next_pow2(num_q_per_kv)
    # rule 3 (as built): the tile height follows the pieces. A launch whose segments are all
    # short gets the smallest power-of-two tile that holds the longest one (>= 32 rows), so
    # short deep pieces do not waste rows and more CTAs stream KV at once.
    max_seg = int(seg[:, 2].max()) if seg.shape[0] else 1
    bq_cap = max(1, rows_per_tile // g_pad)
    bq = min(bq_cap, max(max(1, 32 // g_pad), _next_pow2(max_seg)))
    m_rows = bq * g_pad
    bn, num_warps, num_stages = _launch_config(m_rows, elem_size, bn, num_warps, num_stages)
    if min_split is None:
        min_split = 4 * bn
    lay = _KeyLayout(kv_lens, kblk_pos, block_size)

    # ---- tiles: (req, qmin, qmax, rows[bq]) ----------------------------------------
    t_req, t_qmin, t_qmax, t_rows = [], [], [], []
    n_packed_tiles = 0
    n_pieces_packed = 0

    def emit(req, chunks):
        """chunks: list of (row0, n, pos0) filling one tile."""
        nonlocal n_packed_tiles, n_pieces_packed
        rows = np.concatenate([np.arange(r0, r0 + n) for r0, n, _ in chunks])
        padded = np.full(bq, -1, dtype=np.int64)
        padded[: len(rows)] = rows
        t_req.append(np.array([req]))
        t_qmin.append(np.array([min(p0 for _, _, p0 in chunks)]))
        t_qmax.append(np.array([max(p0 + n - 1 for _, n, p0 in chunks)]))
        t_rows.append(padded[None, :])
        if len(chunks) > 1:
            n_packed_tiles += 1
            n_pieces_packed += len(chunks)

    if not pack:
        # (a)/(c): one tile chain per segment
        for req, r0, n, p0 in seg:
            if n <= 0:
                continue
            nt = -(-int(n) // bq)
            starts = np.arange(nt, dtype=np.int64) * bq
            cnt = np.minimum(bq, n - starts)
            rows = r0 + starts[:, None] + np.arange(bq)[None, :]
            rows = np.where(np.arange(bq)[None, :] < cnt[:, None], rows, -1)
            t_req.append(np.full(nt, req))
            t_qmin.append(p0 + starts)
            t_qmax.append(p0 + starts + cnt - 1)
            t_rows.append(rows)
    else:
        # (b): per request, rows in position order fill tiles greedily; a piece may straddle
        # two tiles; a tile closes early when its position spread would exceed W_pack.
        for req in np.unique(seg[:, 0]):
            ss = seg[seg[:, 0] == req]
            ss = ss[np.argsort(ss[:, 3], kind="stable")]
            hi_all = int((ss[:, 3] + ss[:, 2]).max()) - 1
            w_pack = max(bn, int(eps * hi_all))
            cur, cur_n, cur_lo = [], 0, None
            for _, r0, n, p0 in ss:
                r0, n, p0 = int(r0), int(n), int(p0)
                while n > 0:
                    if cur and (p0 + min(n, bq - cur_n) - 1 - cur_lo > w_pack):
                        emit(int(req), cur)
                        cur, cur_n, cur_lo = [], 0, None
                    if not cur and n >= bq:
                        # whole tiles straight from the segment (vectorised)
                        nt = n // bq
                        starts = np.arange(nt, dtype=np.int64) * bq
                        t_req.append(np.full(nt, req))
                        t_qmin.append(p0 + starts)
                        t_qmax.append(p0 + starts + bq - 1)
                        t_rows.append(r0 + starts[:, None] + np.arange(bq)[None, :])
                        r0, p0, n = r0 + nt * bq, p0 + nt * bq, n - nt * bq
                        continue
                    take = min(n, bq - cur_n)
                    if not cur:
                        cur_lo = p0
                    cur.append((r0, take, p0))
                    cur_n += take
                    r0, p0, n = r0 + take, p0 + take, n - take
                    if cur_n == bq:
                        emit(int(req), cur)
                        cur, cur_n, cur_lo = [], 0, None
            if cur:
                emit(int(req), cur)

    if t_req:
        t_req = np.concatenate(t_req)
        t_qmin = np.concatenate(t_qmin)
        t_qmax = np.concatenate(t_qmax)
        t_rows = np.concatenate(t_rows, axis=0)
    else:
        t_req = t_qmin = t_qmax = np.zeros(0, dtype=np.int64)
        t_rows = np.zeros((0, bq), dtype=np.int64)

    # ---- key bounds per tile --------------------------------------------------------
    if qpos is None:
        nr = int(num_rows) if num_rows is not None else int((seg[:, 1] + seg[:, 2]).max()) if len(seg) else 0
        qp = np.full(nr, -1, dtype=np.int64)
        for req, r0, n, p0 in seg:
            qp[r0:r0 + n] = p0 + np.arange(n)
        qpos = torch.from_numpy(qp.astype(np.int32)).to(device, non_blocking=True)
    return _finish(t_req, t_qmin, t_qmax, t_rows, lay, num_kv_heads, num_q_per_kv, g_pad, bq, bn,
                   block_size, split, c, num_sms, min_split, num_warps, num_stages, mode, device,
                   qpos, n_packed_tiles, n_pieces_packed)


def _finish(t_req, t_qmin, t_qmax, t_rows, lay, num_kv_heads, num_q_per_kv, g_pad, bq, bn,
            block_size, split, c, num_sms, min_split, num_warps, num_stages, mode, device,
            qpos, n_packed_tiles, n_pieces_packed):
    """Key bounds per tile -> items (tiles x kv heads) -> split (c) -> LPT -> device tensors."""
    n_tiles = len(t_req)
    kend = np.zeros(n_tiles, dtype=np.int64)
    kfull = np.zeros(n_tiles, dtype=np.int64)
    for req in np.unique(t_req):
        sel = t_req == req
        kend[sel] = lay.count_le(int(req), t_qmax[sel])
        kfull[sel] = lay.count_le(int(req), t_qmin[sel])

    # ---- items = tiles x kv heads, then split (c) -----------------------------------
    H = num_kv_heads
    it_tile = np.repeat(np.arange(n_tiles), H)
    it_h = np.tile(np.arange(H), n_tiles)
    it_len = np.repeat(kend, H)
    n_items0 = len(it_tile)
    total_keys = int(it_len.sum())
    L = None
    if split and n_items0:
        L = max(min_split, -(-total_keys // int(c * num_sms)))
        L = -(-L // bn) * bn
    if L is not None and (it_len > L).any():
        nsp = np.maximum(1, -(-it_len // L))
    else:
        nsp = np.ones(n_items0, dtype=np.int64)
    n_items = int(nsp.sum())
    rep = np.repeat(np.arange(n_items0), nsp)
    first = np.cumsum(nsp) - nsp
    sub = np.arange(n_items) - np.repeat(first, nsp)
    items = np.zeros((n_items, ITEM_W), dtype=np.int64)
    items[:, 0] = it_tile[rep]
    items[:, 1] = it_h[rep]
    if L is None:
        items[:, 2] = 0
        items[:, 3] = it_len[rep]
    else:
        items[:, 2] = sub * L
        items[:, 3] = np.minimum((sub + 1) * L, it_len[rep])
    items[:, 4] = np.repeat(kfull, H)[rep]
    items[:, 6] = t_req[items[:, 0]]
    # slots for split items, contiguous per group
    is_split = nsp[rep] > 1
    items[:, 5] = -1
    n_slots = int(is_split.sum())
    items[is_split, 5] = np.arange(n_slots)
    gmask = nsp > 1
    gidx = np.nonzero(gmask)[0]
    groups = np.zeros((len(gidx), GROUP_W), dtype=np.int64)
    if len(gidx):
        groups[:, 0] = it_tile[gidx]
        groups[:, 1] = it_h[gidx]
        slot_first = np.cumsum(nsp[gidx]) - nsp[gidx]
        groups[:, 2] = slot_first
        groups[:, 3] = nsp[gidx]
    # LPT: longest key range first (stable)
    order = np.argsort(-(items[:, 3] - items[:, 2]), kind="stable")
    items = items[order]

    def dev(a, w):
        a = np.ascontiguousarray(a.astype(np.int32)).reshape(-1, w)
        return torch.from_numpy(a).to(device, non_blocking=True)

    chain = int((items[:, 3] - items[:, 2]).max()) if n_items else 0
    stats = dict(mode=mode, bq=bq, bn=bn, warps=num_warps, stages=num_stages, m_rows=bq * g_pad, n_tiles=n_tiles, n_items_unsplit=n_items0,
                 n_items=n_items, n_split_groups=len(gidx), split_len=L, max_chain_keys=chain,
                 total_keys=total_keys, packed_tiles=n_packed_tiles, packed_pieces=n_pieces_packed)
    return MsaPlan(
        num_q_per_kv=num_q_per_kv, g_pad=g_pad, bq=bq, bn=bn, block_size=block_size,
        kpos_mode=lay.mode,
        items=dev(items, ITEM_W), tile_rows=dev(t_rows, bq), groups=dev(groups, GROUP_W),
        qpos=qpos, n_slots=n_slots, num_warps=num_warps, num_stages=num_stages, stats=stats,
    )


def plan_msa_rows(
    qpos,
    qreq,
    kv_lens,
    num_kv_heads: int,
    num_q_per_kv: int,
    *,
    block_size: int = 16,
    kblk_pos=None,
    key_pos=None,
    mode: str = "auto",
    num_sms: int | None = None,
    device="cuda",
    rows_per_tile: int = 128,
    bn: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    eps: float = 0.05,
    c: float = 2.0,
    min_split: int | None = None,
    qpos_dev: torch.Tensor | None = None,
    elem_size: int = 2,
) -> MsaPlan:
    """Plan from per-row positions instead of segments (CP / ring-step callers).

    qpos, qreq: CPU int arrays [rows] (any order, any positions). Rows of each request are
    sorted by position and cut into tiles of bq rows; without packing a tile never spans
    more than bq positions (one chain per contiguous run), with packing it may span W_pack.
    key_pos: optional list (per request) of sorted key positions (per-key table, mode 2).
    """
    qp = np.asarray(qpos, dtype=np.int64)
    qr = np.asarray(qreq, dtype=np.int64)
    pack = mode in ("b", "bc", "auto")
    split = mode in ("c", "bc", "auto")
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    g_pad = _next_pow2(num_q_per_kv)
    order = np.lexsort((qp, qr))
    qp_s, qr_s = qp[order], qr[order]
    reqs, starts, counts = np.unique(qr_s, return_index=True, return_counts=True)
    # longest contiguous run decides the tile height
    if len(qp_s):
        brk = np.nonzero((np.diff(qp_s) != 1) | (np.diff(qr_s) != 0))[0] + 1
        runs = np.diff(np.concatenate([[0], brk, [len(qp_s)]]))
        max_run = int(runs.max())
    else:
        max_run = 1
    bq_cap = max(1, rows_per_tile // g_pad)
    bq = min(bq_cap, max(max(1, 32 // g_pad), _next_pow2(max_run)))
    m_rows = bq * g_pad
    bn, num_warps, num_stages = _launch_config(m_rows, elem_size, bn, num_warps, num_stages)
    if min_split is None:
        min_split = 4 * bn
    lay = _KeyLayout(kv_lens, kblk_pos, block_size, key_pos=key_pos)
    t_req, t_qmin, t_qmax, t_rows = [], [], [], []
    n_packed = 0
    for req, s0, n in zip(reqs, starts, counts):
        p = qp_s[s0:s0 + n]
        rows = order[s0:s0 + n]
        span = max(bn, int(eps * int(p[-1]))) if pack else bq - 1
        i = 0
        while i < n:
            end = min(i + bq, n, int(np.searchsorted(p, p[i] + span, side="right")))
            padded = np.full(bq, -1, dtype=np.int64)
            padded[: end - i] = rows[i:end]
            t_req.append(req)
            t_qmin.append(p[i])
            t_qmax.append(p[end - 1])
            t_rows.append(padded)
            if p[end - 1] - p[i] != end - 1 - i:
                n_packed += 1
            i = end
    t_req = np.asarray(t_req, dtype=np.int64)
    t_qmin = np.asarray(t_qmin, dtype=np.int64)
    t_qmax = np.asarray(t_qmax, dtype=np.int64)
    t_rows = np.stack(t_rows) if t_rows else np.zeros((0, bq), dtype=np.int64)
    if qpos_dev is None:
        qpos_dev = torch.from_numpy(qp.astype(np.int32)).to(device, non_blocking=True)
    return _finish(t_req, t_qmin, t_qmax, t_rows, lay, num_kv_heads, num_q_per_kv, g_pad, bq, bn,
                   block_size, split, c, num_sms, min_split, num_warps, num_stages, mode, device,
                   qpos_dev, n_packed, 0)
