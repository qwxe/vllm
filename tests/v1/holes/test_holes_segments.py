# SPDX-License-Identifier: Apache-2.0
"""holes/core: model-runner segment view and its attention semantics (CPU)."""

import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.hole_segments import (
    build_hole_segments,
    segment_common_attn_metadata,
)


def _positions(num_computed, num_sched):
    return np.concatenate(
        [nc + np.arange(n) for nc, n in zip(num_computed, num_sched)]
    ).astype(np.int64)


def test_build_segments():
    req_ids = ["x", "h", "y"]
    nc = np.array([10, 160, 50], dtype=np.int32)
    ns = np.array([1, 48 + 32 + 5, 7], dtype=np.int32)
    pos = _positions(nc, ns)
    seg = build_hole_segments(req_ids, {"h": [(16, 64), (96, 128)]}, ns, nc, pos)
    assert seg.seg_req_idx.tolist() == [0, 1, 1, 1, 2]
    assert seg.seg_query_lens.tolist() == [1, 48, 32, 5, 7]
    assert seg.seg_ends.tolist() == [11, 64, 128, 165, 57]
    assert seg.req_hole_tokens.tolist() == [0, 80, 0]
    assert pos[1:49].tolist() == list(range(16, 64))
    assert pos[49:81].tolist() == list(range(96, 128))
    assert pos[81:86].tolist() == list(range(160, 165))
    assert pos[86:].tolist() == list(range(50, 57))


def test_build_segments_hole_only_step():
    nc = np.array([160], dtype=np.int32)
    ns = np.array([32], dtype=np.int32)
    pos = _positions(nc, ns)
    seg = build_hole_segments(["h"], {"h": [(0, 32)]}, ns, nc, pos)
    assert seg.seg_ends.tolist() == [32] and pos.tolist() == list(range(32))


def _varlen_causal(q, k_cache, v_cache, cu_q, seq_lens, block_table, bs):
    """Reference of the backends' varlen paged attention: query t of entry i
    sees keys [0, seq_len_i - q_len_i + t] (bottom-right causal)."""
    out = torch.empty_like(q)
    for i in range(len(seq_lens)):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        L = int(seq_lens[i])
        slots = [int(block_table[i, p // bs]) * bs + p % bs for p in range(L)]
        k, v = k_cache[slots], v_cache[slots]
        ql = qe - qs
        for t in range(ql):
            lim = L - ql + t + 1
            att = torch.softmax(q[qs + t] @ k[:lim].T / 8.0, dim=-1)
            out[qs + t] = att @ v[:lim]
    return out


def test_segment_metadata_equals_full_prefill():
    torch.manual_seed(0)
    bs, d, N = 16, 16, 160 + 5  # frontier 160, tail 5
    holes = [(32, 80)]
    # "KV of a position" (independent of how it was computed).
    K = torch.randn(N, d)
    V = torch.randn(N, d)
    Q = torch.randn(N, d)
    # Paged cache with a shuffled block table; the hole blocks start empty.
    nblk = (N + bs - 1) // bs
    table = torch.randperm(64)[:nblk].view(1, -1)
    k_cache = torch.full((64 * bs, d), float("nan"))
    v_cache = torch.full((64 * bs, d), float("nan"))
    resident = [p for p in range(160) if not any(s <= p < e for s, e in holes)]
    for p in resident:
        slot = int(table[0, p // bs]) * bs + p % bs
        k_cache[slot], v_cache[slot] = K[p], V[p]

    nc = np.array([160], dtype=np.int32)
    ns = np.array([48 + 5], dtype=np.int32)
    pos = _positions(nc, ns)
    seg = build_hole_segments(["r"], {"r": holes}, ns, nc, pos)
    # Write this step's KV (reshape_and_cache via slot_mapping from positions).
    for p in pos.tolist():
        slot = int(table[0, p // bs]) * bs + p % bs
        k_cache[slot], v_cache[slot] = K[p], V[p]

    cm = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 53], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 53], dtype=torch.int32),
        seq_lens=torch.tensor([165], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=53,
        max_query_len=53,
        max_seq_len=165,
        block_table_tensor=table.to(torch.int32),
        slot_mapping=torch.zeros(53, dtype=torch.int64),
    )
    scm = segment_common_attn_metadata(cm, seg, torch.device("cpu"))
    assert scm.num_reqs == 2 and scm.query_start_loc_cpu.tolist() == [0, 48, 53]
    assert scm.seq_lens.tolist() == [80, 165] and scm.max_seq_len == 165
    assert scm.block_table_tensor.shape[0] == 2
    out = _varlen_causal(
        Q[pos], k_cache, v_cache, scm.query_start_loc_cpu, scm.seq_lens,
        scm.block_table_tensor, bs,
    )
    # Full causal attention over the whole sequence, at the same positions.
    full = torch.softmax((Q @ K.T / 8.0).masked_fill(
        torch.triu(torch.ones(N, N, dtype=torch.bool), 1), float("-inf")), -1) @ V
    torch.testing.assert_close(out, full[pos], rtol=1e-5, atol=1e-5)
    assert not torch.isnan(out).any()


if __name__ == "__main__":
    import sys

    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
