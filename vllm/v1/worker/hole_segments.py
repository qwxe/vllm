# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/core: per-segment attention metadata for requests with holes.

A request with holes computes, in one step, some hole pieces [s_i, e_i)
below its frontier and a tail [num_computed, num_computed + n_tail). The
scheduler sends the pieces in ``SchedulerOutput.scheduled_hole_ranges``. The
model runner lays out the request's tokens as the pieces in position order,
then the tail, and:

* sets explicit positions (so token ids, RoPE and ``slot_mapping`` follow);
* keeps the request-level ``query_start_loc`` (logits are taken at the last
  token of each request = the tail's last token; a step without tail tokens
  is a partial prefill whose sample is discarded);
* gives the attention backend one varlen entry per *segment*: query length
  q_i, ``seq_lens`` = segment end e_i, and the request's block table row
  repeated. Causal masking is bottom-right aligned, so query s_i + t sees
  keys [0, s_i + t]: exactly the keys before it, whether resident, computed
  earlier, or written in this step (KV is written before attention in every
  layer). This is KERNEL-ALGO.md variant (a); it works unchanged on
  FlashAttention, FlashInfer and Triton.

Other agents (MSA backend, CP schemes) consume ``HoleSegments`` (attached to
``CommonAttentionMetadata.hole_segments``) for the per-segment view, with
``seg_req_idx`` mapping a segment back to its request row.
"""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HoleSegments:
    """Segment view of one step's batch (all requests, not only holed ones).

    Segments are in token order: segment k covers query tokens
    [cu_q[k], cu_q[k+1]) of the flattened batch.
    """

    seg_req_idx: np.ndarray
    """(S,) int32: batch row (request index) of each segment."""
    seg_query_lens: np.ndarray
    """(S,) int32: q_i."""
    seg_ends: np.ndarray
    """(S,) int32: e_i, the segment's end position = its seq_len."""
    req_hole_tokens: np.ndarray
    """(num_reqs,) int32: hole tokens of each request in this step."""

    @property
    def num_segments(self) -> int:
        return int(self.seg_req_idx.shape[0])

    @property
    def seg_starts(self) -> np.ndarray:
        return self.seg_ends - self.seg_query_lens

    def cu_query_lens(self) -> np.ndarray:
        cu = np.zeros(self.num_segments + 1, dtype=np.int32)
        np.cumsum(self.seg_query_lens, out=cu[1:])
        return cu


def build_hole_segments(
    req_ids: list[str],
    hole_ranges: dict[str, list[tuple[int, int]]],
    num_scheduled_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    positions: np.ndarray,
) -> HoleSegments:
    """Build the segment view and rewrite ``positions`` in place.

    Args:
        req_ids: request id of each batch row.
        hole_ranges: req_id -> hole pieces of this step (sorted).
        num_scheduled_tokens: (num_reqs,) tokens per request (pieces + tail).
        num_computed_tokens: (num_reqs,) frontier of each request.
        positions: (num_tokens,) positions, initialised to
            num_computed + arange per request; holed requests are rewritten.
    """
    num_reqs = len(req_ids)
    seg_req: list[int] = []
    seg_q: list[int] = []
    seg_end: list[int] = []
    hole_tok = np.zeros(num_reqs, dtype=np.int32)
    offset = 0
    for i, req_id in enumerate(req_ids):
        n = int(num_scheduled_tokens[i])
        nc = int(num_computed_tokens[i])
        pieces = hole_ranges.get(req_id)
        if not pieces:
            seg_req.append(i)
            seg_q.append(n)
            seg_end.append(nc + n)
            offset += n
            continue
        h = sum(e - s for s, e in pieces)
        n_tail = n - h
        assert n_tail >= 0, (req_id, pieces, n)
        hole_tok[i] = h
        pos = offset
        prev_end = -1
        for s, e in pieces:
            assert s >= prev_end and e > s and e <= nc, (req_id, pieces, nc)
            prev_end = e
            positions[pos : pos + e - s] = np.arange(s, e)
            seg_req.append(i)
            seg_q.append(e - s)
            seg_end.append(e)
            pos += e - s
        if n_tail > 0:
            positions[pos : pos + n_tail] = np.arange(nc, nc + n_tail)
            seg_req.append(i)
            seg_q.append(n_tail)
            seg_end.append(nc + n_tail)
        offset += n
    return HoleSegments(
        seg_req_idx=np.asarray(seg_req, dtype=np.int32),
        seg_query_lens=np.asarray(seg_q, dtype=np.int32),
        seg_ends=np.asarray(seg_end, dtype=np.int32),
        req_hole_tokens=hole_tok,
    )


def segment_common_attn_metadata(cm, seg: HoleSegments, device: torch.device):
    """Return attention metadata with one entry per segment, without padding."""
    cu = torch.from_numpy(seg.cu_query_lens())
    seq_lens_cpu = torch.from_numpy(seg.seg_ends.copy())
    idx_cpu = torch.from_numpy(seg.seg_req_idx.astype(np.int64))
    idx = idx_cpu.to(device, non_blocking=True)

    def select_rows(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value[seg.seg_req_idx]
        return value.index_select(0, idx_cpu if value.device.type == "cpu" else idx)

    mm_req_doc_ranges = None
    if cm.mm_req_doc_ranges is not None:
        mm_req_doc_ranges = {
            i: cm.mm_req_doc_ranges[int(req_idx)]
            for i, req_idx in enumerate(seg.seg_req_idx)
            if int(req_idx) in cm.mm_req_doc_ranges
        }

    return cm.replace(
        query_start_loc=cu.to(device, non_blocking=True),
        query_start_loc_cpu=cu,
        seq_lens=seq_lens_cpu.to(device, non_blocking=True),
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _num_computed_tokens_cache=None,
        _token_to_req_indices_cache=None,
        num_reqs=seg.num_segments,
        max_query_len=int(seg.seg_query_lens.max()),
        max_seq_len=int(seg.seg_ends.max()),
        block_table_tensor=cm.block_table_tensor.index_select(0, idx),
        causal=select_rows(cm.causal)
        if isinstance(cm.causal, torch.Tensor)
        else cm.causal,
        encoder_seq_lens=select_rows(cm.encoder_seq_lens),
        encoder_seq_lens_cpu=select_rows(cm.encoder_seq_lens_cpu),
        dcp_local_seq_lens=select_rows(cm.dcp_local_seq_lens),
        dcp_local_seq_lens_cpu_upper_bound=select_rows(
            cm.dcp_local_seq_lens_cpu_upper_bound
        ),
        is_prefilling=select_rows(cm.is_prefilling),
        req_idx=seg.seg_req_idx.copy()
        if cm.req_idx is None
        else cm.req_idx[seg.seg_req_idx],
        mm_req_doc_ranges=mm_req_doc_ranges,
        rswa_prefix_lens=select_rows(cm.rswa_prefix_lens),
        replayssm_decode_base_cpu=select_rows(cm.replayssm_decode_base_cpu),
        hole_segments=seg,
    )
