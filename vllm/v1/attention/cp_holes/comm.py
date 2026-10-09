# SPDX-License-Identifier: Apache-2.0
"""Byte-packed collectives for the CP schemes.

Every payload is packed row-wise into uint8 so that one collective carries
tensors of different dtypes (bf16 O + fp32 LSE, Q + K + V) and so that gloo
(CPU tests) and NCCL (GPUs) see the same data movement. Split sizes are always
derived from the global `StepPlan` on every rank, so no size exchange is
needed.

All functions take a `torch.distributed` ProcessGroup (in vLLM, the CP
`GroupCoordinator.device_group`).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.distributed as dist


class _Work:
    """A collective handle that also pins its buffers until wait()."""

    def __init__(self, work, *keep):
        self.work, self.keep = work, keep

    def wait(self):
        if self.work is not None:
            self.work.wait()
        self.keep = ()


def wait_all(works) -> None:
    for w in works or ():
        if w is not None:
            w.wait()


def row_bytes(shape_tail: Sequence[int], dtype: torch.dtype) -> int:
    n = 1
    for s in shape_tail:
        n *= int(s)
    return n * torch.empty(0, dtype=dtype).element_size()


def pack_rows(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """[rows, ...] tensors (same rows) -> uint8 [rows, sum of row bytes]."""
    rows = tensors[0].shape[0]
    parts = []
    for t in tensors:
        assert t.shape[0] == rows
        per_row = 1
        for x in t.shape[1:]:
            per_row *= int(x)
        parts.append(t.contiguous().view(rows, per_row).view(torch.uint8))
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def unpack_rows(buf: torch.Tensor, specs: Sequence[tuple[Sequence[int],
                                                         torch.dtype]]
                ) -> list[torch.Tensor]:
    """Inverse of pack_rows. specs = [(shape_tail, dtype)].

    `buf` is uint8 [rows, row bytes] or flat uint8 (rows are inferred).
    """
    total = sum(row_bytes(t, d) for t, d in specs)
    if buf.dim() == 1:
        assert buf.numel() % total == 0, (buf.numel(), total)
        buf = buf.view(buf.numel() // total, total)
    rows = buf.shape[0]
    out, off = [], 0
    for tail, dt in specs:
        nb = row_bytes(tail, dt)
        chunk = buf[:, off:off + nb].contiguous()
        out.append(chunk.view(dt).view(rows, *tail))
        off += nb
    assert off == buf.shape[1], (off, buf.shape)
    return out


def all_to_all_bytes(send: Sequence[torch.Tensor],
                     recv_nbytes: Sequence[int],
                     group: dist.ProcessGroup,
                     async_op: bool = False):
    """Variable-size all-to-all of flat uint8 buffers.

    send[r] is the flat uint8 payload for rank r (may be empty). recv_nbytes[r]
    is the number of bytes this rank receives from rank r. Returns
    (list of received flat buffers per source rank, work handle or None).
    """
    dev = send[0].device
    if group is None:  # world size 1
        assert len(send) == 1 and send[0].numel() == recv_nbytes[0]
        return [send[0].reshape(-1)], None
    send_flat = torch.cat([s.reshape(-1) for s in send]) if send else \
        torch.empty(0, dtype=torch.uint8, device=dev)
    recv_flat = torch.empty(sum(recv_nbytes), dtype=torch.uint8, device=dev)
    work = dist.all_to_all_single(recv_flat,
                                  send_flat,
                                  output_split_sizes=list(recv_nbytes),
                                  input_split_sizes=[s.numel() for s in send],
                                  group=group,
                                  async_op=async_op)
    outs, off = [], 0
    for n in recv_nbytes:
        outs.append(recv_flat[off:off + n])
        off += n
    # Keep the send buffer alive until the caller waits on the work.
    return outs, _Work(work, send_flat)


def all_gather_bytes(local: torch.Tensor, nbytes: Sequence[int],
                     group: dist.ProcessGroup, async_op: bool = False):
    """Variable-size all-gather of a flat uint8 buffer (padded to the max)."""
    if group is None:
        return [local.reshape(-1)], None
    mx = max(nbytes) if nbytes else 0
    dev = local.device
    pad = torch.zeros(mx, dtype=torch.uint8, device=dev)
    pad[:local.numel()] = local.reshape(-1)
    outs = [torch.empty(mx, dtype=torch.uint8, device=dev) for _ in nbytes]
    work = dist.all_gather(outs, pad, group=group, async_op=async_op)
    return [o[:n] for o, n in zip(outs, nbytes)], _Work(work, pad)


def ring_exchange(send: torch.Tensor, recv_nbytes: int, rank: int,
                  world: int, group: dist.ProcessGroup,
                  global_ranks: Optional[Sequence[int]] = None):
    """Async send to rank+1 and receive from rank-1 (group-local ranks).

    Returns (recv buffer, list of works). The caller waits on the works before
    reading the buffer.
    """
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    if global_ranks is not None:
        nxt, prv = global_ranks[nxt], global_ranks[prv]
    else:
        nxt = dist.get_global_rank(group, nxt)
        prv = dist.get_global_rank(group, prv)
    recv = torch.empty(recv_nbytes, dtype=torch.uint8, device=send.device)
    ops = []
    if send.numel() > 0:
        ops.append(dist.P2POp(dist.isend, send.reshape(-1), nxt, group=group))
    if recv_nbytes > 0:
        ops.append(dist.P2POp(dist.irecv, recv, prv, group=group))
    works = dist.batch_isend_irecv(ops) if ops else []
    return recv, [_Work(w, send) for w in works]
