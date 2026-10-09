# SPDX-License-Identifier: Apache-2.0
"""Per-rank KV storage seen by the CP schemes.

The schemes only need two operations on the local KV of one layer:

- `write(keys)`: store new K/V at explicit positions (vLLM: `reshape_and_cache`
  driven by a `slot_mapping` computed from the same positions);
- `gather(reqs, limits)`: the local keys of the given step-local requests with
  position < limit (causal truncation at E_step), ordered by request (in the
  order given) and then by position.

`RefKVStore` is a dense reference used by the tests. `PagedKVStore` adapts a
vLLM paged cache (one layer's `[2, num_blocks, block_size, Hkv, D]` tensor plus
per-request local block tables and local position maps).
"""

from __future__ import annotations

from typing import Protocol, Sequence

import torch

from .local_attn import KeySet


class CPKVStore(Protocol):

    def write(self, keys: KeySet) -> None:
        ...

    def gather(self, reqs: Sequence[int], limits: Sequence[int]) -> KeySet:
        ...


class RefKVStore:
    """Dense per-request storage: positions, K and V per request."""

    def __init__(self, hkv: int, d: int, dtype: torch.dtype,
                 device: torch.device):
        self.hkv, self.d, self.dtype, self.device = hkv, d, dtype, device
        self.data: dict[int, KeySet] = {}

    def write(self, keys: KeySet) -> None:
        if keys.n == 0:
            return
        for r in torch.unique(keys.kreq).tolist():
            sel = (keys.kreq == r).nonzero(as_tuple=True)[0]
            new = keys.select(sel)
            old = self.data.get(r)
            if old is not None:
                assert not bool(torch.isin(new.kpos, old.kpos).any()), \
                    f"position written twice for req {r}"
                new = KeySet.cat([old, new])
            order = torch.argsort(new.kpos)
            self.data[r] = new.select(order)

    def gather(self, reqs: Sequence[int], limits: Sequence[int]) -> KeySet:
        parts = []
        for r, lim in zip(reqs, limits):
            ks = self.data.get(r)
            if ks is None:
                continue
            parts.append(ks.select((ks.kpos < lim).nonzero(as_tuple=True)[0]))
        if not parts:
            return KeySet.empty(self.hkv, self.d, self.dtype, self.device)
        return KeySet.cat(parts)


class PagedKVStore:
    """Adapter over one layer of a vLLM paged KV cache.

    kv_cache: [2, num_blocks, block_size, Hkv_local, D] (FlashAttention
    layout). For request i of the step, `block_table[i]` lists the local block
    ids in local order and `local_pos[i]` maps local slot index -> absolute
    position (-1 for an empty slot), and `present[i]` (bool, same shape) marks
    slots whose KV is valid: resident slots at step start; hole slots that are
    allocated in place but not computed yet are False until written. Under interleaved placement with
    u = block_size = cp_kv_cache_interleave_size, local slot j of a request
    holds position (j // u) * u * d + rank * u + j % u, exactly what vLLM's
    `_compute_slot_mapping_kernel` writes.

    gather() copies the selected slots out (the reference path). A paged
    kernel (branch holes/kernel) reads the cache in place instead, with the
    same block table and a per-block start position (`kblk_pos`).
    """

    def __init__(self, kv_cache: torch.Tensor,
                 block_table: Sequence[torch.Tensor],
                 local_pos: Sequence[torch.Tensor],
                 present: Sequence[torch.Tensor]):
        self.kv_cache = kv_cache
        self.block_table = block_table
        self.local_pos = local_pos
        self.present = [p.clone() for p in present]
        self.block_size = kv_cache.shape[2]

    def _slots(self, r: int, pos: torch.Tensor) -> torch.Tensor:
        lp = self.local_pos[r]
        # local_pos is ascending over its valid entries.
        valid = (lp >= 0).nonzero(as_tuple=True)[0]
        idx = torch.searchsorted(lp[valid], pos).clamp(max=valid.numel() - 1)
        j = valid[idx]
        assert bool((lp[j] == pos).all()), "position has no local slot"
        return j

    def _slot_ids(self, r: int, j: torch.Tensor) -> torch.Tensor:
        bt = self.block_table[r]
        return bt[j // self.block_size] * self.block_size + j % self.block_size

    def write(self, keys: KeySet) -> None:
        flat = self.kv_cache.flatten(1, 2)  # [2, slots, Hkv, D]
        for r in torch.unique(keys.kreq).tolist():
            sel = (keys.kreq == r).nonzero(as_tuple=True)[0]
            j = self._slots(r, keys.kpos[sel])
            self.present[r][j] = True
            slots = self._slot_ids(r, j)
            flat[0, slots] = keys.k[sel].to(flat.dtype)
            flat[1, slots] = keys.v[sel].to(flat.dtype)

    def gather(self, reqs: Sequence[int], limits: Sequence[int]) -> KeySet:
        flat = self.kv_cache.flatten(1, 2)
        parts = []
        for r, lim in zip(reqs, limits):
            lp = self.local_pos[r]
            j = (self.present[r] & (lp >= 0) & (lp < lim)).nonzero(
                as_tuple=True)[0]
            slots = self._slot_ids(r, j)
            parts.append(KeySet(flat[0, slots], flat[1, slots], lp[j],
                                torch.full_like(lp[j], r)))
        if not parts:
            return KeySet.empty(flat.shape[2], flat.shape[3], flat.dtype,
                                flat.device)
        return KeySet.cat(parts)


def vllm_local_layout(block_table_row: torch.Tensor, num_blocks: int,
                      block_size: int, rank: int, world: int,
                      interleave: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Local slots of one request under vLLM's CP KV placement.

    vLLM (v1/worker/block_table.py `_compute_slot_mapping_kernel`) maps
    position p of a request to virtual block b = p // (block_size * world);
    with vbo = p - b * block_size * world, p is local to rank
    (vbo // interleave) % world, at local offset
    (vbo // (world * interleave)) * interleave + vbo % interleave of the
    rank's physical block block_table_row[b]. Every rank uses the same logical
    block ids. This returns (block ids in local order, absolute position of
    every local slot) for a PagedKVStore. interleave = block_size = u gives the
    u-token interleave of KERNEL-ALGO 4.0 (u = 128 needs block_size 128).
    world = 1 gives the plain (and the Ulysses head-sharded) layout.
    """
    assert block_size % interleave == 0
    o = torch.arange(block_size, dtype=torch.int64)
    vbo = (o // interleave) * world * interleave + rank * interleave + \
        o % interleave
    b = torch.arange(num_blocks, dtype=torch.int64)
    pos = (b.unsqueeze(1) * block_size * world + vbo.unsqueeze(0)).reshape(-1)
    return block_table_row[:num_blocks].to(torch.int64), pos


def vllm_slot_mapping(block_table_row: torch.Tensor, pos: torch.Tensor,
                      block_size: int, rank: int, world: int,
                      interleave: int) -> torch.Tensor:
    """Torch port of vLLM's CP slot mapping (-1 for positions not local)."""
    vbs = block_size * world
    b = pos // vbs
    vbo = pos - b * vbs
    local = (vbo // interleave) % world == rank
    off = (vbo // (world * interleave)) * interleave + vbo % interleave
    slot = block_table_row.to(torch.int64)[b] * block_size + off
    return torch.where(local, slot, torch.full_like(slot, -1))
