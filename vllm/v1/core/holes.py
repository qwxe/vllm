# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hole-aware prefix caching (holes/core).

A *hole* is a range of a request's prompt whose KV blocks are missing from the
prefix cache while blocks *after* it are still cached. Block hashes are
chained over token ids (not over KV presence), so every block hash of the
prompt is computable and the blocks after a miss can still be looked up.

With holes enabled, the KV cache manager returns a per-block hit vector, the
missing blocks before the last hit are allocated in place (the block table
stays position ordered), and the scheduler computes the holes as separate
segments of the request (front first, then the tail). The model runner turns
each segment into its own varlen entry of the attention metadata
(KERNEL-ALGO.md variant (a)), so the stock attention backends run unchanged.

Everything here is off by default. Enable with
``additional_config={"holes": {"enabled": true, ...}}`` or ``VLLM_HOLES=1``.
"""

import json
import os
from dataclasses import dataclass, fields
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# A token range [start, end).
Range = tuple[int, int]


@dataclass(frozen=True)
class HoleConfig:
    enabled: bool = False
    """Reuse cached blocks after a missing range (hole-aware lookup)."""

    chunk_granule: int = 128
    """A cut inside a segment falls at segment_start + k * chunk_granule
    (segment ends are always allowed). KERNEL-ALGO 5.2."""

    min_piece: int = 256
    """q_min: never leave a remainder 0 < r < min_piece behind a cut, and
    never schedule a whole hole shorter than this as the last item of a step
    when other work of the request is in that step (it is deferred so that it
    shares the next step with the following segment)."""

    step_budget: int = 0
    """Per-request token cap per step while the request has holes
    (B of KERNEL-ALGO 5.2). 0 = the scheduler's max_num_batched_tokens."""

    min_reuse_blocks: int = 1
    """A run of cached blocks after a hole shorter than this is treated as a
    miss (recomputed) instead of being reused."""

    min_mixed_tail: int = 2
    """A tail chunk that shares a step with hole pieces must have at least
    this many tokens; a shorter tail runs alone in the next step. Keeps every
    segment of a multi-segment request above the decode threshold of
    backends that split decodes from prefills (FlashInfer: q <= 1)."""

    test_hooks: bool = False
    """Enable test-only helpers (e.g. Scheduler.holes_evict_token_range)."""

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "HoleConfig":
        if not d:
            return HoleConfig()
        names = {f.name for f in fields(HoleConfig)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown holes config keys: {sorted(unknown)}")
        cfg = HoleConfig(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        assert self.chunk_granule > 0
        assert self.min_piece >= 0
        assert self.step_budget >= 0
        assert self.min_reuse_blocks >= 1
        assert self.min_mixed_tail >= 1


def get_hole_config(vllm_config: Any) -> HoleConfig:
    """Read the hole config from ``additional_config["holes"]``; the env var
    ``VLLM_HOLES`` (``1`` or a JSON object) overrides it."""
    d: dict[str, Any] = {}
    add = getattr(vllm_config, "additional_config", None)
    if isinstance(add, dict) and isinstance(add.get("holes"), dict):
        d.update(add["holes"])
    env = os.environ.get("VLLM_HOLES")
    if env:
        if env.strip().startswith("{"):
            d.update(json.loads(env))
        else:
            d["enabled"] = env.strip().lower() in ("1", "true", "yes", "on")
    return HoleConfig.from_dict(d)


def range_tokens(ranges: list[Range] | None) -> int:
    return sum(e - s for s, e in ranges) if ranges else 0


def hits_to_holes(hit: list[bool], block_size: int) -> list[Range]:
    """Merge the missing blocks of a hit vector into token ranges."""
    holes: list[Range] = []
    i, n = 0, len(hit)
    while i < n:
        if hit[i]:
            i += 1
            continue
        j = i
        while j < n and not hit[j]:
            j += 1
        holes.append((i * block_size, j * block_size))
        i = j
    return holes


def apply_min_reuse(hit: list[bool], min_reuse_blocks: int) -> list[bool]:
    """Drop trailing misses, and turn hit runs after the first miss that are
    shorter than ``min_reuse_blocks`` into misses. The leading hit run (the
    stock prefix) is always kept."""
    hit = list(hit)
    while hit and not hit[-1]:
        hit.pop()
    if min_reuse_blocks > 1 and hit:
        first_miss = next((k for k, h in enumerate(hit) if not h), len(hit))
        i = first_miss
        while i < len(hit):
            if not hit[i]:
                i += 1
                continue
            j = i
            while j < len(hit) and hit[j]:
                j += 1
            if j - i < min_reuse_blocks:
                for k in range(i, j):
                    hit[k] = False
            i = j
        while hit and not hit[-1]:
            hit.pop()
    return hit


def subtract_ranges(pending: list[Range], done: list[Range]) -> list[Range]:
    """pending minus done; both sorted, disjoint."""
    out: list[Range] = []
    for s, e in pending:
        cur = s
        for ds, de in done:
            if de <= cur or ds >= e:
                continue
            if ds > cur:
                out.append((cur, ds))
            cur = max(cur, de)
        if cur < e:
            out.append((cur, e))
    return out


def plan_step(
    pending: list[Range],
    tail_start: int,
    tail_end: int,
    budget: int,
    hard_limit: int,
    cfg: HoleConfig,
    max_segments: int | None = None,
) -> tuple[list[Range], int]:
    """Choose what a request with holes computes in one step (KERNEL-ALGO 5.2).

    Segments are the pending hole ranges in position order (front first),
    then the tail [tail_start, tail_end). Take from them until ``budget``
    tokens: a cut inside a segment falls at segment_start + k * granule; a
    cut that would leave a remainder 0 < r < min_piece moves to the segment
    end when that exceeds the budget by < min_piece (and stays within
    ``hard_limit``), else back so that the remainder is >= min_piece.

    ``max_segments`` caps the number of segments (hole pieces + tail) of the
    request in this step: each is a varlen entry of the attention metadata,
    and backends size buffers by max_num_seqs.
    Work after the cap is left for later steps; the tail never overtakes a
    pending hole.

    Returns (hole pieces to compute, number of tail tokens).
    """
    G, Q = cfg.chunk_granule, cfg.min_piece
    segs: list[tuple[int, int, bool]] = [(s, e, False) for s, e in pending]
    if tail_end > tail_start:
        segs.append((tail_start, tail_end, True))

    taken: list[tuple[int, int, bool, bool]] = []  # (s, cut, is_tail, whole)
    left = budget
    used = 0
    for s, e, is_tail in segs:
        if left <= 0:
            break
        if max_segments is not None and len(taken) >= max_segments:
            break
        n = e - s
        if n <= left:
            cut = e
        else:
            cut = s + (left // G) * G
            r = e - cut
            if 0 < r < Q:
                if e - (s + left) < Q and used + n <= hard_limit:
                    cut = e
                else:
                    cut = s + max(0, (e - Q - s) // G) * G
            if cut <= s:
                break
        taken.append((s, cut, is_tail, cut == e))
        used += cut - s
        left -= cut - s
        if cut < e:
            break

    # A whole hole shorter than q_min is not left as the last item of a step
    # that already holds other work: it goes with the next segment next step.
    if (
        len(taken) > 1
        and not taken[-1][2]
        and taken[-1][3]
        and taken[-1][1] - taken[-1][0] < Q
        and len(taken) < len(segs)
    ):
        taken.pop()

    pieces = [(s, c) for s, c, is_tail, _ in taken if not is_tail]
    n_tail = sum(c - s for s, c, is_tail, _ in taken if is_tail)
    if pieces and 0 < n_tail < cfg.min_mixed_tail:
        n_tail = 0
    return pieces, n_tail
