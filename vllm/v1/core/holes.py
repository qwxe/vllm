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
from dataclasses import dataclass, field, fields
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


# ---- hybrid (Full + Mamba/SWA) replay planner (M note section 4) ---------


@dataclass(frozen=True)
class HybridResidency:
    """What the hybrid lookup found for one request (planner input).

    ``full_missing``: per Full attention group batch, token ranges whose page
    is not retained (after ``min_reuse_blocks``). ``state_groups``: per
    Mamba/SWA group batch, the boundaries (> 0) at which that group has its
    exact state. ``frontiers``: candidate reuse boundaries (all group block
    sizes divide them; 0 included).
    """

    full_missing: list[list[Range]]
    state_groups: list[set[int]]
    frontiers: list[int]


@dataclass(frozen=True)
class HybridReplayPlan:
    """Request-local plan of a hybrid request with a reuse frontier.

    ``spans`` are the ordered repair intervals executed with the whole model
    before the tail; ``span_restore[i]`` says the state of every state group
    is restored from a retained source at ``spans[i][0]`` (False when the span
    starts at 0 with the model's initial state, or continues the previous
    span). The tail starts at ``frontier``; ``tail_restore`` is False when the
    last span ends exactly there (the running state simply continues).
    """

    frontier: int
    spans: tuple[Range, ...]
    span_restore: tuple[bool, ...]
    tail_restore: bool

    def executed_tokens(self, num_tokens: int) -> int:
        return range_tokens(list(self.spans)) + num_tokens - self.frontier


@dataclass
class HybridLookup:
    """Result of the hybrid lookup of one request (holes/M)."""

    plan: HybridReplayPlan
    blocks: Any  # KVCacheBlocks at the plan frontier, null at Full hole pages
    num_local_cached_tokens: int
    num_hole_pages: int
    # gid -> token ranges of Full pages to load from L2 (stage 4); null blocks
    # in ``blocks`` that are loads, not holes
    load_ranges: dict[int, list[Range]] = field(default_factory=dict)


def hybrid_completed_prefix(
    plan: "HybridReplayPlan", completed: list[Range]
) -> int:
    """holes/M (stage 4, store cap): the end of the contiguous prefix of the
    prompt whose KV is valid while a hybrid plan runs: everything below the
    first span (retained), each span only as far as its completed replay
    reaches, and the retained gap after a fully completed span."""
    if not plan.spans:
        return plan.frontier
    cap = plan.spans[0][0]
    for s, e in plan.spans:
        cap = max(cap, s)  # the retained gap after a fully completed span
        done = s
        advanced = True
        while advanced and done < e:
            advanced = False
            for cs, ce in completed:
                if cs <= done < ce:
                    done = ce
                    advanced = True
        cap = min(done, e)
        if cap < e:
            return cap
    return plan.frontier


def _close_spans(
    full_missing: list[Range], state_ok: Any, frontier: int
) -> list[Range]:
    """Dependency closure of the Full write obligations below ``frontier``:
    a span can only start where every state group has an exact retained
    state (or at 0); otherwise it extends back to the nearest such boundary.
    Overlapping and adjacent spans merge (the running state continues)."""
    need = sorted(
        (s, min(e, frontier))
        for rs in full_missing
        for s, e in rs
        if s < frontier
    )
    spans: list[Range] = []
    for s, e in need:
        start = s
        while start > 0 and not state_ok(start):
            start -= 1
        if spans and start <= spans[-1][1]:
            ps, pe = spans[-1]
            spans[-1] = (ps, max(pe, e))
        else:
            spans.append((start, e))
    # extending a start back can overlap an earlier span: merge until stable
    merged: list[Range] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            ps, pe = merged[-1]
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def plan_hybrid_replay(
    num_tokens: int,
    full_missing: list[list[Range]],
    state_groups: list[set[int]],
    frontiers: list[int],
) -> HybridReplayPlan | None:
    """Choose the furthest legal reuse frontier whose whole-model repair plan
    executes fewer tokens than the stock prefix hit (M section 4, steps 3-6).

    ``full_missing``: per Full group, the token ranges whose KV is not
    retained (GPU or L2). ``state_groups``: per Mamba/SWA group, the set of
    boundaries at which that group has an exact retained state. ``frontiers``:
    candidate reuse boundaries (0 is always usable). A frontier is legal if
    every state group has its state exactly there and it leaves the final
    token to compute. Returns None when the stock prefix path is at least as
    good.
    """

    def state_ok(c: int) -> bool:
        return c == 0 or all(c in g for g in state_groups)

    cands = sorted(
        {c for c in frontiers if 0 <= c < num_tokens and state_ok(c)} | {0}
    )
    missing_starts = [s for rs in full_missing for s, _ in rs]
    first_hole = min(missing_starts, default=None)
    # stock hit: furthest legal frontier with no Full hole below it
    stock_c = max(
        (c for c in cands if first_hole is None or c <= first_hole), default=0
    )
    stock_cost = num_tokens - stock_c
    best: HybridReplayPlan | None = None
    for c in reversed(cands):  # furthest first
        if c <= stock_c:
            break
        spans = _close_spans(full_missing, state_ok, c)
        restore = tuple(s > 0 and not (i > 0 and s == spans[i - 1][1])
                        for i, (s, _) in enumerate(spans))
        plan = HybridReplayPlan(
            frontier=c,
            spans=tuple(spans),
            span_restore=restore,
            tail_restore=not (spans and spans[-1][1] == c),
        )
        if plan.executed_tokens(num_tokens) < stock_cost:
            best = plan
            break
    return best


def hybrid_group_kinds(managers: list[Any]) -> tuple[list[str] | None, str | None]:
    """Classify resolved KV cache groups by their manager type (never by
    group id or model name). Supported (M section 7): ordinary Full, ordinary
    sliding window and Mamba in align mode, with at least one Full group.
    Returns ``(kinds, None)`` or ``(None, reason)``."""
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
        SlidingWindowManager,
    )

    kinds: list[str] = []
    for gid, m in enumerate(managers):
        t = type(m)
        if t is FullAttentionManager:
            kinds.append("full")
        elif t is SlidingWindowManager:
            kinds.append("swa")
        elif t is MambaManager:
            if getattr(m, "mamba_cache_mode", None) != "align":
                return None, f"group {gid}: Mamba needs mamba_cache_mode='align'."
            kinds.append("mamba")
        else:
            return None, f"group {gid}: unsupported manager {t.__name__}."
    if "full" not in kinds:
        return None, "no Full attention group (hybrid holes need one)."
    return kinds, None
