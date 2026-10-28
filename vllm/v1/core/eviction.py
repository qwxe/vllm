# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pluggable eviction of cached GPU KV blocks (holes/rate).

Stock vLLM evicts cached blocks with ``ref_cnt == 0`` in the order of
``FreeKVCacheBlockQueue`` (LRU, tail of a request first). That stays the
default: with ``policy == "lru"`` no policy object is installed and every
``BlockPool`` code path is the stock one.

``policy == "rate"`` is the session-state ``rate`` policy of the paper
(PAPER-ALGORITHM.md section 2; the simulator's ``SessRatePool``), applied to
the GPU prefix cache only (no lower tier, no prefetch / offload)::

    score(chunk) = rate(owner) * c(chunk position)       lowest evicted first
    rate         = p_return / max(E[time to next use], floor)

* A chunk is ``chunk_tokens`` (default 256) tokens = ``chunk_tokens //
  block_size`` consecutive blocks of one request's block table. The unit of
  eviction is a *group*: the free cached blocks of one chunk position whose
  last holder (the owner) is the same request. A whole group is evicted at
  once, so the holes it leaves are chunk aligned and >= holes/core's
  ``min_piece``; the blocks it frees beyond what the allocation needs stay
  free (without a hash) and are used first by the next allocations.
* Free blocks without a hash are always used before any cached block.
* The owner is the last request that held the block (set when the request
  frees it). Owner states (``SessionTracker``, fed by the scheduler):

  ======================================================  ==================
  owner prefill not finished                              protected
  session's next request arrived, not through prefill     protected
  session's next request already through prefill          0 (dead)
  owner finished prefill but not finished (preempted)     protected
                                                          (``running_owner=
                                                          "protect"``), or the
                                                          simulator's decode
                                                          rule: p = 1 - p_end,
                                                          E = MRL_dec(e) +
                                                          mean tool gap
  idle t s after the owner finished                       p = qS(t)/(qS(t)+
                                                          p_end), E = MRL_tool(t)
  ======================================================  ==================

  "protected" is the simulator's rate of 1e6, so protected chunks are still
  evictable as a last resort, front first.
* ``c(k)`` is the recompute seconds of chunk ``k`` at its position, from the
  prefill work fit ``S(L, H) = A + b1 (L - H) + b2 (L^2 - H^2) / 2``.
* Victim selection is a lazy min-heap over groups. Entries carry the heap
  epoch and the owner's version; an owner whose state changes (arrival of its
  successor, the successor's prefill end, its own finish) gets its version
  bumped and its groups re-pushed with fresh scores. The whole heap is
  rebuilt at most once per ``rescore_s`` (idle scores drift with time).

Sessions: a request belongs to session ``sampling_params.extra_args
["session_id"]``; a request without one is a session of its own.
"""

from __future__ import annotations

import heapq
import json
import os
import time
from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)

# Rate of a protected owner (the simulator's _BIG).
PROTECTED_RATE = 1e6

# The simulator's prefill work fit (oracle/ocs.py), seconds.
PAPER_COST_A = 1.74645466e-02
PAPER_COST_B1 = 3.87150522e-05
PAPER_COST_B2 = 1.85005256e-09

POLICIES = ("lru", "rate")


@dataclass(frozen=True)
class EvictionConfig:
    policy: str = "lru"
    """"lru" (stock vLLM) or "rate"."""

    params: Any = None
    """rate: path to a params JSON (``tool_gaps``, ``decode_durs`` in seconds,
    ``p_end``, optional ``floor``), or the dict itself."""

    chunk_tokens: int = 256
    """Tokens per eviction chunk; rounded down to a multiple of the block size
    (at least one block)."""

    rescore_s: float = 1.0
    """The heap is rebuilt at most this often (0 = at every eviction)."""

    cost_b1: float = PAPER_COST_B1
    """Recompute seconds per token (linear term of the prefill fit)."""

    cost_b2: float = PAPER_COST_B2
    """Recompute seconds per token per position (attention term)."""

    running_owner: str = "protect"
    """Owner past its prefill but not finished (i.e. preempted): "protect" or
    "decode" (the simulator's decode rule)."""

    time_scale: float = 1.0
    """Multiplies the params' time samples (tool gaps, decode durations) and
    the floor; for time-compressed experiments."""

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "EvictionConfig":
        if not d:
            return EvictionConfig()
        names = {f.name for f in fields(EvictionConfig)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"Unknown eviction config keys: {sorted(unknown)}")
        cfg = EvictionConfig(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        assert self.policy in POLICIES, f"eviction policy {self.policy!r}"
        assert self.chunk_tokens > 0
        assert self.rescore_s >= 0
        assert self.cost_b1 >= 0 and self.cost_b2 >= 0
        assert self.running_owner in ("protect", "decode")
        assert self.time_scale > 0
        if self.policy == "rate":
            assert self.params is not None, "rate eviction needs params"


def get_eviction_config(vllm_config: Any) -> EvictionConfig:
    """``additional_config["eviction"]``; the env var ``VLLM_EVICTION`` (a
    policy name or a JSON object) overrides it."""
    d: dict[str, Any] = {}
    add = getattr(vllm_config, "additional_config", None)
    if isinstance(add, dict) and isinstance(add.get("eviction"), dict):
        d.update(add["eviction"])
    env = os.environ.get("VLLM_EVICTION")
    if env:
        if env.strip().startswith("{"):
            d.update(json.loads(env))
        else:
            d["policy"] = env.strip()
    if "params" not in d and os.environ.get("VLLM_EVICTION_PARAMS"):
        d["params"] = os.environ["VLLM_EVICTION_PARAMS"]
    return EvictionConfig.from_dict(d)


# ----------------------------------------------------------------------------
# Static inputs: empirical samples and the recompute cost
# ----------------------------------------------------------------------------


class EmpiricalSample:
    """Empirical sample with survival and mean residual life (the simulator's
    ``_Sample``)."""

    def __init__(self, xs: Iterable[float]):
        self.x = sorted(float(v) for v in xs)
        n = len(self.x)
        if n == 0:
            raise ValueError("empty sample")
        self.suf = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            self.suf[i] = self.suf[i + 1] + self.x[i]
        self.mean = self.suf[0] / n

    def surv(self, t: float) -> float:
        return 1.0 - bisect_right(self.x, t) / len(self.x)

    def mrl(self, t: float) -> float:
        i = bisect_right(self.x, t)
        n = len(self.x) - i
        return self.suf[i] / n - t if n > 0 else 0.0


@dataclass
class RateParams:
    tool: EmpiricalSample
    dec: EmpiricalSample
    p_end: float
    floor: float = 1.0

    @staticmethod
    def load(src: Any, time_scale: float = 1.0) -> "RateParams":
        if isinstance(src, (str, os.PathLike)):
            with open(src) as f:
                d = json.load(f)
        else:
            d = dict(src)
        k = float(time_scale)
        return RateParams(
            tool=EmpiricalSample(k * float(v) for v in d["tool_gaps"]),
            dec=EmpiricalSample(k * float(v) for v in d["decode_durs"]),
            p_end=float(d["p_end"]),
            floor=k * float(d.get("floor", 1.0)),
        )


def chunk_recompute_s(k: int, chunk_tokens: int, b1: float, b2: float) -> float:
    """Seconds to recompute tokens [k C, (k+1) C) (the simulator's _chunk_s)."""
    s, e = k * chunk_tokens, (k + 1) * chunk_tokens
    return b1 * chunk_tokens + b2 * (e * e - s * s) / 2.0


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------


@dataclass
class _ReqState:
    session: str
    arrival: float
    prefill_done: float | None = None
    done: float | None = None
    prev: str | None = None
    next: str | None = None


@dataclass
class SessionTracker:
    """Observable request / session state, fed by the scheduler.

    ``on_change(rid)`` is called for every request whose chunks may change
    score because of an event.
    """

    on_change: Callable[[str], None] | None = None
    reqs: dict[str, _ReqState] = field(default_factory=dict)
    last: dict[str, str] = field(default_factory=dict)

    def _bump(self, rid: str | None) -> None:
        if rid is not None and self.on_change is not None:
            self.on_change(rid)

    def arrived(self, rid: str, session: str | None, t: float) -> None:
        s = session if session is not None else f"\x00req:{rid}"
        r = _ReqState(session=s, arrival=t)
        prev = self.last.get(s)
        if prev is not None and prev in self.reqs:
            p = self.reqs[prev]
            if p.next is None:
                p.next = rid
                r.prev = prev
        self.reqs[rid] = r
        self.last[s] = rid
        self._bump(r.prev)

    def prefill_done(self, rid: str, t: float) -> None:
        r = self.reqs.get(rid)
        if r is None or r.prefill_done is not None:
            return
        r.prefill_done = t
        self._bump(rid)
        self._bump(r.prev)

    def finished(self, rid: str, t: float) -> None:
        r = self.reqs.get(rid)
        if r is None or r.done is not None:
            return
        r.done = t
        self._bump(rid)
        if r.prefill_done is None:
            # Aborted before its prefill: the predecessor is no longer queued.
            self._bump(r.prev)

    def forget(self, rid: str) -> str | None:
        """Drop a finished request whose record nobody reads any more: not
        its session's latest request, and no predecessor left (a predecessor
        reads its successor's state). Returns the successor, which may have
        become forgettable, if the record was dropped."""
        r = self.reqs.get(rid)
        if r is None or r.done is None or self.last.get(r.session) == rid:
            return None
        if r.prev is not None and r.prev in self.reqs:
            return None
        del self.reqs[rid]
        if r.next is not None and r.next in self.reqs:
            self.reqs[r.next].prev = None
            return r.next
        return None


# ----------------------------------------------------------------------------
# The policy
# ----------------------------------------------------------------------------


class RateEvictionPolicy:
    """Chooses which cached free blocks a ``BlockPool`` evicts.

    The pool calls the hooks below (only when a policy is installed):
    ``on_free`` (a block entered the free queue), ``on_remove`` (left it:
    touched or allocated), ``on_uncache`` (a free block lost its hash),
    ``take_free`` (allocate n blocks), ``reset``. The KV cache manager calls
    ``on_request_free`` with a request's block table before freeing it.
    """

    def __init__(
        self,
        cfg: EvictionConfig,
        block_size: int,
        clock: Callable[[], float] = time.monotonic,
        params: RateParams | None = None,
    ):
        self.cfg = cfg
        self.block_size = block_size
        self.blocks_per_chunk = max(1, cfg.chunk_tokens // block_size)
        self.chunk_tokens = self.blocks_per_chunk * block_size
        self.params = params or RateParams.load(cfg.params, cfg.time_scale)
        self.clock = clock
        self.sessions = SessionTracker(on_change=self._owner_changed)
        self.pool: BlockPool | None = None
        self._cost_cache: dict[int, float] = {}
        # Stats: evicted blocks / groups by owner state.
        self.evicted_blocks: dict[str, int] = {}
        self.evicted_groups = 0
        self._reset_state()

    # ------------------------------------------------------------ state
    def _reset_state(self) -> None:
        # block_id -> (owner request id, block index in its block table)
        self._meta: dict[int, tuple[str | None, int]] = {}
        # Free blocks without a hash, in the order they became so.
        self._uncached: dict[int, KVCacheBlock] = {}
        # (owner, chunk) -> {block_id: block}: cached free blocks.
        self._groups: dict[tuple[str | None, int], dict[int, KVCacheBlock]] = {}
        self._block_group: dict[int, tuple[str | None, int]] = {}
        self._owner_chunks: dict[str | None, set[int]] = {}
        self._owner_ver: dict[str | None, int] = {}
        self._heap: list[tuple] = []
        self._epoch = 0
        self._built_at: float | None = None
        self._seq = 0

    def attach(self, pool: BlockPool) -> None:
        self.pool = pool
        self._reset_state()
        for blk in pool.free_block_queue.get_all_free_blocks():
            self.on_free(blk)

    def reset(self) -> None:
        """The pool's prefix cache was reset: every free block is uncached."""
        assert self.pool is not None
        self.attach(self.pool)

    # ------------------------------------------------------------ scoring
    def chunk_cost(self, k: int) -> float:
        c = self._cost_cache.get(k)
        if c is None:
            c = self._cost_cache[k] = chunk_recompute_s(
                k, self.chunk_tokens, self.cfg.cost_b1, self.cfg.cost_b2
            )
        return c

    def owner_rate(self, owner: str | None, now: float) -> tuple[str, float]:
        """(state, rate) of the chunks last held by request ``owner``."""
        prm = self.params
        if owner is None:
            return "none", 1.0 / max(prm.tool.mean, prm.floor)
        r = self.sessions.reqs.get(owner)
        if r is None:
            return "none", 1.0 / max(prm.tool.mean, prm.floor)
        if r.prefill_done is None and r.done is None:
            return "prefill", PROTECTED_RATE
        nx = self.sessions.reqs.get(r.next) if r.next is not None else None
        if nx is not None and not (nx.done is not None and nx.prefill_done is None):
            if nx.prefill_done is not None:
                return "dead", 0.0
            return "queued", PROTECTED_RATE
        q = 1.0 - prm.p_end
        if r.done is None:
            if self.cfg.running_owner == "protect":
                return "running", PROTECTED_RATE
            e = max(0.0, now - (r.prefill_done or now))
            return "decode", q / max(prm.dec.mrl(e) + prm.tool.mean, prm.floor)
        t = max(0.0, now - r.done)
        st = q * prm.tool.surv(t)
        p = st / max(st + prm.p_end, 1e-12)
        return "idle", p / max(prm.tool.mrl(t), prm.floor)

    def _entry(self, key: tuple[str | None, int], now: float, cache: dict) -> tuple:
        owner, k = key
        sr = cache.get(owner)
        if sr is None:
            sr = cache[owner] = self.owner_rate(owner, now)
        self._seq += 1
        return (
            sr[1] * self.chunk_cost(k),
            k,
            self._seq,
            key,
            self._epoch,
            self._owner_ver.get(owner, 0),
            sr[0],
        )

    def _push(self, key: tuple[str | None, int], now: float | None = None) -> None:
        heapq.heappush(
            self._heap, self._entry(key, self.clock() if now is None else now, {})
        )

    def _rebuild(self, now: float) -> None:
        self._epoch += 1
        cache: dict = {}
        self._heap = [self._entry(key, now, cache) for key, g in self._groups.items() if g]
        heapq.heapify(self._heap)
        self._built_at = now

    def _owner_changed(self, owner: str) -> None:
        self._owner_ver[owner] = self._owner_ver.get(owner, 0) + 1
        chunks = self._owner_chunks.get(owner)
        if not chunks:
            return
        now = self.clock()
        cache: dict = {}
        for k in chunks:
            heapq.heappush(self._heap, self._entry((owner, k), now, cache))

    # ------------------------------------------------------------ session events
    def on_arrival(self, request_id: str, session_id: str | None) -> None:
        self.sessions.arrived(request_id, session_id, self.clock())
        self._maybe_forget(self.sessions.reqs[request_id].prev)

    def on_prefill_done(self, request_id: str) -> None:
        self.sessions.prefill_done(request_id, self.clock())

    def on_finished(self, request_id: str) -> None:
        self.sessions.finished(request_id, self.clock())
        if not self._owner_chunks.get(request_id):
            self._maybe_forget(request_id)

    def _maybe_forget(self, rid: str | None) -> None:
        """Drop session records that no cached chunk depends on any more."""
        while rid is not None:
            if self._owner_chunks.get(rid):
                return
            r = self.sessions.reqs.get(rid)
            if r is None or r.done is None:
                return
            nxt = self.sessions.forget(rid)
            if rid in self.sessions.reqs:
                return
            self._owner_ver.pop(rid, None)
            rid = nxt

    # ------------------------------------------------------------ pool hooks
    def on_request_free(self, request_id: str, blocks: Sequence[KVCacheBlock]) -> None:
        """``blocks`` is the request's block table (position order), about to
        be freed: the request becomes the owner of every block in it."""
        meta = self._meta
        for i, blk in enumerate(blocks):
            if not blk.is_null:
                meta[blk.block_id] = (request_id, i)

    def on_free(self, blk: KVCacheBlock) -> None:
        if blk.block_hash is None:
            self._uncached[blk.block_id] = blk
            return
        owner, idx = self._meta.get(blk.block_id, (None, 0))
        key = (owner, idx // self.blocks_per_chunk)
        g = self._groups.get(key)
        if g is None:
            g = self._groups[key] = {}
            self._owner_chunks.setdefault(owner, set()).add(key[1])
        if not g:
            self._push(key)
        g[blk.block_id] = blk
        self._block_group[blk.block_id] = key

    def on_remove(self, blk: KVCacheBlock) -> None:
        bid = blk.block_id
        if self._uncached.pop(bid, None) is not None:
            return
        key = self._block_group.pop(bid, None)
        if key is None:
            return
        g = self._groups[key]
        del g[bid]
        if not g:
            del self._groups[key]
            owner = key[0]
            chunks = self._owner_chunks.get(owner)
            if chunks is not None:
                chunks.discard(key[1])
                if not chunks:
                    del self._owner_chunks[owner]
                    self._maybe_forget(owner)

    def on_uncache(self, blk: KVCacheBlock) -> None:
        """A free block lost its hash (evicted from the prefix cache)."""
        if blk.block_id in self._block_group:
            self.on_remove(blk)
            self._uncached[blk.block_id] = blk

    def take_free(self, n: int) -> list[KVCacheBlock]:
        """Remove ``n`` blocks from the free queue for allocation, uncached
        blocks first, then by evicting the lowest-score groups."""
        assert self.pool is not None
        if len(self._uncached) < n:
            now = self.clock()
            if self._built_at is None or now - self._built_at >= self.cfg.rescore_s:
                self._rebuild(now)
            while len(self._uncached) < n:
                if not self._evict_one():
                    break
        if len(self._uncached) < n:
            raise ValueError(f"Cannot get {n} free blocks from the pool")
        queue = self.pool.free_block_queue
        out = []
        it = iter(self._uncached)
        ids = [next(it) for _ in range(n)]
        for bid in ids:
            blk = self._uncached.pop(bid)
            queue.remove(blk)
            out.append(blk)
        return out

    def _evict_one(self) -> bool:
        """Evict the lowest-score valid group into the uncached set."""
        assert self.pool is not None
        for attempt in range(2):
            heap = self._heap
            while heap:
                _, _, _, key, epoch, over, state = heapq.heappop(heap)
                if epoch != self._epoch or over != self._owner_ver.get(key[0], 0):
                    continue
                g = self._groups.get(key)
                if not g:
                    continue
                blocks = sorted(g.values(), key=lambda b: self._meta[b.block_id][1]
                                if b.block_id in self._meta else 0)
                for blk in blocks:
                    # Removes the hash (KV events, metrics) and, through
                    # on_uncache, moves the block to the uncached set.
                    self.pool._maybe_evict_cached_block(blk)
                    if blk.block_id in self._block_group:
                        # The hash was not in the map under this block (it
                        # should be): drop it anyway, a reused block must
                        # carry no hash.
                        blk.reset_hash()
                        self.on_uncache(blk)
                self.evicted_blocks[state] = self.evicted_blocks.get(state, 0) + len(blocks)
                self.evicted_groups += 1
                return True
            if not self._groups:
                return False
            # Every non-empty group always has a valid entry; rebuild if not.
            logger.warning_once("rate eviction: heap out of sync, rebuilding")
            self._rebuild(self.clock())
        return False

    # ------------------------------------------------------------ introspection
    def candidate_order(self) -> list[tuple[str | None, int, float, str]]:
        """Current groups in eviction order, scored now (for tests / debug):
        (owner, chunk, score, state)."""
        now = self.clock()
        cache: dict = {}
        rows = []
        for key, g in self._groups.items():
            if g:
                e = self._entry(key, now, cache)
                rows.append((e[0], e[1], e[2], key, e[6]))
        rows.sort(key=lambda r: r[:3])
        return [(r[3][0], r[3][1], r[0], r[4]) for r in rows]

    def stats(self) -> dict[str, Any]:
        return {
            "evicted_blocks": dict(self.evicted_blocks),
            "evicted_groups": self.evicted_groups,
            "free_uncached": len(self._uncached),
            "free_cached_groups": len(self._groups),
            "tracked_requests": len(self.sessions.reqs),
        }


def make_eviction_policy(
    cfg: EvictionConfig, block_size: int, clock: Callable[[], float] | None = None
) -> RateEvictionPolicy | None:
    if cfg.policy == "lru":
        return None
    if cfg.policy == "rate":
        if clock is None:
            return RateEvictionPolicy(cfg, block_size)
        return RateEvictionPolicy(cfg, block_size, clock=clock)
    raise ValueError(cfg.policy)


def session_id_of(request: Any) -> str | None:
    sid = getattr(request, "session_id", None)
    if sid is not None:
        return str(sid)
    sp = getattr(request, "sampling_params", None)
    extra = getattr(sp, "extra_args", None) if sp is not None else None
    if isinstance(extra, dict):
        sid = extra.get("session_id")
        if sid is not None:
            return str(sid)
    return None


__all__ = [
    "EvictionConfig",
    "RateEvictionPolicy",
    "RateParams",
    "EmpiricalSample",
    "SessionTracker",
    "chunk_recompute_s",
    "get_eviction_config",
    "make_eviction_policy",
    "session_id_of",
    "PROTECTED_RATE",
]
