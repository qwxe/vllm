# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/rate at L2: the session-rate CachePolicy of the CPU offload tier.

Design: ``notes/design/T20261028-fork-l2-rate.md`` r5 sections 4-7 (the
simulator's marconi-base ``rate`` policy applied to the host pool). The policy
keeps no clock-driven state: session rates are computed lazily at every
``evict`` call (``rescore_s`` is therefore ignored, [dep Q5] placeholder: the
per-node update timing of the note is under discussion with the user).

A *node* is a maximal run of Full-attention chunks of one request chain between
node ends (a branch, a leaf or a position that holds a state key), plus the
state chunks (Mamba, SWA window) anchored at its end. A node exposes at most
two actions:

* F: its front-most resident chunk (leaves a hole);
* T: the tail from F to the node end plus the resident endpoint state chunks.

State chunks leave L2 only inside a T. [dep U2] the 256-token grouping below
the chunk size is under discussion: placeholder = one chunk is one front unit.

The loss of an action is the whole-model recompute work the readers would have
to run for their next request (``closure_work``), priced with
``cost(a,b) = cost_b1 (b-a) + cost_b2 (b^2-a^2)/2``; the score is
``sum_r rate_r * max(0, work_after - work_before) / freed_chunks``.
L1 (GPU) residency is ignored, as in shipped vLLM.
"""

import bisect
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.core.eviction import (
    PROTECTED_RATE,
    EvictionConfig,
    RateParams,
    SessionTracker,
)
from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus

logger = init_logger(__name__)

_UNKNOWN = object()  # parent / anchor not decided yet


def session_rate(
    sessions: SessionTracker,
    params: RateParams,
    session: str,
    now: float,
    running_owner: str = "decode",
) -> float:
    """Rate of one reader session from the state of its latest request
    (marconi-base rev 12 sections 7.1/9, F7 (c)); no dead state."""
    rid = sessions.last.get(session)
    r = sessions.reqs.get(rid) if rid is not None else None
    if r is None:
        return sum(1.0 / max(g, params.floor) for g in params.tool.x) / len(
            params.tool.x
        )
    if r.prefill_done is None and r.done is None:
        return PROTECTED_RATE
    q = 1.0 - params.p_end
    if r.done is None:
        if running_owner == "protect":
            return PROTECTED_RATE
        e = max(0.0, now - (r.prefill_done if r.prefill_done is not None else now))
        return q / max(params.dec.mrl(e) + params.tool.mean, params.floor)
    t = max(0.0, now - r.done)
    st = q * params.tool.surv(t)
    p = st / max(st + params.p_end, 1e-12)
    return p / max(params.tool.mrl(t), params.floor)


class _Meta:
    __slots__ = ("owner", "session", "end_pos", "group", "seq")

    def __init__(self, owner, session, end_pos, group):
        self.owner = owner
        self.session = session
        self.end_pos = end_pos
        self.group = group
        self.seq = -1


class RateCachePolicy(CachePolicy):
    def __init__(
        self,
        cache_capacity: int,
        policy_config: Mapping[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(cache_capacity)
        if not policy_config or "rate" not in policy_config:
            raise ValueError("RateCachePolicy needs policy_config['rate']")
        self.cfg = EvictionConfig.from_dict({**policy_config["rate"], "policy": "rate"})
        self.params = RateParams.load(self.cfg.params, self.cfg.time_scale)
        self.clock = clock
        self.group_tokens: dict[int, int] = dict(
            policy_config.get("group_tokens_per_chunk", {})
        )
        # Sliding-window width in chunks per SWA group (Part B supplies it;
        # default: the window is the single chunk ending at the boundary).
        self.group_window_chunks: dict[int, int] = dict(
            policy_config.get("group_window_chunks", {})
        )
        self.kinds: list[str] = []
        self.min_reuse = 1
        self.sessions = SessionTracker()
        self.last_victim_score: float | None = None
        self._warned_pos = False
        self._reset_index()

    # ------------------------------------------------------------ state
    def _reset_index(self) -> None:
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self.meta: dict[OffloadKey, _Meta] = {}
        self.pending: dict[OffloadKey, _Meta] = {}
        self._seq = 0
        # (group, end position) of every key a live chain refers to.
        self.kinfo: dict[OffloadKey, tuple[int, int]] = {}
        # Tree over Full keys: parent decided once ("first seen wins").
        self.tparent: dict[OffloadKey, Any] = {}
        self.tchildren: dict[OffloadKey, set[OffloadKey]] = defaultdict(set)
        self.anchor: dict[OffloadKey, Any] = {}
        self.anchored: dict[OffloadKey, set[OffloadKey]] = defaultdict(set)
        self.req_chain: dict[str, dict[tuple[int, int], OffloadKey]] = {}
        self.session_keys: dict[str, set[OffloadKey]] = {}
        self.readers: dict[OffloadKey, set[str]] = defaultdict(set)
        self.owned: dict[str, set[OffloadKey]] = defaultdict(set)

    def set_group_windows(self, window_chunks: Mapping[int, int]) -> None:
        self.group_window_chunks.update(window_chunks)

    def set_group_kinds(self, kinds: Sequence[str], min_reuse_blocks: int) -> None:
        self.kinds = list(kinds)
        self.min_reuse = max(1, int(min_reuse_blocks))

    def _kind(self, g: int) -> str:
        return self.kinds[g] if g < len(self.kinds) else "full"

    def _g0(self) -> int:
        for g, k in enumerate(self.kinds):
            if k == "full":
                return g
        return 0

    # ------------------------------------------------------------ rates
    def _reader_rate(self, session: str, now: float) -> float:
        return session_rate(
            self.sessions, self.params, session, now, self.cfg.running_owner
        )

    # ------------------------------------------------------------ session events
    def on_request_arrival(self, req_context: ReqContext) -> None:
        if req_context.req_id not in self.sessions.reqs:
            self.sessions.arrived(
                req_context.req_id, req_context.session_id, self.clock()
            )

    def on_prefill_done(self, req_context: ReqContext) -> None:
        self.sessions.prefill_done(req_context.req_id, self.clock())

    def _session_of(self, ctx: ReqContext) -> str:
        r = self.sessions.reqs.get(ctx.req_id)
        if r is not None:
            return r.session
        return ctx.session_id if ctx.session_id is not None else f"\x00req:{ctx.req_id}"

    def _maybe_forget(self, rid: str | None) -> None:
        while rid is not None:
            if self.owned.get(rid):
                return
            r = self.sessions.reqs.get(rid)
            if r is None or r.done is None:
                return
            nxt = self.sessions.forget(rid)
            if rid in self.sessions.reqs:
                return
            rid = nxt

    # ------------------------------------------------------------ key bookkeeping
    def _note_keys(self, ctx: ReqContext, keys: Iterable[OffloadKey]) -> None:
        chain = self.req_chain.setdefault(ctx.req_id, {})
        for k in keys:
            g = get_offload_group_idx(k)
            pos = ctx.get_offload_key_position(k)
            if pos is None:
                if not self._warned_pos:
                    logger.warning(
                        "rate policy: no position for a key of request %s; "
                        "using 0",
                        ctx.req_id,
                    )
                    self._warned_pos = True
                pos = 0
            self.kinfo[k] = (g, pos)
            chain[(g, pos)] = k

    def _decide(self, k: OffloadKey, chain: Mapping[tuple[int, int], OffloadKey],
                final: bool) -> None:
        """Decide the parent (Full key) or the anchor (state key) of an
        inserted key from a request chain; undecided stays ``_UNKNOWN`` until
        a request that finishes sees the whole chain (``final``)."""
        g, pos = self.kinfo[k]
        if self._kind(g) == "full":
            if self.tparent.get(k, _UNKNOWN) is not _UNKNOWN:
                return
            t = self.group_tokens.get(g)
            if t is None or pos - t <= 0:
                parent: Any = None
            else:
                cand = chain.get((g, pos - t))
                if cand is None:
                    if not final:
                        return
                    parent = None
                elif cand in self.tparent and cand != k:
                    parent = cand
                else:
                    parent = None  # orphan: the parent was never stored
            self.tparent[k] = parent
            if parent is not None:
                self.tchildren[parent].add(k)
        else:
            if self.anchor.get(k, _UNKNOWN) is not _UNKNOWN:
                return
            cand = chain.get((self._g0(), pos))
            if cand is None:
                if not final:
                    return
                a: Any = None
            else:
                a = cand if cand in self.tparent else None
            self.anchor[k] = a
            if a is not None:
                self.anchored[a].add(k)

    def _alive(self, k: OffloadKey) -> bool:
        if k in self.chunks or self.tchildren.get(k):
            return True
        return any(s in self.chunks for s in self.anchored.get(k, ()))

    def _prune(self, k: OffloadKey | None) -> None:
        """Drop absent structure no resident key depends on any more."""
        while k is not None and k in self.tparent and not self._alive(k):
            parent = self.tparent.pop(k)
            self.tchildren.pop(k, None)
            for s in self.anchored.pop(k, ()):
                self.anchor.pop(s, None)
                self._unref(s)
            if parent is not None and parent is not _UNKNOWN:
                self.tchildren[parent].discard(k)
            self._unref(k)
            k = parent

    def _unref(self, k: OffloadKey) -> None:
        if k not in self.chunks and k not in self.tparent and k not in self.anchor \
                and not self.readers.get(k) and k not in self.pending:
            self.kinfo.pop(k, None)
            self.readers.pop(k, None)

    # ------------------------------------------------------------ CachePolicy
    @override
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    @override
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        return

    @override
    def on_store_miss(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        keys = list(keys)
        self._note_keys(req_context, keys)
        session = self._session_of(req_context)
        for k in keys:
            if k in self.chunks or k in self.pending:
                continue
            g, pos = self.kinfo[k]
            self.pending[k] = _Meta(req_context.req_id, session, pos, g)

    @override
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        m = self.pending.get(key)
        if m is None:
            g = get_offload_group_idx(key)
            g, pos = self.kinfo.get(key, (g, 0))
            m = _Meta(None, "\x00none", pos, g)
            self.pending[key] = m
            self.kinfo[key] = (g, pos)
        self._seq += 1
        m.seq = self._seq
        self.chunks[key] = chunk
        self.meta[key] = m
        if m.owner is not None:
            self.owned[m.owner].add(key)
        if self._kind(m.group) == "full":
            self.tparent.setdefault(key, _UNKNOWN)
        self._decide(key, self.req_chain.get(m.owner, {}), final=False)
        self.session_keys.setdefault(m.session, set()).add(key)
        self.readers[key].add(m.session)

    def _drop(self, key: OffloadKey) -> _Meta:
        self.chunks.pop(key)
        m = self.meta.pop(key)
        if m.owner is not None:
            self.owned[m.owner].discard(key)
            if not self.owned[m.owner]:
                del self.owned[m.owner]
        return m

    @override
    def remove(self, key: OffloadKey) -> None:
        """A failed store: the key leaves its node and the reader sets. The
        pending metadata stays, so a retried store inserts it again."""
        m = self._drop(key)
        self.pending[key] = m
        for s in self.readers.pop(key, ()):
            self.session_keys.get(s, set()).discard(key)
        self._prune(key)
        self._maybe_forget(m.owner)

    @override
    def on_request_finished(
        self,
        key_groups: Sequence[Sequence[OffloadKey]],
        insertion_only_keys: set[OffloadKey],
        reused_keys: set[OffloadKey],
        req_context: ReqContext,
    ) -> None:
        del insertion_only_keys, reused_keys
        rid = req_context.req_id
        self.on_request_arrival(req_context)
        keys = [k for kg in key_groups for k in kg]
        self._note_keys(req_context, keys)
        chain = self.req_chain.get(rid, {})
        for k in keys:
            if k in self.chunks:
                self._decide(k, chain, final=True)
        for k in [k for k, m in self.pending.items() if m.owner == rid]:
            del self.pending[k]
            self._unref(k)
        self.sessions.finished(rid, self.clock())
        session = self.sessions.reqs[rid].session
        if self.sessions.last.get(session) == rid:
            new = set(keys)
            old = self.session_keys.get(session, set())
            for k in old - new:
                self.readers[k].discard(session)
                self._unref(k)
            for k in new:
                self.readers[k].add(session)
            self.session_keys[session] = new
        self.req_chain.pop(rid, None)
        self._maybe_forget(rid)

    @override
    def clear(self) -> None:
        self._reset_index()

    # ------------------------------------------------------------ closure work
    def _cost(self, a: int, b: int) -> float:
        return self.cfg.cost_b1 * (b - a) + self.cfg.cost_b2 * (b * b - a * a) / 2.0

    def _closure_work(self, session: str, present: set[OffloadKey]) -> float:
        """Recompute work of the reader's next request when only ``present``
        keys of its chain are resident: the missing Full chunks (runs of hits
        shorter than ``min_reuse`` after the first miss count as missing) form
        spans; a span or the final frontier may only start at a boundary where
        every state group of the chain is complete, else it moves back to the
        previous complete boundary (0 at worst); overlapping spans are unioned.
        """
        chain = self.session_keys.get(session)
        if not chain:
            return 0.0
        by_group: dict[int, dict[int, OffloadKey]] = defaultdict(dict)
        for k in chain:
            g, pos = self.kinfo[k]
            by_group[g][pos] = k
        state_groups = [g for g in by_group if self._kind(g) != "full"]
        full_groups = [g for g in by_group if self._kind(g) == "full"]
        if state_groups:
            dend = max(p for g in state_groups for p in by_group[g])
        else:
            dend = max((p for g in full_groups for p in by_group[g]), default=0)
        spans: list[tuple[int, int]] = []
        for g in full_groups:
            t = self.group_tokens.get(g)
            if not t:
                continue
            items = sorted((p, k) for p, k in by_group[g].items() if p <= dend)
            hit = [k in present for _, k in items]
            if self.min_reuse > 1 and False in hit:
                i = hit.index(False)
                while i < len(hit):
                    if not hit[i]:
                        i += 1
                        continue
                    j = i
                    while j < len(hit) and hit[j]:
                        j += 1
                    if j - i < self.min_reuse:
                        for x in range(i, j):
                            hit[x] = False
                    i = j
            for (p, _), h in zip(items, hit):
                if not h:
                    if spans and spans[-1][1] == p - t:
                        spans[-1] = (spans[-1][0], p)
                    else:
                        spans.append((p - t, p))
        if state_groups:
            bounds = sorted(
                {p for g in state_groups for p in by_group[g]}
            )

            def complete(p: int) -> bool:
                if p <= 0:
                    return True
                for g in state_groups:
                    t = self.group_tokens.get(g, 0)
                    w = 1 if self._kind(g) == "mamba" else self.group_window_chunks.get(g, 1)
                    for i in range(w):
                        k = by_group[g].get(p - i * t)
                        if k is None or k not in present:
                            return False
                return True

            def back(a: int) -> int:
                i = bisect.bisect_left(bounds, a)
                for p in reversed(bounds[:i]):
                    if complete(p):
                        return p
                return 0

            if dend > 0 and not complete(dend):
                spans.append((back(dend), dend))
            spans = [(a if complete(a) else back(a), b) for a, b in spans]
        spans.sort()
        merged: list[list[int]] = []
        for a, b in spans:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return sum(self._cost(a, b) for a, b in merged)

    # ------------------------------------------------------------ nodes and actions
    def _nodes(
        self, present: set[OffloadKey]
    ) -> dict[OffloadKey, dict[str, Any]]:
        """Nodes of the resident keys: id (end key) -> {full, state, st, end}."""
        node_of: dict[OffloadKey, OffloadKey] = {}
        info: dict[OffloadKey, dict[str, Any]] = {}
        order = sorted(
            self.tparent, key=lambda k: self.kinfo[k][1], reverse=True
        )
        for k in order:
            ch = self.tchildren.get(k)
            is_end = (not ch or len(ch) != 1) or bool(self.anchored.get(k))
            nid = k if is_end else node_of[next(iter(ch))]
            node_of[k] = nid
            g, pos = self.kinfo[k]
            n = info.setdefault(
                nid,
                {"full": [], "state": [], "st": pos, "end": self.kinfo[nid][1]},
            )
            n["st"] = min(n["st"], pos - self.group_tokens.get(g, 0))
            if k in present:
                n["full"].append(k)
        for sk, a in self.anchor.items():
            if sk in present and a is not None and a in node_of:
                info[node_of[a]]["state"].append(sk)
        for sk in present:
            g = self.kinfo[sk][0]
            if self._kind(g) != "full" and self.anchor.get(sk) is None:
                pos = self.kinfo[sk][1]
                n = info.setdefault(
                    sk, {"full": [], "state": [], "st": pos, "end": pos}
                )
                n["state"].append(sk)
        out = {}
        for nid, n in info.items():
            if n["full"] or n["state"]:
                n["full"].sort(key=lambda k: self.kinfo[k][1])
                n["state"].sort()
                out[nid] = n
        return out

    def node_layout_for_test(self) -> dict[OffloadKey, tuple]:
        return {
            nid: (tuple(n["full"]), tuple(n["state"]), n["st"], n["end"])
            for nid, n in self._nodes(set(self.chunks)).items()
        }

    def snapshot_for_test(self) -> tuple:
        return (
            tuple(sorted(self.chunks)),
            tuple(sorted((k, repr(v)) for k, v in self.tparent.items())),
            tuple(sorted((k, tuple(sorted(v))) for k, v in self.readers.items())),
        )

    def _score_of(
        self,
        keys: list[OffloadKey],
        present: set[OffloadKey],
        rates: dict[str, float],
        before: dict[str, float],
        now: float,
    ) -> float:
        sessions: set[str] = set()
        for k in keys:
            sessions |= self.readers.get(k, set())
        after = present - set(keys)
        total = 0.0
        for s in sessions:
            if s not in rates:
                rates[s] = self._reader_rate(s, now)
            if s not in before:
                before[s] = self._closure_work(s, present)
            total += rates[s] * max(0.0, self._closure_work(s, after) - before[s])
        return total / len(keys)

    def _actions(
        self,
        n: dict[str, Any],
        present: set[OffloadKey],
        protected: set[OffloadKey],
        rates: dict[str, float],
        before: dict[str, float],
        now: float,
    ) -> tuple[float, int, list[OffloadKey]] | None:
        def ok(k: OffloadKey) -> bool:
            return self.chunks[k].ref_cnt == 0 and k not in protected

        full, state = n["full"], n["state"]
        if full:
            if not ok(full[0]):
                return None
            f_keys = [full[0]]
        else:
            f_keys = None
        t_keys = full + state
        if f_keys is None:
            if not state or not all(ok(k) for k in state):
                return None
            f_keys = None
        t_ok = all(ok(k) for k in t_keys) and (f_keys is None or len(t_keys) > 1)
        best = None
        if f_keys is not None:
            sf = self._score_of(f_keys, present, rates, before, now)
            best = (sf, f_keys)
        if t_ok:
            st = self._score_of(t_keys, present, rates, before, now)
            if best is None or st < best[0]:
                best = (st, t_keys)
        if best is None:
            return None
        return best[0], self.meta[best[1][0]].seq, best[1]

    def node_action_scores_for_test(self, node_id: OffloadKey):
        """(score of F, score of T or None) of one node with everything
        eligible."""
        present = set(self.chunks)
        n = self._nodes(present)[node_id]
        now = self.clock()
        rates: dict[str, float] = {}
        before: dict[str, float] = {}
        f = self._score_of([n["full"][0]], present, rates, before, now)
        keys = n["full"] + n["state"]
        t = self._score_of(keys, present, rates, before, now) if len(keys) > 1 else None
        return f, t

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        now = self.clock()
        rates: dict[str, float] = {}
        present = set(self.chunks)
        removed: list[OffloadKey] = []
        last_score = None
        while len(removed) < n:
            before: dict[str, float] = {}
            best = None
            for nid, node in self._nodes(present).items():
                a = self._actions(node, present, protected, rates, before, now)
                if a is None:
                    continue
                cand = (a[0], a[1], nid, a[2])
                if best is None or cand[:3] < best[:3]:
                    best = cand
            if best is None:
                return None
            last_score = best[0]
            keys = best[3]
            removed.extend(keys)
            present.difference_update(keys)
        out = []
        for k in removed:
            chunk = self.chunks[k]
            m = self._drop(k)
            self._prune(k)
            self._unref(k)
            self._maybe_forget(m.owner)
            out.append((k, chunk))
        self.last_victim_score = last_score
        return out
