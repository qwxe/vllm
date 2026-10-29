# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/rate: the session-state `rate` eviction policy (CPU only).

Run with pytest or ``python test_rate_eviction.py``. The simulator parity test
imports LLMServingSim's ``serving.core.cost_aware_pool`` when it is reachable
(``LLMSERVINGSIM`` env var or ~/LLMServingSim) and otherwise checks the
recorded order.
"""

import os
import random
import sys

try:
    from . import holes_utils as U
except ImportError:  # run as a script
    import holes_utils as U

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.eviction import (
    EvictionConfig,
    RateEvictionPolicy,
    chunk_recompute_s,
    session_id_of,
)
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

PARAMS = {
    "tool_gaps": [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 300],
    "decode_durs": [1, 2, 4, 8, 16],
    "p_end": 0.3,
    "floor": 1.0,
}
BPC = 256 // U.BS  # blocks per 256-token chunk


# ---------------------------------------------------------------- harness


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class PoolHarness:
    """A real BlockPool with the rate policy and a fake clock."""

    def __init__(self, num_blocks, running_owner="protect", rescore_s=0.0):
        self.clock = Clock()
        cfg = EvictionConfig(
            policy="rate", params=PARAMS, rescore_s=rescore_s, running_owner=running_owner
        )
        self.pool = BlockPool(num_blocks, True, U.BS)
        self.pol = RateEvictionPolicy(cfg, U.BS, clock=self.clock)
        self.pool.set_eviction_policy(self.pol)
        self._n = 0

    def hold(self, positions, cached=True):
        """Allocate one block per position (block index in a table)."""
        blocks = self.pool.get_new_blocks(len(positions))
        table = [self.pool.null_block] * (max(positions) + 1)
        for p, b in zip(positions, blocks):
            table[p] = b
            if cached:
                self._n += 1
                h = make_block_hash_with_group_id(BlockHash(b"h%d" % self._n), 0)
                b.set_block_hash(h)
                self.pool.cached_block_hash_to_block.insert(h, b)
        return table

    def release(self, rid, table):
        self.pol.on_request_free(rid, table)
        real = [b for b in table if not b.is_null]
        self.pool.free_blocks(reversed(real))

    def check(self):
        pol = self.pol
        free = {b.block_id for b in self.pool.free_block_queue.get_all_free_blocks()}
        grouped = set(pol._block_group)
        unc = set(pol._uncached)
        assert not (grouped & unc)
        assert free == grouped | unc, (free ^ (grouped | unc))
        assert len(free) == self.pool.get_num_free_blocks()
        for bid in grouped:
            assert self.pool.blocks[bid].block_hash is not None
        for bid in unc:
            assert self.pool.blocks[bid].block_hash is None
        for key, g in pol._groups.items():
            assert g and all(pol._block_group[b] == key for b in g)


def _owner_of(h, blocks):
    return {h.pol._meta[b.block_id] for b in blocks}


# ---------------------------------------------------------------- parity case
# Requests are ints, consecutive within a session (the simulator's rule).
# (rid, session, index, arrival, prefill_done, done, chunks)
CASE = [
    (10, "A", 0, 0.0, 1.0, 5.0, [0, 1, 2]),     # idle 995 s, successor not here
    (20, "B", 0, 900.0, 950.0, 990.0, [0, 1, 40]),  # idle 10 s, one deep chunk
    (30, "C", 0, 800.0, 850.0, 900.0, [0, 1]),  # successor queued -> protected
    (31, "C", 1, 995.0, None, None, []),
    (40, "D", 0, 100.0, 200.0, 500.0, [0, 1, 2]),  # successor through prefill -> dead
    (41, "D", 1, 600.0, 601.0, None, []),
    (50, "E", 0, 999.0, None, None, [0]),        # prefilling -> protected
    (60, "F", 0, 940.0, 950.0, None, [0, 3]),    # decoding 50 s (simulator rule)
    (70, "G", 0, 0.0, 10.0, 20.0, [5]),          # idle 980 s, mid position
]
NOW = 1000.0

# The simulator's eviction order on CASE (computed with SessRatePool, see
# _simulator_order); ties (score 0) broken by (owner, chunk).
EXPECTED = [
    (10, 0), (10, 1), (10, 2), (40, 0), (40, 1), (40, 2), (70, 5),
    (20, 0), (20, 1), (20, 40),
    (60, 0), (60, 3),
    (30, 0), (50, 0), (30, 1),
]


def _rate_order(running_owner="decode", via_heap=True):
    nchunks = sum(len(c) for *_, c in CASE)
    h = PoolHarness(1 + nchunks * BPC, running_owner=running_owner)
    pol = h.pol
    events = []
    for rid, sess, _, arr, pf, done, chunks in CASE:
        events.append((arr, 0, "arr", rid, sess))
        if pf is not None:
            events.append((pf, 1, "pf", rid, sess))
        if done is not None:
            events.append((done, 2, "done", rid, sess))
    tables = {}
    for t, _, kind, rid, sess in sorted(events):
        h.clock.t = t
        if kind == "arr":
            pol.on_arrival(str(rid), sess)
            chunks = next(c for r, *_, c in CASE if r == rid)
            if chunks:
                pos = [k * BPC + j for k in chunks for j in range(BPC)]
                tables[rid] = h.hold(pos)
        elif kind == "pf":
            pol.on_prefill_done(str(rid))
        else:
            # A request frees its blocks when it finishes.
            pol.on_finished(str(rid))
            if rid in tables:
                h.release(str(rid), tables.pop(rid))
    # Unfinished owners (prefilling, decoding) free theirs on preemption.
    h.clock.t = NOW - 1
    for rid, tab in tables.items():
        h.release(str(rid), tab)
    h.check()
    h.clock.t = NOW
    scores = {(int(o), k): sc for o, k, sc, _ in pol.candidate_order()}
    if not via_heap:
        return _norm(list(scores), scores), scores
    order = []
    while h.pool.get_num_free_blocks():
        got = h.pool.get_new_blocks(BPC)
        h.check()
        groups = {(o, i // BPC) for o, i in _owner_of(h, got)}
        assert len(groups) == 1, groups
        o, k = groups.pop()
        order.append((int(o), k))
    # Popped in non-decreasing score order.
    assert all(scores[a] <= scores[b] for a, b in zip(order, order[1:])), order
    return _norm(order, scores), scores


def _norm(order, scores):
    """Order by score, ties by (owner, chunk)."""
    return sorted(order, key=lambda x: (scores[x], x))


def _simulator_order():
    root = os.environ.get("LLMSERVINGSIM", os.path.expanduser("~/LLMServingSim"))
    if not os.path.isdir(os.path.join(root, "serving", "core")):
        return None
    sys.path.insert(0, root)
    try:
        from serving.core import cost_aware_pool as C
        from serving.core import sess_state as SS
        from serving.core.block_pool import Device
    except Exception:  # noqa: BLE001
        return None
    finally:
        sys.path.remove(root)
    for d in (SS.SESS, SS.CTXEND, SS.INPUT, SS.ARR, SS.PF, SS.DONE, SS.NEXTF):
        d.clear()
    S = 1_000_000_000
    by_sess = {}
    for rid, sess, idx, *_ in CASE:
        by_sess.setdefault(sess, []).append(rid)
    for sess, rids in by_sess.items():
        # The simulator knows the session length; give every session one
        # more request than the case has, so no owner is a session's last.
        n = len(rids) + 1
        SS.register_session(sess, min(rids), [{"input_toks": 1, "output_toks": 1}] * n)
    nchunks = sum(len(c) for *_, c in CASE)
    cfg = C.CostAwareConfig(tau_ns=1, rescore_ns=0)
    pool = C.SessRatePool(Device.CPU, nchunks, 256, 1, 4096, cfg, mode="rate", params=PARAMS)
    n = 0
    for rid, sess, _, arr, pf, done, chunks in CASE:
        SS.ARR[rid] = int(arr * S)
        if pf is not None:
            SS.PF[rid] = int(pf * S)
        if done is not None:
            SS.DONE[rid] = int(done * S)
        pool.set_clock(int((NOW - 1) * S))
        pool.cur_owner = rid
        for k in chunks:
            n += 1
            pool.cache_copy(("h", n), index=k)
    pool.set_clock(int(NOW * S))
    rows = [(score, slot.owner, slot.index) for slot, _, score in pool._scores()]
    rows.sort()
    return [(o, k) for _, o, k in rows], {(o, k): s for s, o, k in rows}


def test_parity_with_simulator():
    order, ours = _rate_order(running_owner="decode")
    assert order == EXPECTED, order
    cand, _ = _rate_order(running_owner="decode", via_heap=False)
    assert cand == EXPECTED, cand
    sim = _simulator_order()
    if sim is None:
        print("  (simulator not importable: checked the recorded order only)")
        return
    sim_order, sim_scores = sim
    assert sim_order == EXPECTED, sim_order
    # Scores agree too.
    for key, s in sim_scores.items():
        assert abs(ours[key] - s) <= 1e-9 * max(1.0, abs(s)), (key, ours[key], s)


def test_running_owner_protected_by_default():
    order, _ = _rate_order(running_owner="protect")
    # Owner 60 (past prefill, not finished) moves behind every live owner.
    assert order.index((60, 0)) > order.index((20, 40))


def test_chunk_cost_position():
    b1, b2 = 3.87e-5, 1.85e-9
    c0 = chunk_recompute_s(0, 256, b1, b2)
    c400 = chunk_recompute_s(400, 256, b1, b2)  # ~100K tokens
    assert abs(c0 - (b1 * 256 + b2 * 256 * 256 / 2)) < 1e-15
    assert 5.0 < c400 / c0 < 7.0  # "a token at 100K costs ~5.7x one at the front"


# ---------------------------------------------------------------- pool level


def test_uncached_blocks_first_and_whole_chunks():
    h = PoolHarness(1 + 2 * BPC + 3)
    h.clock.t = 0.0
    h.pol.on_arrival("a", "s")
    h.pol.on_prefill_done("a")
    tab = h.hold(list(range(2 * BPC)))
    extra = h.hold([0, 1, 2], cached=False)
    h.pol.on_finished("a")
    h.release("a", tab)
    h.release("x", extra)
    h.check()
    h.clock.t = 100.0
    # 3 uncached blocks are used before any cached chunk is touched.
    got = h.pool.get_new_blocks(3)
    assert all(b.block_hash is None for b in got)
    assert len(h.pool.cached_block_hash_to_block) == 2 * BPC
    # One more block evicts one whole chunk (front first: cheapest).
    got = h.pool.get_new_blocks(1)
    assert len(h.pool.cached_block_hash_to_block) == BPC
    assert h.pol._meta[got[0].block_id] == ("a", 0)
    assert len(h.pol._uncached) == BPC - 1
    h.check()
    # The rest of that chunk is used next, without another eviction.
    h.pool.get_new_blocks(BPC - 1)
    assert len(h.pool.cached_block_hash_to_block) == BPC
    h.check()


def test_touch_protects_and_owner_moves():
    h = PoolHarness(1 + 2 * BPC)
    for r in ("a", "b"):
        h.pol.on_arrival(r, r)
        h.pol.on_prefill_done(r)
    ta = h.hold(list(range(BPC)))
    tb = h.hold(list(range(BPC)))
    h.clock.t = 1.0
    h.pol.on_finished("a")
    h.release("a", ta)
    h.clock.t = 2.0
    h.pol.on_finished("b")
    h.release("b", tb)
    # A hit on a's chunk takes it out of the candidates.
    h.pool.touch([b for b in ta if not b.is_null])
    h.check()
    assert {o for o, *_ in h.pol.candidate_order()} == {"b"}
    # A new request of session a holds it, then frees it: the owner moves.
    h.pol.on_arrival("a2", "a")
    h.pol.on_prefill_done("a2")
    h.clock.t = 3.0
    h.pol.on_finished("a2")
    h.release("a2", ta)
    h.check()
    assert {o for o, *_ in h.pol.candidate_order()} == {"a2", "b"}
    # a's record is no longer needed by any chunk.
    assert "a" not in h.pol.sessions.reqs


def test_evict_blocks_hook_moves_to_uncached():
    h = PoolHarness(1 + BPC)
    h.pol.on_arrival("a", None)
    tab = h.hold(list(range(BPC)))
    h.pol.on_finished("a")
    h.release("a", tab)
    h.pool.evict_blocks({tab[3].block_id})
    h.check()
    assert len(h.pol._uncached) == 1


def test_invariants_fuzz():
    rng = random.Random(0)
    for seed in range(5):
        rng.seed(seed)
        h = PoolHarness(1 + 200, rescore_s=rng.choice([0.0, 5.0]))
        live = {}
        n = 0
        for _ in range(400):
            h.clock.t += rng.random()
            op = rng.random()
            if op < 0.35 and h.pool.get_num_free_blocks() > 40:
                n += 1
                rid = f"r{n}"
                h.pol.on_arrival(rid, f"s{rng.randrange(6)}")
                if rng.random() < 0.8:
                    h.pol.on_prefill_done(rid)
                k = rng.randrange(1, 40)
                live[rid] = h.hold(list(range(k)), cached=rng.random() < 0.9)
            elif op < 0.6 and live:
                rid = rng.choice(sorted(live))
                if rng.random() < 0.8:
                    h.pol.on_prefill_done(rid)
                    h.pol.on_finished(rid)
                h.release(rid, live.pop(rid))
            elif op < 0.8:
                cands = [b for b in h.pool.blocks[1:] if b.ref_cnt == 0 and b.block_hash]
                if cands and h.pool.get_num_free_blocks() > 40:
                    n += 1
                    rid = f"t{n}"
                    h.pol.on_arrival(rid, None)
                    pick = rng.sample(cands, min(len(cands), rng.randrange(1, 20)))
                    h.pool.touch(pick)
                    live[rid] = pick
            elif op < 0.9:
                bid = rng.randrange(1, 201)
                h.pool.evict_blocks({bid})
            else:
                k = rng.randrange(1, 30)
                if k <= h.pool.get_num_free_blocks():
                    n += 1
                    rid = f"g{n}"
                    h.pol.on_arrival(rid, None)
                    blocks = h.pool.get_new_blocks(k)
                    live[rid] = blocks
            h.check()


# ---------------------------------------------------------------- scheduler


def _scheduler(eviction, num_blocks, holes=True):
    model_config = ModelConfig(
        model=U.TINY, skip_tokenizer_init=True, max_model_len=8192, dtype="bfloat16"
    )
    sched = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=2048,
        max_model_len=8192,
        enable_chunked_prefill=True,
        is_encoder_decoder=False,
    )
    cache = CacheConfig(
        block_size=U.BS, gpu_memory_utilization=0.9, cache_dtype="auto",
        enable_prefix_caching=True,
    )
    cache.num_gpu_blocks = num_blocks
    add = {}
    if holes:
        add["holes"] = {"enabled": True}
    if eviction is not None:
        add["eviction"] = eviction
    vc = VllmConfig(
        model_config=model_config, cache_config=cache, scheduler_config=sched,
        additional_config=add,
    )
    return Scheduler(
        vllm_config=vc,
        kv_cache_config=U.kv_cache_config(num_blocks),
        structured_output_manager=StructuredOutputManager(vc),
        block_size=U.BS,
        log_stats=True,
    )


def _req(rid, prompt, session, max_tokens=4):
    return Request(
        request_id=rid,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(
            max_tokens=max_tokens, ignore_eos=True, extra_args={"session_id": session}
        ),
        pooling_params=None,
        block_hasher=U.HASHER,
    )


def test_session_id_prefers_native_and_falls_back_only_when_unset():
    req = _req("session", [1, 2], "legacy")
    assert session_id_of(req) == "legacy"
    req.session_id = "native"
    assert session_id_of(req) == "native"
    req.session_id = ""
    assert session_id_of(req) == ""
    req.session_id = None
    assert session_id_of(req) == "legacy"


def _drain(s, max_steps=200, prefill_stats=None):
    outs = []
    for _ in range(max_steps):
        if not s.requests or all(r.is_finished() for r in s.requests.values()):
            break
        outs.append(U.step(s, prefill_stats=prefill_stats))
    return outs


def _session_trace(eviction, num_blocks=1 + 160):
    """Two sessions, 1024-token first turns; a third request fills the cache;
    then session A's second turn. Returns the trace, A2's prefill stats and scheduler."""
    s = _scheduler(eviction, num_blocks)
    if s.eviction_policy is not None:
        clk = Clock(0.0)
        s.eviction_policy.clock = clk
    else:
        clk = Clock(0.0)
    pa = list(range(1, 1 + 64 * U.BS))
    pb = list(range(5001, 5001 + 64 * U.BS))
    pc = list(range(9001, 9001 + 40 * U.BS))
    trace = []
    stats = {}

    def run():
        for o in _drain(s, prefill_stats=stats):
            trace.append(
                (
                    dict(o.num_scheduled_tokens),
                    [r.block_ids for r in o.scheduled_new_reqs],
                    o.scheduled_hole_ranges,
                )
            )

    a1 = _req("a1", pa, "A", max_tokens=1)
    s.add_request(a1)
    run()
    clk.t = 10.0
    b1 = _req("b1", pb, "B", max_tokens=1)
    s.add_request(b1)
    run()
    clk.t = 400.0  # a1 idle 400 s, b1 idle 390 s
    c1 = _req("c1", pc, "C", max_tokens=1)
    s.add_request(c1)
    run()
    clk.t = 401.0
    a2 = _req("a2", pa + [7] + list(range(20001, 20001 + 100)), "A", max_tokens=2)
    s.add_request(a2)
    run()
    return trace, stats[a2.request_id], s


def test_lru_default_is_stock():
    s = _scheduler(None, 50)
    assert s.eviction_policy is None and s.kv_cache_manager.block_pool.eviction_policy is None
    s = _scheduler({"policy": "lru"}, 50)
    assert s.eviction_policy is None
    t_none, a_none, _ = _session_trace(None)
    t_lru, a_lru, _ = _session_trace({"policy": "lru"})
    assert t_none == t_lru
    # Stock LRU evicted a's tail (freed tail first, oldest): a2 hits a prefix.
    assert a_none.num_cached_tokens < 64 * U.BS


def test_rate_holes_front_first_and_reuse():
    t_rate, a2, s = _session_trace({"policy": "rate", "params": PARAMS})
    pol = s.eviction_policy
    assert pol is not None and pol.evicted_groups > 0
    # c1 needed 40 blocks with 160 - 128 - 1 = 31 free: one chunk evicted,
    # from the session idle longest (a1), front first.
    hole_steps = [t[2].get("a2") for t in t_rate if t[2] and "a2" in t[2]]
    assert hole_steps and hole_steps[0][0] == (0, 256), hole_steps
    # a2 reused a1's chunks after the hole.
    assert a2.num_cached_tokens == 64 * U.BS - 256, a2.num_cached_tokens


def test_rate_dead_session_first():
    """A session whose next request is through prefill: its leftover chunks
    (not in the successor's context) go first, although LRU would take the
    older session Y."""

    def run(eviction):
        s = _scheduler(eviction, 1 + 3 * 64 + 8)
        clk = Clock(0.0)
        if s.eviction_policy is not None:
            s.eviction_policy.clock = clk
        py = list(range(90001, 90001 + 64 * U.BS))
        s.add_request(_req("y1", py, "Y", max_tokens=1))
        _drain(s)
        clk.t = 1.0
        p1 = list(range(1, 1 + 64 * U.BS))
        s.add_request(_req("x1", p1, "X", max_tokens=1))
        _drain(s)
        clk.t = 2.0
        # The successor shares only the first 16 blocks (an edited prompt).
        p2 = p1[: 16 * U.BS] + list(range(70001, 70001 + 48 * U.BS))
        s.add_request(_req("x2", p2, "X", max_tokens=1))
        _drain(s)
        clk.t = 3.0
        s.add_request(_req("z1", list(range(30001, 30001 + 40 * U.BS)), "Z", max_tokens=1))
        _drain(s)
        probe_y = _req("py", py, "Q")
        probe_x = _req("px", p1, "Q")
        kv = s.kv_cache_manager
        return s, kv.get_computed_blocks(probe_y)[1], kv.get_cache_hits_with_holes(probe_x)[0]

    s, y_hit, x_hits = run({"policy": "rate", "params": PARAMS})
    pol = s.eviction_policy
    assert pol.evicted_blocks == {"dead": 16}, pol.evicted_blocks
    assert y_hit == 64 * U.BS - U.BS  # y1 intact (the last block is never a hit)
    assert x_hits[16:32] == [None] * 16 and all(x_hits[:16]), x_hits
    s, y_hit, _ = run({"policy": "lru"})
    assert y_hit < 64 * U.BS - U.BS  # LRU took y1's tail


if __name__ == "__main__":
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
