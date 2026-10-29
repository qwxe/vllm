# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes-v2 stage 2, L commit 1: the session-rate CachePolicy at L2.

CPU only, no GPU. The policy is driven directly with ``ReqContext`` fakes and
through ``CPUOffloadingManager`` with fake transfers. Every expected value is
hand-derived (L note sections 5, 6, 10: T1-T9, T14, T17). Parts that the note
leaves open are marked [dep U2] (grouping below 256 tokens: placeholder = one
chunk is one front unit) and [dep Q5] (per-node update timing: placeholder =
lazy rates at every ``evict`` call).
"""

import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

T = 16  # tokens per chunk in all fixtures
PARAMS = {
    "tool_gaps": [2, 4, 6, 8],
    "decode_durs": [1, 3],
    "p_end": 0.25,
    "floor": 0.5,
}


def rate_cfg(**over):
    rate = {
        "params": PARAMS,
        "rescore_s": 0,
        "cost_b1": 0.5,
        "cost_b2": 1.0,
        "running_owner": "decode",
    }
    rate.update(over)
    return {"rate": rate, "group_tokens_per_chunk": {0: T, 1: T, 2: T}}


class Clock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def key(i, g=0):
    return make_offload_key(b"h%03d" % i, g)


def ctx(req, session=None):
    return ReqContext(req_id=req, session_id=session)


def make_policy(cap=64, kinds=("full",), min_reuse=1, clock=None, **over):
    cls = CachePolicyFactory.get_cache_policy_cls("rate")
    pol = cls(cache_capacity=cap, policy_config=rate_cfg(**over), clock=clock or Clock())
    pol.set_group_kinds(list(kinds), min_reuse)
    return pol


def put_chain(pol, c, ids, g=0, start_pos=T, ready=True, pos=None):
    """Store keys for ids with end positions start_pos, start_pos+T, ..."""
    keys = []
    for n, i in enumerate(ids):
        k = key(i, g)
        p = pos[n] if pos else start_pos + n * T
        c.set_offload_key_position(k, p)
        keys.append(k)
    pol.on_store_miss(keys, c)
    chunks = {}
    for n, k in enumerate(keys):
        ch = ChunkStatus(len(pol.chunks) + 1000 * (g + 1) + n)
        pol.insert(k, ch)
        if ready:
            ch.ref_cnt = 0
            pol.mark_evictable(k)
        chunks[k] = ch
    return keys, chunks


def adopt(c, src, keys):
    """The request ``c`` reuses keys another request positioned (prefix hits
    carry the same absolute positions in every request)."""
    for k in keys:
        c.set_offload_key_position(k, src._offload_key_positions[k])


def finish(pol, c, key_groups, clock=None, t=None):
    if t is not None:
        clock.t = t
    pol.on_request_finished(
        tuple(tuple(g) for g in key_groups), set(), set(), c
    )


def set_rates(pol, rates):
    pol._reader_rate = lambda session, now: rates.get(session, 1.0)


# ---------------------------------------------------------------- T9 / T14 / factory


def test_rate_policy_is_registered_and_arc_lru_stay():
    cls = CachePolicyFactory.get_cache_policy_cls("rate")
    assert cls.__name__ == "RateCachePolicy"
    assert CachePolicyFactory.get_cache_policy_cls("lru") is LRUCachePolicy
    assert CachePolicyFactory.get_cache_policy_cls("arc").__name__ == "ARCCachePolicy"


def test_default_policies_are_built_without_policy_config():
    for name, cls_name in (("lru", "LRUCachePolicy"), ("arc", "ARCCachePolicy")):
        mgr = CPUOffloadingManager(num_chunks=4, cache_policy=name)
        assert type(mgr._policy).__name__ == cls_name
        # the hand-off of the hole config is a no-op for stock policies
        mgr.set_group_kinds(["full"], 1)


def test_missing_params_and_unknown_key_are_value_errors():
    from tests.v1.kv_offload.test_factory import _make_offloading_config
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    def spec(rate):
        extra = {"eviction_policy": "rate"}
        if rate is not None:
            extra["rate"] = rate
        return CPUOffloadingSpec(_make_offloading_config(extra_config=extra))

    with pytest.raises(ValueError):
        spec(None)  # missing params
    with pytest.raises(ValueError):
        spec({"params": PARAMS, "chunk_tokens": 256})  # unknown nested key
    with pytest.raises(ValueError):
        spec({"params": PARAMS, "running_owner": "bogus"})
    s = spec({"params": PARAMS})
    mgr = s.get_manager()
    assert type(mgr._policy).__name__ == "RateCachePolicy"
    # the L2 spec sets running_owner = "decode" explicitly (F9)
    assert mgr._policy.cfg.running_owner == "decode"
    assert s.get_manager() is mgr
    assert spec({"params": PARAMS, "running_owner": "protect"}).get_manager()._policy.cfg.running_owner == "protect"


# ---------------------------------------------------------------- T1 rate states


def test_t1_rate_states_hand_derived():
    clk = Clock()
    pol = make_policy(clock=clk)
    q = 0.75
    # prefill not done: protected
    pol.on_request_arrival(ctx("a1", "sa"))
    assert pol._reader_rate("sa", 0.0) == 1e6
    # decode, elapsed e=2: q / max(MRL_dec(2) + mean(G), floor) = .75/(1+5)
    clk.t = 0.0
    pol.on_prefill_done(ctx("a1", "sa"))
    assert pol._reader_rate("sa", 2.0) == pytest.approx(q / 6.0)
    # decode, elapsed 0: MRL_dec(0) = mean = 2 -> .75 / 7
    assert pol._reader_rate("sa", 0.0) == pytest.approx(q / 7.0)
    # finished 3 s ago: S=q*surv(3)=.75*.75; p=S/(S+.25); MRL_tool(3)=3
    clk.t = 4.0
    pol.on_request_finished((), set(), set(), ctx("a1", "sa"))
    st = q * 0.75
    p = st / (st + 0.25)
    assert pol._reader_rate("sa", 7.0) == pytest.approx(p / 3.0)
    # successor queued: the latest request (prefill not done) protects
    clk.t = 8.0
    pol.on_request_arrival(ctx("a2", "sa"))
    assert pol._reader_rate("sa", 8.0) == 1e6
    # no record at all: mean_g 1/max(g, floor)
    assert pol._reader_rate("unknown", 8.0) == pytest.approx(
        (1 / 2 + 1 / 4 + 1 / 6 + 1 / 8) / 4
    )


def test_t1_no_dead_state_and_running_owner_protect():
    clk = Clock()
    pol = make_policy(clock=clk)
    for rid in ("a1", "a2"):
        pol.on_request_arrival(ctx(rid, "sa"))
        pol.on_prefill_done(ctx(rid, "sa"))
    # the GPU policy would call a1 dead (successor through prefill); here the
    # session's latest request (a2, decoding) decides: no dead state
    assert pol._reader_rate("sa", 2.0) == pytest.approx(0.75 / 6.0)
    prot = make_policy(clock=clk, running_owner="protect")
    prot.on_request_arrival(ctx("b1", "sb"))
    prot.on_prefill_done(ctx("b1", "sb"))
    assert prot._reader_rate("sb", 2.0) == 1e6


def test_t1_chunk_with_two_reader_sessions_sums_and_newer_request_replaces():
    clk = Clock()
    pol = make_policy(clock=clk)
    ca, cb = ctx("a1", "sa"), ctx("b1", "sb")
    pol.on_request_arrival(ca)
    pol.on_request_arrival(cb)
    ka, _ = put_chain(pol, ca, [1, 2, 3])
    finish(pol, ca, [ka], clk, 1.0)
    # b1 reads the same prefix chunks 1,2 (loaded)
    adopt(cb, ca, ka[:2])
    finish(pol, cb, [ka[:2]], clk, 1.0)
    set_rates(pol, {"sa": 2.0, "sb": 3.0})
    # evicting chunk 1 (front, sole F): both sessions lose 136 of work
    assert pol.node_action_scores_for_test(ka[2])[0] == pytest.approx(5.0 * 136)
    # sa's newer request reads only chunk 3: sa no longer reads 1, 2
    pol.on_request_arrival(ctx("a2", "sa"))
    a2 = ctx("a2", "sa")
    adopt(a2, ca, [ka[2]])
    finish(pol, a2, [[ka[2]]], clk, 2.0)
    assert pol.node_action_scores_for_test(ka[2])[0] == pytest.approx(3.0 * 136)


# ---------------------------------------------------------------- T2-T4, T7: driver


def test_t2_fixture_a_front_first_scores():
    """[dep U2] one reader of rate 1 needs [0,64): F scores c0=136, c1=392,
    c2=648, c3=904 (cost_b1=.5, cost_b2=1); T is never cheaper per chunk."""
    pol = make_policy()
    c = ctx("r", "s")
    pol.on_request_arrival(c)
    keys, _ = put_chain(pol, c, [1, 2, 3, 4])
    finish(pol, c, [keys])
    set_rates(pol, {"s": 1.0})
    order = []
    for expect in (136.0, 392.0, 648.0, 904.0):
        ((k, _),) = pol.evict(1, set())
        order.append(k)
        assert pol.last_victim_score == pytest.approx(expect)
    assert order == keys
    assert pol.evict(1, set()) is None


def test_t3_fixture_b_tail_cheaper_than_front():
    """[dep U2] readers 0.01 (needs [0,64)) and 1 (needs [0,16)): score(F)=
    137.36, score(T)=39.2: T is exposed and all four chunks leave at once; with
    n=1 the overshoot is 3."""
    pol = make_policy()
    ca, cb = ctx("ra", "sa"), ctx("rb", "sb")
    pol.on_request_arrival(ca)
    pol.on_request_arrival(cb)
    keys, _ = put_chain(pol, ca, [1, 2, 3, 4])
    finish(pol, ca, [keys])
    adopt(cb, ca, keys[:1])
    finish(pol, cb, [keys[:1]])
    set_rates(pol, {"sa": 0.01, "sb": 1.0})
    f, t = pol.node_action_scores_for_test(keys[-1])
    assert f == pytest.approx(137.36)
    assert t == pytest.approx(39.2)
    evicted = pol.evict(1, set())
    assert [k for k, _ in evicted] == keys
    assert pol.get(keys[0]) is None


def test_t4_fixture_c_veto_and_none_leaves_index_unchanged():
    pol = make_policy()
    ca, cb = ctx("ra", "sa"), ctx("rb", "sb")
    pol.on_request_arrival(ca)
    pol.on_request_arrival(cb)
    keys, chunks = put_chain(pol, ca, [1, 2, 3, 4])
    finish(pol, ca, [keys])
    adopt(cb, ca, keys[:1])
    finish(pol, cb, [keys[:1]])
    set_rates(pol, {"sa": 0.01, "sb": 1.0})
    chunks[keys[2]].ref_cnt = 1  # c2 is being loaded: T is vetoed
    ((k, _),) = pol.evict(1, set())
    assert k == keys[0]
    assert pol.last_victim_score == pytest.approx(137.36)
    # c0 protected too (a fresh node): nothing eligible at the front
    pol2 = make_policy()
    c2 = ctx("r2", "s2")
    pol2.on_request_arrival(c2)
    keys2, chunks2 = put_chain(pol2, c2, [11, 12, 13])
    finish(pol2, c2, [keys2])
    set_rates(pol2, {"s2": 1.0})
    before = pol2.snapshot_for_test()
    assert pol2.evict(1, {keys2[0]}) is None
    assert pol2.snapshot_for_test() == before
    # a request needing two while only one front-eligible exists fails atomically
    chunks2[keys2[0]].ref_cnt = 1
    assert pol2.evict(1, set()) is None
    assert pol2.snapshot_for_test() == before


def _t7_node():
    pol = make_policy(kinds=("full", "mamba", "swa"))
    ca, cb = ctx("ra", "sa"), ctx("rb", "sb")
    pol.on_request_arrival(ca)
    pol.on_request_arrival(cb)
    kf, _ = put_chain(pol, ca, [1, 2, 3])
    km, chm = put_chain(pol, ca, [21], g=1, pos=[48])
    ks, chs = put_chain(pol, ca, [31], g=2, pos=[48])
    finish(pol, ca, [kf, km, ks])
    adopt(cb, ca, kf[:1])
    finish(pol, cb, [kf[:1]])  # a hot reader that needs only the front chunk
    # the cold reader (needs everything) makes T cheaper per freed chunk:
    # F = 136 + 0.001*136, T = (136 + 0.001*1176) / 5
    set_rates(pol, {"sa": 0.001, "sb": 1.0})
    return pol, kf, km, ks, chm, chs


def test_t7_hybrid_state_chunks_leave_only_inside_a_tail_action():
    """[dep U2] a node with Full chunks, a Mamba key and an SWA key at its end:
    the state keys leave together, only inside T; a veto keeps them."""
    pol, kf, km, ks, chm, chs = _t7_node()
    assert pol.node_layout_for_test() == {
        kf[2]: (tuple(kf), tuple(sorted(km + ks)), 0, 48)
    }
    f, t = pol.node_action_scores_for_test(kf[2])
    assert f == pytest.approx(136.136)
    assert t == pytest.approx((136 + 0.001 * 1176) / 5)
    ev = pol.evict(1, set())
    assert sorted(k for k, _ in ev) == sorted(kf + km + ks)  # overshoot of 4
    # T vetoed by a loading SWA chunk: only Full chunks go, front first
    pol, kf, km, ks, chm, chs = _t7_node()
    chs[ks[0]].ref_cnt = 1
    ev = pol.evict(3, set())
    assert [k for k, _ in ev] == kf
    assert pol.get(km[0]) is not None and pol.get(ks[0]) is not None
    assert pol.evict(1, set()) is None  # state chunks never leave alone
    chs[ks[0]].ref_cnt = 0
    ev = pol.evict(1, set())
    assert sorted(k for k, _ in ev) == sorted(km + ks)  # the state pair together


# ---------------------------------------------------------------- T6 tree


def test_t6_tree_branch_state_end_leaf_and_middle_eviction():
    """[dep U2] r1 stores c1..c4; r2 shares c1,c2 then diverges (c3x, c4x); a
    Mamba checkpoint of r1 sits at 32 (end of c2 anyway) and r3 has one at c1."""
    pol = make_policy(kinds=("full", "mamba"))
    c1, c2 = ctx("r1", "s1"), ctx("r2", "s2")
    pol.on_request_arrival(c1)
    pol.on_request_arrival(c2)
    k1, ch1 = put_chain(pol, c1, [1, 2, 3, 4])
    finish(pol, c1, [k1])
    # r2: c1, c2 are L2 hits (loaded), c3x, c4x are new
    kx, _ = put_chain(pol, c2, [3 + 100, 4 + 100], start_pos=48)
    adopt(c2, c1, k1[:2])
    finish(pol, c2, [k1[:2] + kx])
    lay = pol.node_layout_for_test()
    assert lay == {
        k1[1]: ((k1[0], k1[1]), (), 0, 32),  # branch node
        k1[3]: ((k1[2], k1[3]), (), 32, 64),
        kx[1]: ((kx[0], kx[1]), (), 32, 64),
    }
    # a middle Full key is evicted: no merge, the node keeps its remaining keys
    pol.remove(k1[2])
    lay = pol.node_layout_for_test()
    assert lay[k1[3]] == ((k1[3],), (), 32, 64)
    assert lay[k1[1]][0] == (k1[0], k1[1])


def test_t6_state_key_makes_a_node_end():
    pol = make_policy(kinds=("full", "mamba"))
    c = ctx("r", "s")
    pol.on_request_arrival(c)
    kf, _ = put_chain(pol, c, [1, 2, 3, 4])
    km, _ = put_chain(pol, c, [21], g=1, pos=[32])
    finish(pol, c, [kf, km])
    lay = pol.node_layout_for_test()
    assert lay == {
        kf[1]: ((kf[0], kf[1]), tuple(km), 0, 32),
        kf[3]: ((kf[2], kf[3]), (), 32, 64),
    }


def test_t6_orphan_parent_starts_a_rootless_node():
    """A chunk whose parent is not in L2 (never stored) is a node of its own
    with positions staying absolute; a later insert of the parent does not
    merge them."""
    pol = make_policy()
    c = ctx("r", "s")
    pol.on_request_arrival(c)
    kb, _ = put_chain(pol, c, [3, 4], start_pos=48)  # c1, c2 never stored
    finish(pol, c, [kb])
    assert pol.node_layout_for_test() == {kb[1]: ((kb[0], kb[1]), (), 32, 64)}
    c2 = ctx("r2", "s2")
    pol.on_request_arrival(c2)
    ka, _ = put_chain(pol, c2, [1, 2], start_pos=16)
    finish(pol, c2, [ka])
    # the earlier link decision is not rewritten: no merge
    assert pol.node_layout_for_test() == {
        kb[1]: ((kb[0], kb[1]), (), 32, 64),
        ka[1]: ((ka[0], ka[1]), (), 0, 32),
    }


# ---------------------------------------------------------------- T5 manager integration


def make_mgr(n):
    mgr = CPUOffloadingManager(
        num_chunks=n, cache_policy="rate", policy_config=rate_cfg()
    )
    mgr._policy.clock = Clock()
    return mgr


def test_t5_manager_pinned_and_pending_chunks_are_never_evicted():
    mgr = make_mgr(3)
    ks = [key(i) for i in (1, 2, 3)]
    for n, k in enumerate(ks):  # three requests, three one-chunk nodes
        c = ctx(f"r{n}", f"s{n}")
        mgr.on_new_request(c)
        c.set_offload_key_position(k, T)
        assert mgr.prepare_store([k], c).keys_to_store == [k]
        mgr.complete_store([k], c)
        mgr.on_request_finished(c)
    # a load pins chunk 1, the oldest (tie-break by insertion order): c2 goes
    cb = ctx("rb", "sb")
    mgr.on_new_request(cb)
    mgr.prepare_load([ks[0]], cb)
    cb.set_offload_key_position(key(9), T)
    out = mgr.prepare_store([key(9)], cb)
    assert out is not None and out.evicted_keys == [ks[1]]
    # now c3 is the only idle chunk; the pending new chunk and the pinned one stay
    cb.set_offload_key_position(key(10), 2 * T)
    out = mgr.prepare_store([key(10)], cb)
    assert out is not None and out.evicted_keys == [ks[2]]
    cb.set_offload_key_position(key(11), 3 * T)
    assert mgr.prepare_store([key(11)], cb) is None
    assert mgr.lookup(ks[0], cb).name == "HIT"
    assert mgr.lookup(key(9), cb).name == "HIT_PENDING"


def test_t5_failed_store_then_retry_keeps_pending_and_single_miss_observation():
    mgr = make_mgr(2)
    pol = mgr._policy
    c = ctx("r", "s")
    mgr.on_new_request(c)
    k = key(1)
    c.set_offload_key_position(k, T)
    out = mgr.prepare_store([k], c)
    assert out.keys_to_store == [k]
    mgr.complete_store([k], c, success=False)
    assert pol.get(k) is None
    out = mgr.prepare_store([k], c)  # retry: on_store_miss only once per request
    assert out.keys_to_store == [k]
    mgr.complete_store([k], c)
    assert pol.get(k) is not None and pol.get(k).is_ready
    # the pending metadata survived the failed store: position T is known
    mgr.on_request_finished(c)
    assert pol.node_layout_for_test() == {k: ((k,), (), 0, T)}


def test_t5_mamba_single_key_handoff_store():
    """A Mamba boundary key stored on its own (no Full parent offered) joins
    its node by position once the request finishes."""
    mgr = CPUOffloadingManager(num_chunks=4, cache_policy="rate", policy_config=rate_cfg())
    mgr.set_group_kinds(["full", "mamba"], 1)
    c = ctx("r", "s")
    mgr.on_new_request(c)
    kf = [key(1), key(2)]
    for n, k in enumerate(kf):
        c.set_offload_key_position(k, (n + 1) * T)
    mgr.prepare_store(kf, c)
    mgr.complete_store(kf, c)
    km = key(21, 1)
    c.set_offload_key_position(km, 32)
    mgr.prepare_store([km], c)
    mgr.complete_store([km], c)
    mgr.on_request_finished(c)
    lay = mgr._policy.node_layout_for_test()
    assert lay == {kf[1]: (tuple(kf), (km,), 0, 32)}


# ---------------------------------------------------------------- T17 per instance parameters


def test_t17_two_instances_with_different_cost_params_score_differently():
    """Heterogeneous serving: one policy per engine, per-instance cost_b1 /
    cost_b2: the same single-chunk node scores b1*16 + b2*128 in each."""

    def front_score(b1, b2):
        pol = make_policy(cost_b1=b1, cost_b2=b2)
        c = ctx("r", "s")
        pol.on_request_arrival(c)
        keys, _ = put_chain(pol, c, [1, 2])
        finish(pol, c, [keys])
        set_rates(pol, {"s": 1.0})
        pol.evict(1, set())
        return pol.last_victim_score

    assert front_score(1.0, 0.0) == pytest.approx(16.0)
    assert front_score(0.0, 1.0) == pytest.approx(128.0)


# ---------------------------------------------------------------- T13 [dep Q5] update rule


def test_t13_clear_keeps_sessions_and_drops_nodes():
    """[dep Q5] placeholder table 5.1: clear() drops the node index, sessions
    stay."""
    clk = Clock()
    pol = make_policy(clock=clk)
    c = ctx("r", "s")
    pol.on_request_arrival(c)
    k, _ = put_chain(pol, c, [1, 2])
    finish(pol, c, [k])
    pol.clear()
    assert pol.node_layout_for_test() == {}
    assert pol._reader_rate("s", 5.0) != 1e6 and pol.sessions.reqs
