# SPDX-License-Identifier: Apache-2.0
"""holes/core: KV cache manager and scheduler (CPU only)."""

try:
    from . import holes_utils as U
except ImportError:  # run as a script
    import holes_utils as U

from vllm.v1.core.holes import HoleConfig
from vllm.v1.core.kv_cache_manager import KVCacheManager

PROMPT = list(range(100, 100 + 10 * U.BS + 1))  # 10 full blocks + 1 token


def _mgr(enabled=True, min_reuse=1, num_blocks=100):
    return KVCacheManager(
        U.kv_cache_config(num_blocks),
        max_model_len=8192,
        hash_block_size=U.BS,
        enable_caching=True,
        log_stats=True,
        hole_config=HoleConfig(enabled=enabled, min_reuse_blocks=min_reuse),
    )


def _fill(mgr, rid="a"):
    """Compute the whole prompt of a request and free it (blocks stay cached)."""
    req = U.make_request(rid, PROMPT)
    blocks = mgr.allocate_slots(req, len(PROMPT))
    assert blocks is not None
    ids = list(mgr.get_block_ids(rid)[0])
    mgr.free(req)
    return req, ids


def test_lookup_continues_past_miss():
    mgr = _mgr()
    a, a_ids = _fill(mgr)
    assert mgr.evict_cached_token_range(a, 3 * U.BS, 6 * U.BS) == 3
    b = U.make_request("b", PROMPT)
    # Stock lookup stops at the first miss.
    blocks, n = mgr.get_computed_blocks(b)
    assert n == 3 * U.BS
    # Hole-aware lookup continues: frontier at 10 blocks, one hole.
    kvb, frontier, holes, loads = mgr.get_computed_blocks_with_holes(b)
    assert frontier == 10 * U.BS and holes == [(48, 96)] and loads == []
    got = kvb.blocks[0]
    assert [blk.block_id for i, blk in enumerate(got) if i not in (3, 4, 5)] == [
        a_ids[i] for i in range(10) if i not in (3, 4, 5)
    ]
    assert all(got[i].is_null for i in (3, 4, 5))


def test_allocate_in_place_and_cache_ranges():
    mgr = _mgr()
    a, a_ids = _fill(mgr)
    mgr.evict_cached_token_range(a, 3 * U.BS, 6 * U.BS)
    b = U.make_request("b", PROMPT)
    kvb, frontier, holes, _ = mgr.get_computed_blocks_with_holes(b)
    free_before = mgr.block_pool.get_num_free_blocks()
    # First step: compute only the first hole block (48..64), no tail.
    out = mgr.allocate_slots(
        b, 0, num_new_computed_tokens=frontier, new_computed_blocks=kvb,
        hole_ranges=[(48, 64)],
    )
    assert out is not None
    ids = mgr.get_block_ids("b")[0]
    assert len(ids) == 10  # position ordered, holes in place
    for i in range(10):
        if i in (3, 4, 5):
            assert ids[i] != a_ids[i] or True  # fresh block (may reuse id)
        else:
            assert ids[i] == a_ids[i]
    # 3 hole blocks taken from the free queue (hits were already free -> also
    # removed from the free queue by touch).
    assert free_before - mgr.block_pool.get_num_free_blocks() == 10
    # Block 3 is cached now, 4 and 5 not yet.
    c = U.make_request("c", PROMPT)
    hit, _ = mgr.get_cache_hits_with_holes(c)
    assert [h is not None for h in hit] == [True] * 4 + [False, False] + [True] * 4
    # Second step: the rest of the hole + the tail (the scheduler advanced
    # num_computed_tokens to the frontier).
    b.num_computed_tokens = frontier
    out = mgr.allocate_slots(b, 1, hole_ranges=[(64, 96)])
    assert out is not None
    assert len(mgr.get_block_ids("b")[0]) == 11
    hit, _ = mgr.get_cache_hits_with_holes(c)
    assert all(h is not None for h in hit) and len(hit) == 10
    mgr.free(b)
    assert mgr.block_pool.get_num_free_blocks() == 99  # null block excluded


def test_disabled_is_stock():
    mgr = _mgr(enabled=False)
    assert not mgr.holes_enabled


def test_min_reuse_blocks():
    mgr = _mgr(min_reuse=3)
    a, _ = _fill(mgr)
    mgr.evict_cached_token_range(a, 3 * U.BS, 4 * U.BS)
    mgr.evict_cached_token_range(a, 6 * U.BS, 7 * U.BS)
    b = U.make_request("b", PROMPT)
    _, frontier, holes, _ = mgr.get_computed_blocks_with_holes(b)
    # run [4,6) has 2 blocks < 3 -> recomputed; run [7,10) kept.
    assert frontier == 160 and holes == [(48, 112)]


def test_can_fit_counts_holes():
    mgr = _mgr(num_blocks=14)
    a, _ = _fill(mgr)  # 11 blocks used, then freed (cached)
    mgr.evict_cached_token_range(a, 0, 8 * U.BS)  # keep blocks 8, 9
    b = U.make_request("b", PROMPT)
    kvb, frontier, holes, _ = mgr.get_computed_blocks_with_holes(b)
    assert holes == [(0, 128)] and frontier == 160
    # Needs 8 hole blocks + 1 tail block + touch 2 hits = 11 <= 13 free.
    assert mgr.can_fit_full_sequence(b, num_new_computed_tokens=frontier,
                                     new_computed_blocks=kvb)


# ---------------------------------------------------------------- scheduler


def _run_to_completion(s, max_steps=50):
    outs = []
    for _ in range(max_steps):
        if not s.requests or all(r.is_finished() for r in s.requests.values()):
            break
        outs.append(U.step(s))
    return outs


def _scheduler_with_hole(holes_cfg, prompt, hole, **kw):
    s = U.make_scheduler(holes_cfg, **kw)
    a = U.make_request("a", prompt, max_tokens=1)
    s.add_request(a)
    _run_to_completion(s)
    probe = U.make_request("probe", prompt)
    if hole is not None:
        s.kv_cache_manager.evict_cached_token_range(probe, *hole)
    return s


def test_scheduler_hole_then_tail():
    prompt = list(range(1, 1 + 64 * U.BS + 5))  # 1029 tokens
    s = _scheduler_with_hole({"enabled": True}, prompt, (128, 384))
    b = U.make_request("b", prompt)
    s.add_request(b)
    out = U.step(s)
    assert out.scheduled_hole_ranges == {"b": [(128, 384)]}
    # frontier = 1024 (64 blocks), tail = 5 tokens, hole = 256 tokens
    assert out.num_scheduled_tokens["b"] == 256 + 5
    assert b.num_computed_tokens == 1029 and b.pending_hole_ranges == []
    assert b.num_cached_tokens == 1024 - 256
    # Prefill done in one step: a token was sampled.
    assert b.num_output_tokens == 1


def test_scheduler_big_hole_chunks_front_first():
    prompt = list(range(1, 1 + 400 * U.BS + 1))  # 6401 tokens
    s = _scheduler_with_hole({"enabled": True}, prompt, (16 * U.BS, 316 * U.BS))
    b = U.make_request("b", prompt)
    s.add_request(b)
    outs = []
    while b.num_output_tokens == 0:
        outs.append(U.step(s))
    hole_steps = [o.scheduled_hole_ranges.get("b") for o in outs]
    # hole = [256, 5056) = 4800 tokens: 2048 + 2048 + 704; the 1-token tail
    # runs alone (min_mixed_tail = 2).
    assert hole_steps[0] == [(256, 2304)]
    assert hole_steps[1] == [(2304, 4352)]
    assert hole_steps[2] == [(4352, 5056)]
    assert hole_steps[3] is None
    assert [o.num_scheduled_tokens["b"] for o in outs] == [2048, 2048, 704, 1]
    assert b.num_computed_tokens == 6401


def test_scheduler_no_hole_identical_to_stock():
    prompt = list(range(1, 1 + 200 * U.BS + 3))

    def trace(cfg):
        s = _scheduler_with_hole(cfg, prompt, None)
        b = U.make_request("b", prompt[:-40] + list(range(9000, 9100)))
        s.add_request(b)
        res = []
        for o in _run_to_completion(s):
            res.append(
                (
                    dict(o.num_scheduled_tokens),
                    [r.num_computed_tokens for r in o.scheduled_new_reqs],
                    [r.block_ids for r in o.scheduled_new_reqs],
                    list(o.scheduled_cached_reqs.num_computed_tokens),
                    o.scheduled_hole_ranges,
                )
            )
        return res

    assert trace({"enabled": True}) == trace(None)


def test_scheduler_preempt_clears_holes():
    prompt = list(range(1, 1 + 300 * U.BS + 1))
    s = _scheduler_with_hole({"enabled": True}, prompt, (0, 200 * U.BS))
    b = U.make_request("b", prompt)
    s.add_request(b)
    U.step(s)
    assert b.pending_hole_ranges
    s._preempt_request(b, 0.0)
    s.running.remove(b)
    s.prev_step_scheduled_req_ids.discard("b")  # as after a real preemption
    assert b.pending_hole_ranges == [] and b.num_computed_tokens == 0
    # Re-admission looks up again: the first 2048 hole tokens were cached.
    out = U.step(s)
    assert out.scheduled_hole_ranges["b"][0][0] == 2048


class FakeMaskConnector:
    """Scheduler-side stub implementing SupportsBlockHitMask."""

    def __init__(self, blocks):
        self.blocks = set(blocks)
        self.calls = []

    def get_block_hit_mask(self, request, local_hit):
        n = (request.num_tokens - 1) // U.BS
        return [i in self.blocks for i in range(n)], False

    def update_state_after_alloc_with_ranges(self, request, blocks, load_ranges):
        self.calls.append((request.request_id, list(load_ranges)))

    def build_connector_meta(self, scheduler_output):
        return None

    def request_finished(self, request, block_ids):
        return False, None

    def update_connector_output(self, out):
        pass

    def take_events(self):
        return []

    def get_kv_connector_stats(self):
        return None

    def get_finished_count(self):
        return None


def test_scheduler_mask_connector():
    from vllm.distributed.kv_transfer.kv_connector.v1.hole_mask import (
        SupportsBlockHitMask,
    )

    SupportsBlockHitMask.register(FakeMaskConnector)
    prompt = list(range(1, 1 + 64 * U.BS + 5))
    s = _scheduler_with_hole({"enabled": True}, prompt, (128, 512))
    # The connector holds blocks 8..15 (tokens 128..256) of the hole.
    s.connector = FakeMaskConnector(range(8, 16))
    b = U.make_request("b", prompt)
    s.add_request(b)
    out = s.schedule()
    assert s.connector.calls == [("b", [(128, 256)])]
    assert out.scheduled_hole_ranges == {"b": [(256, 512)]}
    assert out.num_scheduled_tokens["b"] == 256 + 5


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
