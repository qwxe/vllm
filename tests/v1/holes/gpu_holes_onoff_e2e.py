# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes GPU end-to-end check: two engines, holes off vs holes on.

Runs the same deterministic multi-turn workload twice, each arm in its own
child process (one GPU engine at a time):

* ``off``: holes disabled, stock vLLM prefix caching enabled (LRU eviction);
  it reuses only the cached prefix up to the first evicted block and
  recomputes everything after it;
* ``on``: holes enabled with the rate eviction policy (holes take effect only
  with the rate policy), so evicted middle/front chunks of a cached prefix are
  recomputed as holes while the blocks after them are reused.

Both arms use the same small KV pool (``--num-blocks``) so that eviction, and
in the ``on`` arm holes, actually occur. With the defaults (2 sessions x 6
turns, 240 blocks) the two sessions' contexts outgrow the pool from turn 3
on, so part of each session's prefix is evicted while the other runs (with 4
sessions almost the whole prefix was evicted and holes occurred only
sometimes). Requests run one at a time in a fixed
round-robin order over sessions, greedy decoding, bf16 by default. The check
is exact token equality of every request's output, holes off vs on, plus a
non-vacuity check: the ``on`` arm must have scheduled hole ranges, and at
least one holed request must have cached > 0 (blocks after the hole reused).

Usage: HOLES_MODEL=<path> python gpu_holes_onoff_e2e.py [--eager] [--num-blocks N]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("HOLES_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Rate-policy parameters (inline, seconds). The workload runs back to back,
# so idle times are short; the values only rank the cached chunks.
PARAMS = {
    "tool_gaps": [0.5, 1.0, 2.0, 4.0, 8.0],
    "decode_durs": [1.0],
    "p_end": 0.25,
    "floor": 1.0,
}


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["off", "on"], default=None,
                    help="internal: run one arm and write --out")
    ap.add_argument("--out", default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--num-blocks", type=int, default=240)
    ap.add_argument("--max-model-len", type=int, default=3584)
    ap.add_argument("--sessions", type=int, default=2)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--first", type=int, default=1500)
    ap.add_argument("--new", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=24)
    return ap.parse_args()


def run_arm(args):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    import vllm

    print("vllm from", vllm.__file__, flush=True)
    add = {}
    if args.arm == "on":
        add = {"holes": {"enabled": True},
               "eviction": {"policy": "rate", "params": PARAMS}}
    llm = LLM(
        MODEL,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enforce_eager=args.eager,
        seed=0,
        num_gpu_blocks_override=args.num_blocks,
        additional_config=add,
    )
    sched = llm.llm_engine.engine_core.engine_core.scheduler
    holes_on = bool(getattr(sched, "holes_enabled", False))
    assert holes_on == (args.arm == "on"), f"holes_enabled={holes_on} in arm {args.arm}"
    pol = getattr(sched, "eviction_policy", None)
    print(f"arm={args.arm} holes_enabled={holes_on} policy={type(pol).__name__} "
          f"blocks={sched.kv_cache_manager.block_pool.num_gpu_blocks}", flush=True)

    counters = {"steps": 0, "hole_steps": 0, "hole_tokens": 0}
    hole_reqs = set()
    orig_update = sched.update_from_output

    def update(so, mro):
        counters["steps"] += 1
        for rid, rs in (getattr(so, "scheduled_hole_ranges", None) or {}).items():
            if rs:
                counters["hole_steps"] += 1
                counters["hole_tokens"] += sum(e - s for s, e in rs)
                hole_reqs.add(rid)
        return orig_update(so, mro)

    sched.update_from_output = update

    # Deterministic natural text (Zen of Python, rot13-decoded), as in
    # gpu_holes_e2e.py; each session and turn reads its own offset.
    tok = llm.get_tokenizer()
    import this  # noqa: F401
    zen = "".join({**{chr(i + 65): chr((i + 13) % 26 + 65) for i in range(26)},
                   **{chr(i + 97): chr((i + 13) % 26 + 97) for i in range(26)}}.get(c, c)
                  for c in this.s)
    paras = [f"Section {k}. " + zen.splitlines()[2 + k % 19] + f" (note {k * 7 % 13})\n"
             for k in range(1200)]
    text_ids = tok.encode("".join(paras))

    def piece(n, seed):
        off = (seed * 977) % (len(text_ids) - n - 1)
        return text_ids[off: off + n]

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True)
    ctx = {s: [] for s in range(args.sessions)}
    recs = []
    for k in range(args.turns):
        for s in range(args.sessions):
            n = args.first if k == 0 else args.new
            prompt = ctx[s] + piece(n, 31 * s + k + 1)
            rid_before = set(hole_reqs)
            sp_s = sp.clone()
            sp_s.extra_args = {"session_id": f"sess{s}"}
            o = llm.generate([TokensPrompt(prompt_token_ids=prompt)], sp_s, use_tqdm=False)[0]
            out = list(o.outputs[0].token_ids)
            recs.append({"session": s, "turn": k, "prompt_len": len(prompt),
                         "cached": o.num_cached_tokens or 0, "out": out,
                         "holed": len(hole_reqs) > len(rid_before)})
            ctx[s] = prompt + out
    res = {"arm": args.arm, "counters": counters, "hole_requests": len(hole_reqs),
           "policy_stats": pol.stats() if pol is not None and hasattr(pol, "stats") else None,
           "requests": recs}
    with open(args.out, "w") as f:
        json.dump(res, f)
    print(f"arm={args.arm} done counters={counters} hole_requests={len(hole_reqs)} "
          f"policy={res['policy_stats']}", flush=True)
    return 0


def main():
    args = build_args()
    if args.arm is not None:
        return run_arm(args)
    tmp = tempfile.mkdtemp(prefix="holes_onoff_")
    res = {}
    for arm in ("off", "on"):
        out = os.path.join(tmp, f"arm_{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm, "--out", out,
               "--gpu-util", str(args.gpu_util), "--dtype", args.dtype,
               "--num-blocks", str(args.num_blocks),
               "--max-model-len", str(args.max_model_len), "--sessions", str(args.sessions),
               "--turns", str(args.turns), "--first", str(args.first),
               "--new", str(args.new), "--max-tokens", str(args.max_tokens)]
        if args.eager:
            cmd.append("--eager")
        print("running", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"arm {arm} failed rc={rc}")
            print("RESULT FAIL")
            return 1
        with open(out) as f:
            res[arm] = json.load(f)

    off, on = res["off"]["requests"], res["on"]["requests"]
    assert len(off) == len(on)
    same = 0
    print("\nSUMMARY session turn prompt_len cached(off) cached(on) holed(on) tokens_equal")
    for a, b in zip(off, on):
        eq = a["out"] == b["out"] and a["prompt_len"] == b["prompt_len"]
        same += eq
        print(f"   {a['session']} {a['turn']} {a['prompt_len']} {a['cached']} {b['cached']} "
              f"{b['holed']} {eq}")
    c = res["on"]["counters"]
    print(f"tokens_equal {same}/{len(off)}; on-arm hole_requests={res['on']['hole_requests']} "
          f"hole_steps={c['hole_steps']} hole_tokens={c['hole_tokens']}; "
          f"off-arm hole_tokens={res['off']['counters']['hole_tokens']}; "
          f"on-arm policy={res['on']['policy_stats']}")
    ok = same == len(off)
    if c["hole_tokens"] == 0:
        print("no holes occurred in the on arm: comparison is vacuous")
        ok = False
    holed_cached = [r["cached"] for r in on if r["holed"]]
    print(f"on-arm holed requests cached tokens: {holed_cached}")
    if not any(x > 0 for x in holed_cached):
        print("no holed request reused cached blocks after its hole")
        ok = False
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    t = time.time()
    rc = main()
    print(f"elapsed {time.time() - t:.1f}s")
    sys.exit(rc)
