# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes at L2 (CPU OffloadingConnector, rate policy) GPU end-to-end check.

L note ``T20261028-fork-l2-rate`` section 10, stages 1-4 (single group,
Qwen2.5-0.5B-Instruct). The workload is SYNTHETIC (seeded multi-turn sessions
of random token ids; no claim about real traces). Four arms, each in its own
child process, one GPU engine at a time, requests one at a time (interleaved
sessions), greedy decoding, bf16:

* A reference: large GPU pool, no connector, holes off. It builds the prompts
  (every turn = previous prompt + A's output + new random ids) and records
  per-request wall durations (the ``decode_durs`` calibration of the params).
* B L2 offload with ``lru``, holes off.
* C L2 offload with ``rate`` ("ours", L3-blind), holes on, small GPU pool,
  ``cpu_bytes_to_use`` about 1/3 of the workload's offloadable chunks.
* D L2 offload with the shipped ``arc`` policy, holes off.

B, C and D replay A's prompts (teacher forcing), so every request is compared
with A on its own. Pass for exactness: (C) token ids equal A per request; L2
evictions > 0; at least one admission with an L2 range loaded and a recomputed
hole. A near-tie divergence is checked with logprobs by hand and is not a
pass. No speed claim.

Usage: HOLES_MODEL=<path> python gpu_l2_e2e.py [--sessions N] [--eager]
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

MODEL = os.environ.get("HOLES_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
BLOCK = 16


def plan_workload(
    seed: int,
    n_sessions: int = 24,
    concurrency: int = 6,
    max_turns: int = 8,
    p_end: float = 1 / 8,
) -> dict:
    """Seeded synthetic plan: the request order (sessions interleaved, at most
    ``concurrency`` open), the new-token count of every turn, and the
    inter-turn gaps (lognormal, mu = ln 2 s, sigma = 0.7) that form
    ``tool_gaps``. A session ends after each turn with probability ``p_end``
    and after ``max_turns``."""
    rng = random.Random(seed)
    n_turns = {}
    for s in range(n_sessions):
        n = 1
        while n < max_turns and rng.random() >= p_end:
            n += 1
        n_turns[s] = n
    pending = list(range(n_sessions))
    active: list[int] = []
    issued = {s: 0 for s in range(n_sessions)}
    reqs = []
    while pending or active:
        while pending and len(active) < concurrency:
            active.append(pending.pop(0))
        for s in list(active):
            t = issued[s]
            reqs.append(
                {
                    "session": s,
                    "turn": t,
                    "new_tokens": rng.randint(256, 1024) if t == 0 else rng.randint(64, 256),
                }
            )
            issued[s] = t + 1
            if issued[s] >= n_turns[s]:
                active.remove(s)
    gaps = [rng.lognormvariate(math.log(2.0), 0.7) for _ in range(len(reqs) - n_sessions)]
    return {
        "seed": seed,
        "synthetic": True,
        "requests": reqs,
        "tool_gaps": gaps,
        "p_end": p_end,
    }


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], default=None, help="internal")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref", default=None, help="internal: arm A json")
    ap.add_argument("--params", default=None, help="internal: rate params json")
    ap.add_argument("--blocks", type=int, default=0, help="internal: GPU blocks")
    ap.add_argument("--cpu-bytes", type=int, default=0, help="internal")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--ref-blocks", type=int, default=6000)
    ap.add_argument("--gap-scale", type=float, default=0.05,
                    help="tool gaps are scaled to the wall time of the run")
    return ap.parse_args()


def run_arm(args):
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    plan = plan_workload(args.seed, args.sessions, args.concurrency, args.max_turns)
    add = {}
    ktc = None
    if args.arm != "A":
        extra = {"cpu_bytes_to_use": args.cpu_bytes}
        if args.arm == "C":
            extra["eviction_policy"] = "rate"
            extra["rate"] = {"params": args.params}
            add = {"holes": {"enabled": True}}
        elif args.arm == "D":
            extra["eviction_policy"] = "arc"
        ktc = KVTransferConfig(
            kv_connector="OffloadingConnector", kv_role="kv_both",
            kv_connector_extra_config=extra,
        )
    llm = LLM(
        MODEL, dtype=args.dtype, max_model_len=args.max_model_len,
        max_num_batched_tokens=2048, max_num_seqs=1,
        gpu_memory_utilization=args.gpu_util, enable_prefix_caching=True,
        enforce_eager=args.eager, seed=0, num_gpu_blocks_override=args.blocks,
        additional_config=add, kv_transfer_config=ktc,
    )
    sched = llm.llm_engine.engine_core.engine_core.scheduler
    holes_on = bool(getattr(sched, "holes_enabled", False))
    assert holes_on == (args.arm == "C"), f"holes_enabled={holes_on} in arm {args.arm}"
    counters = {"hole_steps": 0, "hole_tokens": 0, "l2_evicted": 0, "loaded_hole_admissions": 0}
    orig_update = sched.update_from_output

    def update(so, mro):
        for rs in (getattr(so, "scheduled_hole_ranges", None) or {}).values():
            if rs:
                counters["hole_steps"] += 1
                counters["hole_tokens"] += sum(e - s for s, e in rs)
        return orig_update(so, mro)

    sched.update_from_output = update
    if ktc is not None:
        mgr = sched.connector.connector_scheduler.manager
        orig_evict = mgr._policy.evict

        def evict(n, protected):
            out = orig_evict(n, protected)
            counters["l2_evicted"] += len(out) if out else 0
            return out

        mgr._policy.evict = evict
    if args.arm == "C":
        orig_alloc = sched.connector.update_state_after_alloc_with_ranges

        def alloc(request, blocks, load_ranges):
            if load_ranges and request.pending_hole_ranges:
                counters["loaded_hole_admissions"] += 1
            return orig_alloc(request, blocks, load_ranges)

        sched.connector.update_state_after_alloc_with_ranges = alloc

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True, logprobs=2)
    rng = random.Random(args.seed + 1)
    recs = []
    ctx: dict[int, list[int]] = {}
    ref = None
    if args.ref:
        with open(args.ref) as f:
            ref = json.load(f)["requests"]
    for i, r in enumerate(plan["requests"]):
        s = r["session"]
        if ref is not None:
            prompt = ref[i]["prompt"]  # teacher forcing: A's context
        else:
            prompt = ctx.get(s, []) + [rng.randrange(1000, 30000) for _ in range(r["new_tokens"])]
        sp_s = sp.clone()
        sp_s.extra_args = {"session_id": f"sess{s}"}
        t0 = time.time()
        o = llm.generate([TokensPrompt(prompt_token_ids=prompt)], sp_s, use_tqdm=False)[0]
        dur = time.time() - t0
        out = list(o.outputs[0].token_ids)
        gaps = []  # top-1 minus top-2 logprob of every generated position
        for lp in o.outputs[0].logprobs or []:
            v = sorted((x.logprob for x in lp.values()), reverse=True)
            gaps.append(v[0] - v[1] if len(v) > 1 else float("inf"))
        recs.append({"session": s, "turn": r["turn"], "prompt": prompt, "out": out,
                     "gaps": gaps, "dur": dur, "cached": o.num_cached_tokens or 0})
        ctx[s] = prompt + out
    pol = None
    if ktc is not None:
        pol = type(sched.connector.connector_scheduler.manager._policy).__name__
    with open(args.out, "w") as f:
        json.dump({"arm": args.arm, "counters": counters, "policy": pol, "requests": recs}, f)
    print(f"arm={args.arm} policy={pol} counters={counters}", flush=True)
    return 0


def chunk_bytes():
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(MODEL)
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    return c.num_hidden_layers * 2 * c.num_key_value_heads * head_dim * 2 * BLOCK


def main():
    args = build_args()
    if args.arm is not None:
        return run_arm(args)
    tmp = tempfile.mkdtemp(prefix="holes_l2_")
    plan = plan_workload(args.seed, args.sessions, args.concurrency, args.max_turns)
    base = [sys.executable, os.path.abspath(__file__), "--seed", str(args.seed),
            "--sessions", str(args.sessions), "--concurrency", str(args.concurrency),
            "--max-turns", str(args.max_turns), "--max-tokens", str(args.max_tokens),
            "--max-model-len", str(args.max_model_len), "--gpu-util", str(args.gpu_util),
            "--dtype", args.dtype]
    if args.eager:
        base.append("--eager")

    def child(arm, extra):
        out = os.path.join(tmp, f"arm_{arm}.json")
        cmd = base + ["--arm", arm, "--out", out] + extra
        print("running", " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"arm {arm} failed rc={rc}\nRESULT FAIL")
            sys.exit(1)
        with open(out) as f:
            return json.load(f), out

    res = {}
    res["A"], ref_path = child("A", ["--blocks", str(args.ref_blocks)])
    recs = res["A"]["requests"]
    # Sizing from A's actual lengths: GPU pool about 1/3 of the peak live blocks
    # (at least one longest request), CPU pool about 1/3 of all offloadable chunks.
    last = {}
    for i, r in enumerate(recs):
        last[r["session"]] = i
    live, peak = {}, 0
    for i, r in enumerate(recs):
        live[r["session"]] = (len(r["prompt"]) + len(r["out"])) // BLOCK + 1
        peak = max(peak, sum(live.values()))
        if last[r["session"]] == i:
            live.pop(r["session"])
    longest = max((len(r["prompt"]) + len(r["out"])) // BLOCK + 2 for r in recs)
    gpu_blocks = max(peak // 3, longest + 40)
    total_blocks = sum(
        (len(recs[last[s]]["prompt"]) + len(recs[last[s]]["out"])) // BLOCK for s in last
    )
    cpu_bytes = (total_blocks // 3) * chunk_bytes()
    params = {
        "tool_gaps": [g * args.gap_scale for g in plan["tool_gaps"]],
        "decode_durs": [r["dur"] for r in recs],
        "p_end": plan["p_end"],
        "floor": 0.01,
    }
    params_path = os.path.join(tmp, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f)
    print(f"sizing: peak_live_blocks={peak} longest={longest} gpu_blocks={gpu_blocks} "
          f"total_chunks={total_blocks} cpu_bytes={cpu_bytes} (synthetic workload)", flush=True)
    # The small pool must hold one longest request (the engine checks it
    # against max_model_len), so B-D run with max_model_len = longest + 16.
    longest_tokens = max(len(r["prompt"]) + len(r["out"]) for r in recs) + BLOCK
    common = ["--max-model-len", str(longest_tokens), "--ref", ref_path, "--blocks", str(gpu_blocks), "--cpu-bytes", str(cpu_bytes),
              "--params", params_path]
    for arm in ("B", "C", "D"):
        res[arm], _ = child(arm, common)

    print("\nSUMMARY (synthetic workload) arm tokens_equal_to_A")
    ok = True
    for arm in ("B", "C", "D"):
        bad = [i for i, (a, b) in enumerate(zip(recs, res[arm]["requests"]))
               if a["out"] != b["out"]]
        print(f"  {arm} policy={res[arm]['policy']} equal {len(recs) - len(bad)}/{len(recs)} "
              f"counters={res[arm]['counters']}")
        for i in bad:
            a, b = recs[i], res[arm]["requests"][i]
            k = next((j for j, (x, y) in enumerate(zip(a["out"], b["out"])) if x != y), 0)
            print(f"    {arm} request {i} (session {a['session']} turn {a['turn']}) first differs at "
                  f"token {k}; A top1-top2 logprob gap {a['gaps'][k]:.3f}, {arm} gap "
                  f"{b['gaps'][k]:.3f}; cached A={a['cached']} {arm}={b['cached']}")
        if arm == "C":
            if bad:
                print(f"  C: {len(bad)} mismatching request(s): a near-tie divergence is "
                      "checked with the logprob gaps above and is not a pass")
                ok = False
            c = res["C"]["counters"]
            if c["l2_evicted"] == 0:
                print("  C: no L2 eviction: vacuous")
                ok = False
            if c["loaded_hole_admissions"] == 0:
                print("  C: no admission with an L2 range loaded and a recomputed hole")
                ok = False
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    t = time.time()
    rc = main()
    print(f"elapsed {time.time() - t:.1f}s")
    sys.exit(rc)
