# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/rate GPU end-to-end: a multi-turn session workload, LRU vs rate.

One engine per process (one arm per invocation), through ``gpurun``:

    python gpu_rate_e2e.py workload --dataset JSONL --out DIR  # CPU: build the workload
    python gpu_rate_e2e.py calib --out DIR               # fit the cost line (GPU)
    python gpu_rate_e2e.py run --arm big  --out DIR      # reference, no eviction
    python gpu_rate_e2e.py run --arm lru  --out DIR      # small cache, stock LRU
    python gpu_rate_e2e.py run --arm rate --out DIR      # small cache, rate
    python gpu_rate_e2e.py report --out DIR              # CPU: compare arms

The workload is a scaled-down slice of the TraceLab codex sessions: each
session's turn k sends the previous prompt + the previous *generated* output +
new tokens, after the session's tool gap. All arms run with holes on and
greedy decoding; ``big`` has a cache large enough to never evict, so its
outputs are the reference.
"""

import argparse
import heapq
import json
import os
import random
import statistics
import sys
import time

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


# ---------------------------------------------------------------- workload


def make_workload(args):
    rng = random.Random(args.seed)
    sessions = []
    gaps_all = []
    n_req = 0
    with open(args.dataset) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            subs = d["sub_requests"]
            for j, x in enumerate(subs):
                if j + 1 < len(subs):
                    gaps_all.append(x["tool_duration_ns"] / 1e9)
            n_req += len(subs)
            if len(sessions) >= args.sessions:
                continue
            turns = []
            prev_end = 0
            for j, x in enumerate(subs[: args.max_turns]):
                if j == 0:
                    new = max(64, x["input_toks"] // args.first_div)
                else:
                    new = max(16, (x["input_toks"] - prev_end) // args.tok_div)
                new = min(new, args.max_new)
                prev_end = x["input_toks"] + x["output_toks"]
                out = min(args.max_out, max(8, x["output_toks"] // args.tok_div))
                gap = x["tool_duration_ns"] / 1e9 * args.time_scale
                turns.append(
                    {
                        "new_ids": [rng.randrange(1000, 150000) for _ in range(new)],
                        "max_tokens": out,
                        "gap": round(gap, 4),
                    }
                )
            sessions.append(
                {
                    "sid": d["session_id"],
                    "arrival": round(d["arrival_time_ns"] / 1e9 * args.time_scale, 4),
                    "turns": turns,
                }
            )
    params = {
        "tool_gaps": gaps_all,
        # Unused with running_owner="protect" (no decode state on one GPU);
        # the codex output sizes at ~20 ms per token.
        "decode_durs": [0.02 * 400.0],
        "p_end": 100.0 / n_req,
        "floor": 1.0,
        "src": "codex_n100_s400_k600_seed0: tool gaps, p_end = sessions / requests",
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "workload.json"), "w") as f:
        json.dump({"sessions": sessions, "time_scale": args.time_scale, "args": vars(args)}, f)
    with open(os.path.join(args.out, "params_codex.json"), "w") as f:
        json.dump(params, f)
    nt = sum(len(s["turns"]) for s in sessions)
    print(f"workload: {len(sessions)} sessions, {nt} requests; params p_end={params['p_end']:.3f}")


# ---------------------------------------------------------------- engine


def make_llm(args, num_blocks, eviction):
    from vllm import LLM

    add = {"holes": {"enabled": True}}
    if eviction is not None:
        add["eviction"] = eviction
    kw = dict(
        model=MODEL,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=2048,
        seed=0,
        additional_config=add,
    )
    if num_blocks:
        kw["num_gpu_blocks_override"] = num_blocks
    return LLM(**kw)


def scheduler_of(llm):
    return llm.llm_engine.engine_core.engine_core.scheduler


def calib(args):
    """Fresh prefills of L tokens: fit S = A + b1 L + b2 L^2 / 2."""
    import numpy as np

    from vllm import SamplingParams

    llm = make_llm(args, 0, None)
    rng = random.Random(1)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    rows = []
    for L in [256, 512, 1024, 2048, 3072, 4096, 6144, 8000][:: 1]:
        if L + 1 > args.max_model_len:
            continue
        ts = []
        for rep in range(4):
            ids = [rng.randrange(1000, 150000) for _ in range(L)]
            t = time.perf_counter()
            llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)
            ts.append(time.perf_counter() - t)
        rows.append((L, min(ts[1:])))
        print(f"calib L={L} t={rows[-1][1] * 1e3:.1f} ms", flush=True)
    L = np.array([r[0] for r in rows], float)
    S = np.array([r[1] for r in rows], float)
    X = np.stack([np.ones_like(L), L, L * L / 2.0], 1)
    coef, *_ = np.linalg.lstsq(X, S, rcond=None)
    A, b1, b2 = (float(c) for c in coef)
    fit = {"A": A, "b1": b1, "b2": max(b2, 0.0), "rows": rows}
    with open(os.path.join(args.out, "cost_fit.json"), "w") as f:
        json.dump(fit, f, indent=1)
    print("cost fit", json.dumps({k: v for k, v in fit.items() if k != "rows"}))


def run(args):
    from vllm import SamplingParams

    with open(os.path.join(args.out, "workload.json")) as f:
        wl = json.load(f)
    sessions = wl["sessions"]
    if args.arm == "big":
        num_blocks, eviction = args.big_blocks, None
    elif args.arm == "lru":
        num_blocks, eviction = args.num_blocks, {"policy": "lru"}
    else:
        eviction = {
            "policy": "rate",
            "params": os.path.join(args.out, "params_codex.json"),
            "time_scale": wl["time_scale"],
        }
        fit_path = os.path.join(args.out, "cost_fit.json")
        if os.path.exists(fit_path) and not args.paper_cost:
            with open(fit_path) as f:
                fit = json.load(f)
            eviction["cost_b1"] = fit["b1"]
            eviction["cost_b2"] = fit["b2"]
        num_blocks = args.num_blocks
    llm = make_llm(args, num_blocks, eviction)
    import vllm

    print("vllm from", vllm.__file__, flush=True)
    sch = scheduler_of(llm)
    print("eviction policy:", type(sch.eviction_policy).__name__, flush=True)
    # Warm up (Triton JIT, allocator) so the first arm's TTFT is not inflated,
    # then clear the prefix cache.
    wrng = random.Random(7)
    for L in (300, 700, 1500, 3000, 6000):
        ids = [wrng.randrange(1000, 150000) for _ in range(L)]
        llm.generate(
            [{"prompt_token_ids": ids}, {"prompt_token_ids": ids[: L // 2] + ids[:16]}],
            SamplingParams(max_tokens=4, temperature=0.0),
            use_tqdm=False,
        )
    assert llm.reset_prefix_cache()
    counters = {"preemptions": 0, "scheduled_tokens": 0, "steps": 0, "hole_tokens": 0}
    orig_preempt = sch._preempt_request

    def preempt(request, timestamp):
        counters["preemptions"] += 1
        return orig_preempt(request, timestamp)

    sch._preempt_request = preempt
    orig_update = sch.update_from_output

    def update(so, mro):
        counters["scheduled_tokens"] += so.total_num_scheduled_tokens
        counters["steps"] += 1
        for rs in (so.scheduled_hole_ranges or {}).values():
            counters["hole_tokens"] += sum(e - s for s, e in rs)
        return orig_update(so, mro)

    sch.update_from_output = update

    engine = llm.llm_engine
    # (due time, seq, session index, turn index)
    pending = [(s["arrival"], i, i, 0) for i, s in enumerate(sessions)]
    heapq.heapify(pending)
    ctx = {i: [] for i in range(len(sessions))}
    live = {}  # rid -> record
    recs = []
    seq = len(sessions)
    t0 = time.monotonic()
    while pending or engine.has_unfinished_requests():
        now = time.monotonic() - t0
        while pending and pending[0][0] <= now:
            due, _, si, k = heapq.heappop(pending)
            s = sessions[si]
            turn = s["turns"][k]
            prompt = ctx[si] + turn["new_ids"]
            if len(prompt) + turn["max_tokens"] > args.max_model_len:
                continue  # session ends: context cap
            rid = f"s{si}t{k}"
            sp = SamplingParams(
                max_tokens=turn["max_tokens"],
                temperature=0.0,
                ignore_eos=True,
                extra_args={"session_id": s["sid"]},
            )
            engine.add_request(rid, {"prompt_token_ids": prompt}, sp)
            live[rid] = {
                "rid": rid, "si": si, "turn": k, "due": due, "prompt": prompt,
                "prompt_len": len(prompt), "added": now, "ttft": None,
            }
        if not engine.has_unfinished_requests():
            if pending:
                time.sleep(max(0.0, min(0.02, pending[0][0] - now)))
            continue
        outs = engine.step()
        tnow = time.monotonic() - t0
        for o in outs:
            r = live.get(o.request_id)
            if r is None:
                continue
            if r["ttft"] is None and o.outputs and len(o.outputs[0].token_ids) > 0:
                r["ttft"] = tnow - r["due"]
            if o.finished:
                out_ids = list(o.outputs[0].token_ids)
                r.update(
                    out=out_ids,
                    cached=o.num_cached_tokens or 0,
                    done=tnow,
                )
                del r["prompt"]
                recs.append(r)
                del live[o.request_id]
                si, k = r["si"], r["turn"]
                ctx[si] = ctx[si] + sessions[si]["turns"][k]["new_ids"] + out_ids
                if k + 1 < len(sessions[si]["turns"]):
                    seq += 1
                    gap = sessions[si]["turns"][k]["gap"]
                    heapq.heappush(pending, (tnow + gap, seq, si, k + 1))
    wall = time.monotonic() - t0
    pol = sch.eviction_policy
    res = {
        "arm": args.arm,
        "num_blocks": sch.kv_cache_manager.block_pool.num_gpu_blocks,
        "eviction": eviction,
        "wall_s": wall,
        "counters": counters,
        "policy_stats": pol.stats() if pol is not None else None,
        "requests": sorted(recs, key=lambda r: (r["si"], r["turn"])),
    }
    with open(os.path.join(args.out, f"arm_{args.arm}.json"), "w") as f:
        json.dump(res, f)
    summarize(res)


def summarize(res):
    reqs = res["requests"]
    prompt = sum(r["prompt_len"] for r in reqs)
    cached = sum(r["cached"] for r in reqs)
    ttft = [r["ttft"] for r in reqs]
    c = res["counters"]
    print(
        f"[{res['arm']}] blocks={res['num_blocks']} n={len(reqs)} prompt={prompt} "
        f"hit={cached} ({cached / max(prompt, 1):.3f}) recomputed={prompt - cached} "
        f"sched_tokens={c['scheduled_tokens']} hole_tokens={c['hole_tokens']} "
        f"preempt={c['preemptions']} ttft_mean={statistics.mean(ttft):.4f} "
        f"ttft_p50={statistics.median(ttft):.4f} "
        f"ttft_p90={statistics.quantiles(ttft, n=10)[8]:.4f} wall={res['wall_s']:.1f}s "
        f"policy={res['policy_stats']}",
        flush=True,
    )


def report(args):
    arms = {}
    for a in ("big", "lru", "rate"):
        p = os.path.join(args.out, f"arm_{a}.json")
        if os.path.exists(p):
            with open(p) as f:
                arms[a] = json.load(f)
    for res in arms.values():
        summarize(res)
    ref = {r["rid"]: r for r in arms["big"]["requests"]} if "big" in arms else None
    if ref is None:
        return
    for a, res in arms.items():
        if a == "big":
            continue
        same = diverged = missing = 0
        first_div = []
        for r in res["requests"]:
            q = ref.get(r["rid"])
            if q is None:
                missing += 1
                continue
            if q["out"] == r["out"] and q["prompt_len"] == r["prompt_len"]:
                same += 1
            else:
                diverged += 1
                n = next(
                    (i for i, (x, y) in enumerate(zip(q["out"], r["out"])) if x != y),
                    min(len(q["out"]), len(r["out"])),
                )
                first_div.append((r["rid"], n, q["prompt_len"] == r["prompt_len"]))
        print(
            f"[{a}] outputs identical to big: {same}/{same + diverged} "
            f"(missing {missing}); first divergences: {first_div[:10]}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["workload", "calib", "run", "report"])
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--dataset",
        default=os.environ.get("RATE_E2E_DATASET"),
        help="agentic session JSONL (sub_requests with input_toks / output_toks / "
        "tool_duration_ns), e.g. TraceLab codex_n100_s400_k600_seed0.jsonl; "
        "required by `workload`, also read from $RATE_E2E_DATASET",
    )
    ap.add_argument("--arm", choices=["big", "lru", "rate"], default="rate")
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--first-div", type=int, default=16)
    ap.add_argument("--tok-div", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-out", type=int, default=48)
    ap.add_argument("--time-scale", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=6)
    ap.add_argument("--num-blocks", type=int, default=1200)
    ap.add_argument("--big-blocks", type=int, default=0, help="0 = all that fit")
    ap.add_argument("--paper-cost", action="store_true")
    args = ap.parse_args()
    if args.cmd == "workload" and not args.dataset:
        ap.error("workload needs --dataset (or $RATE_E2E_DATASET)")
    {"workload": make_workload, "calib": calib, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.exit(main())
