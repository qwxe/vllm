# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/rate-eval: repeated, deterministic LRU-vs-rate comparison on one GPU.

Derived from ``gpu_rate_e2e.py``. Differences:

* ``--clock virtual``: the engine is driven by a virtual clock. Arrivals and
  tool gaps are in virtual seconds, each engine step advances the clock by a
  fitted step-time model (``--step-model``), and the rate policy's session
  tracker reads the same clock. Scheduling is then a deterministic function
  of the workload, so the work metrics (hit / recomputed tokens) and the
  modelled TTFT carry no laptop timing noise. ``--clock real`` replays in
  wall time as before (measured TTFT, noisy).
* Several workloads: ``--offset`` takes a different slice of sessions of the
  dataset (arrivals rebased to 0), so repetitions vary the workload.
* Arms: ``big`` (no eviction), ``lru`` (stock), ``rate``, ``rate_nopos``
  (rate without position cost, front first) and ``rate_tail`` (rate's owner
  order, no position cost, tail first: hole-free).
* Params: ``dataset`` (all tool gaps of the dataset, p_end = sessions /
  requests of the whole dataset, as gpu_rate_e2e.py) or ``insample`` (the
  gaps and p_end of the replayed workload itself, after truncation).
* Every step is logged (wall time, tokens, attention work), so the step-time
  model is fitted from real runs (``fit`` command).

Commands (GPU ones through gpurun):

    python gpu_rate_eval.py workload --dataset JSONL --out D --offset 0 --tag w0
    python gpu_rate_eval.py run --out D --tag w0 --arm lru --clock real --rep 0
    python gpu_rate_eval.py fit --out D            # CPU: step model from real runs
    python gpu_rate_eval.py run --out D --tag w0 --arm rate --clock virtual
    python gpu_rate_eval.py report --out D         # CPU: tables
"""

import argparse
import glob
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
    rng = random.Random(1000 + args.offset)
    rows = []
    gaps_all = []
    n_req_all = 0
    with open(args.dataset) as f:
        for line in f:
            d = json.loads(line)
            subs = d["sub_requests"]
            for j, x in enumerate(subs):
                if j + 1 < len(subs):
                    gaps_all.append(x["tool_duration_ns"] / 1e9)
            n_req_all += len(subs)
            rows.append(d)
    n_sess_all = len(rows)
    rows.sort(key=lambda d: d["arrival_time_ns"])
    rows = rows[args.offset: args.offset + args.sessions]
    t0 = rows[0]["arrival_time_ns"] / 1e9
    sessions = []
    used_gaps = []
    n_used = 0
    for d in rows:
        subs = d["sub_requests"]
        turns = []
        prev_end = 0
        ctx = 0
        for j, x in enumerate(subs[: args.max_turns]):
            if j == 0:
                new = max(64, x["input_toks"] // args.first_div)
            else:
                new = max(16, (x["input_toks"] - prev_end) // args.tok_div)
            new = min(new, args.max_new)
            prev_end = x["input_toks"] + x["output_toks"]
            out = min(args.max_out, max(8, x["output_toks"] // args.tok_div))
            gap = x["tool_duration_ns"] / 1e9 * args.time_scale
            if ctx + new + out > args.max_model_len:
                break  # the context cap ends the session (same rule as run)
            ctx += new + out
            turns.append(
                {
                    "new_ids": [rng.randrange(1000, 150000) for _ in range(new)],
                    "max_tokens": out,
                    "gap": round(gap, 4),
                }
            )
        for k in range(len(turns) - 1):
            used_gaps.append(turns[k]["gap"] / args.time_scale)
        n_used += len(turns)
        sessions.append(
            {
                "sid": d["session_id"],
                "arrival": round((d["arrival_time_ns"] / 1e9 - t0) * args.time_scale, 4),
                "turns": turns,
            }
        )
    os.makedirs(args.out, exist_ok=True)
    common = {"decode_durs": [0.02 * 400.0], "floor": 1.0}
    pd = dict(common, tool_gaps=gaps_all, p_end=n_sess_all / n_req_all,
              src=f"dataset: {n_sess_all} sessions / {n_req_all} requests")
    pi = dict(common, tool_gaps=used_gaps or gaps_all, p_end=len(sessions) / max(1, n_used),
              src=f"insample: {len(sessions)} sessions / {n_used} requests")
    with open(os.path.join(args.out, f"workload_{args.tag}.json"), "w") as f:
        json.dump({"sessions": sessions, "time_scale": args.time_scale, "args": vars(args)}, f)
    for name, p in (("dataset", pd), ("insample", pi)):
        with open(os.path.join(args.out, f"params_{name}_{args.tag}.json"), "w") as f:
            json.dump(p, f)
    ptok = 0
    for s in sessions:
        c = 0
        for t in s["turns"]:
            c += len(t["new_ids"])
            ptok += c
            c += t["max_tokens"]
    print(f"workload {args.tag}: {len(sessions)} sessions, {n_used} requests, ~{ptok} prompt tokens; "
          f"p_end dataset {pd['p_end']:.3f} insample {pi['p_end']:.3f}; gaps dataset "
          f"mean {statistics.mean(gaps_all):.2f} s, insample mean "
          f"{statistics.mean(used_gaps) if used_gaps else float('nan'):.2f} s (unscaled)")


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


def arm_eviction(args, wl):
    if args.arm == "big":
        return args.big_blocks, None
    if args.arm == "lru":
        return args.num_blocks, {"policy": "lru"}
    ev = {
        "policy": "rate",
        "params": os.path.join(args.out, f"params_{args.params}_{args.tag}.json"),
        "time_scale": wl["time_scale"] * args.tau_scale,
    }
    fit_path = os.path.join(args.out, "cost_fit.json")
    if os.path.exists(fit_path):
        with open(fit_path) as f:
            fit = json.load(f)
        ev["cost_b1"] = fit["b1"]
        ev["cost_b2"] = fit["b2"]
    if args.arm == "rate_nopos":
        ev["position_cost"] = False
    elif args.arm == "rate_tail":
        ev["position_cost"] = False
        ev["order"] = "tail"
    elif args.arm == "lru_like":
        # rate's machinery with its owner order only (no position cost, tail
        # first) and no queued protection: free uncached blocks first, then
        # idle owners by rate(t), which is monotone in the time since finish
        # for these params, i.e. LRU by finish time.
        ev["position_cost"] = False
        ev["order"] = "tail"
        ev["protect_queued"] = False
    return args.num_blocks, ev


def run(args):
    from vllm import SamplingParams

    with open(os.path.join(args.out, f"workload_{args.tag}.json")) as f:
        wl = json.load(f)
    sessions = wl["sessions"]
    num_blocks, eviction = arm_eviction(args, wl)
    llm = make_llm(args, num_blocks, eviction)
    import vllm

    print("vllm from", vllm.__file__, flush=True)
    sch = llm.llm_engine.engine_core.engine_core.scheduler
    pol = sch.eviction_policy
    print("eviction policy:", type(pol).__name__, eviction, flush=True)
    # Warm up (JIT, allocator), then clear the prefix cache.
    wrng = random.Random(7)
    for L in (300, 700, 1500, 3000, 6000):
        if L + 16 > args.max_model_len:
            continue
        ids = [wrng.randrange(1000, 150000) for _ in range(L)]
        llm.generate(
            [{"prompt_token_ids": ids}, {"prompt_token_ids": ids[: L // 2] + ids[:16]}],
            SamplingParams(max_tokens=4, temperature=0.0),
            use_tqdm=False,
        )
    assert llm.reset_prefix_cache()

    virtual = args.clock == "virtual"
    vclock = [0.0]
    model = None
    if virtual:
        with open(args.step_model) as f:
            model = json.load(f)["coef"]
    if pol is not None:
        if virtual:
            pol.clock = lambda: vclock[0]
        else:
            t_base = time.monotonic()
            pol.clock = lambda: time.monotonic() - t_base

    counters = {"preemptions": 0, "scheduled_tokens": 0, "steps": 0, "hole_tokens": 0}
    steps = []
    cur = {}
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
        ntok = so.total_num_scheduled_tokens
        attn = 0.0
        npre = 0
        nseq = 0
        for rid, n in so.num_scheduled_tokens.items():
            req = sch.requests.get(rid)
            c = max(0, (req.num_computed_tokens if req is not None else 0) - n)
            attn += n * (c + n / 2.0)
            nseq += 1
            if n > 1:
                npre += 1
        cur.update(ntok=ntok, attn=attn, nseq=nseq, npre=npre)
        return orig_update(so, mro)

    sch.update_from_output = update

    def step_dt(f):
        c = model
        return c[0] + c[1] * f["ntok"] + c[2] * f["attn"] + c[3] * f["nseq"] + c[4] * f["npre"]

    engine = llm.llm_engine
    pending = [(s["arrival"], i, i, 0) for i, s in enumerate(sessions)]
    heapq.heapify(pending)
    ctx = {i: [] for i in range(len(sessions))}
    live = {}
    recs = []
    seq = len(sessions)
    t0 = time.monotonic()

    def now_s():
        return vclock[0] if virtual else time.monotonic() - t0

    while pending or engine.has_unfinished_requests():
        now = now_s()
        while pending and pending[0][0] <= now:
            due, _, si, k = heapq.heappop(pending)
            s = sessions[si]
            turn = s["turns"][k]
            prompt = ctx[si] + turn["new_ids"]
            if len(prompt) + turn["max_tokens"] > args.max_model_len:
                continue
            rid = f"s{si}t{k}"
            sp = SamplingParams(
                max_tokens=turn["max_tokens"],
                temperature=0.0,
                ignore_eos=True,
                extra_args={"session_id": s["sid"]},
            )
            engine.add_request(rid, {"prompt_token_ids": prompt}, sp)
            live[rid] = {
                "rid": rid, "si": si, "turn": k, "due": due,
                "prompt_len": len(prompt), "ttft": None,
            }
        if not engine.has_unfinished_requests():
            if pending:
                if virtual:
                    vclock[0] = max(vclock[0], pending[0][0])
                else:
                    time.sleep(max(0.0, min(0.02, pending[0][0] - now)))
            continue
        cur.clear()
        tw = time.perf_counter()
        outs = engine.step()
        wall = time.perf_counter() - tw
        if cur:
            f = dict(cur, wall=wall)
            steps.append(f)
            if virtual:
                vclock[0] += step_dt(f)
        tnow = now_s()
        for o in outs:
            r = live.get(o.request_id)
            if r is None:
                continue
            if r["ttft"] is None and o.outputs and len(o.outputs[0].token_ids) > 0:
                r["ttft"] = tnow - r["due"]
            if o.finished:
                out_ids = list(o.outputs[0].token_ids)
                r.update(cached=o.num_cached_tokens or 0, done=tnow, nout=len(out_ids))
                recs.append(r)
                del live[o.request_id]
                si, k = r["si"], r["turn"]
                ctx[si] = ctx[si] + sessions[si]["turns"][k]["new_ids"] + out_ids
                if k + 1 < len(sessions[si]["turns"]):
                    seq += 1
                    gap = sessions[si]["turns"][k]["gap"]
                    heapq.heappush(pending, (tnow + gap, seq, si, k + 1))
    res = {
        "arm": args.arm, "tag": args.tag, "clock": args.clock, "rep": args.rep,
        "params": args.params, "tau_scale": args.tau_scale,
        "num_blocks": sch.kv_cache_manager.block_pool.num_gpu_blocks,
        "eviction": eviction,
        "wall_s": time.monotonic() - t0,
        "end_s": now_s(),
        "counters": counters,
        "policy_stats": pol.stats() if pol is not None else None,
        "steps": steps,
        "requests": sorted(recs, key=lambda r: (r["si"], r["turn"])),
    }
    name = f"res_{args.tag}_{args.arm}{args.suffix}_{args.clock}_b{res['num_blocks']}_r{args.rep}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(res, f)
    print(summary_line(res), flush=True)


def summary_line(res):
    reqs = res["requests"]
    prompt = sum(r["prompt_len"] for r in reqs)
    cached = sum(r["cached"] for r in reqs)
    ttft = [r["ttft"] for r in reqs]
    c = res["counters"]
    return (
        f"[{res['tag']} {res['arm']} {res['clock']} r{res['rep']}] blocks={res['num_blocks']} n={len(reqs)} "
        f"prompt={prompt} hit={cached} ({cached / max(prompt, 1):.3f}) recomputed={prompt - cached} "
        f"sched={c['scheduled_tokens']} holes={c['hole_tokens']} preempt={c['preemptions']} "
        f"ttft mean={statistics.mean(ttft):.4f} p50={statistics.median(ttft):.4f} "
        f"p90={statistics.quantiles(ttft, n=10)[8]:.4f} end={res['end_s']:.1f} wall={res['wall_s']:.1f} "
        f"policy={res['policy_stats']}"
    )


def fit(args):
    """Least-squares step-time model from the real-clock runs' step logs."""
    import numpy as np

    X, y = [], []
    for p in sorted(glob.glob(os.path.join(args.out, "res_*_real_*.json"))):
        with open(p) as f:
            r = json.load(f)
        for s in r["steps"]:
            X.append([1.0, s["ntok"], s["attn"], s["nseq"], s["npre"]])
            y.append(s["wall"])
    X = np.array(X)
    y = np.array(y)
    # Drop the slowest 1 % of steps (laptop hiccups) before the fit.
    keep = y <= np.quantile(y, 0.99)
    coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    pred = X @ coef
    rel = np.abs(pred - y) / y
    out = {"coef": [float(c) for c in coef], "n_steps": int(len(y)),
           "rel_err_p50": float(np.median(rel)), "rel_err_p90": float(np.quantile(rel, 0.9)),
           "sum_pred_over_sum_wall": float(pred.sum() / y.sum()),
           "features": ["1", "ntok", "attn", "nseq", "npre"]}
    with open(os.path.join(args.out, "step_model.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["workload", "run", "fit"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--tag", default="w0")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--arm", choices=["big", "lru", "rate", "rate_nopos", "rate_tail", "lru_like"], default="rate")
    ap.add_argument("--clock", choices=["real", "virtual"], default="virtual")
    ap.add_argument("--step-model")
    ap.add_argument("--params", choices=["dataset", "insample"], default="dataset")
    ap.add_argument("--tau-scale", type=float, default=1.0)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--first-div", type=int, default=16)
    ap.add_argument("--tok-div", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--max-out", type=int, default=48)
    ap.add_argument("--time-scale", type=float, default=0.5)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gpu-util", type=float, default=0.7)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=6)
    ap.add_argument("--num-blocks", type=int, default=900)
    ap.add_argument("--big-blocks", type=int, default=0)
    args = ap.parse_args()
    {"workload": make_workload, "run": run, "fit": fit}[args.cmd](args)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    sys.exit(main())
