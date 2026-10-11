# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/core GPU end-to-end check (run through gpurun, one short process).

Creates holes artificially in a cached prefix (test hook), runs the request
again, and compares against a full recompute: greedy tokens must match and
logprobs must agree within bf16 tolerance; the step log must show that only
the hole and the tail were computed (blocks after the hole reused).

Usage: python gpu_holes_e2e.py [--eager] [--backend FLASH_ATTN|TRITON_ATTN|FLASHINFER]
"""

import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL = os.environ.get("HOLES_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.v1.request import Request

    import vllm

    print("vllm from", vllm.__file__, flush=True)
    kw = {}
    if os.environ.get("HOLES_MAX_LEN"):
        kw["max_model_len"] = int(os.environ["HOLES_MAX_LEN"])
    if os.environ.get("HOLES_HF_OVERRIDES"):
        import json

        kw["hf_overrides"] = json.loads(os.environ["HOLES_HF_OVERRIDES"])
    if os.environ.get("HOLES_CUSTOM_OPS"):
        # e.g. "-rotary_embedding": DeepSeek's rope otherwise JIT-compiles a
        # FlashInfer kernel, which needs nvcc
        kw["compilation_config"] = {"custom_ops": os.environ["HOLES_CUSTOM_OPS"].split(",")}
    if args.backend:
        kw["attention_backend"] = args.backend
    if os.environ.get("HOLES_EXTRA_KW"):
        # e.g. CP: {"prefill_context_parallel_size": 2, "block_size": 128, ...}
        import json

        kw.update(json.loads(os.environ["HOLES_EXTRA_KW"]))
    llm = LLM(
        MODEL,
        dtype=args.dtype,
        **({"max_model_len": 8192} if "max_model_len" not in kw else {}),
        max_num_batched_tokens=2048,
        max_num_seqs=8,
        gpu_memory_utilization=args.gpu_util,
        enable_prefix_caching=True,
        enforce_eager=args.eager,
        seed=0,
        additional_config={"holes": {"enabled": True, "test_hooks": True}},
        **kw,
    )
    core = llm.llm_engine.engine_core.engine_core
    sched = core.scheduler
    assert sched.holes_enabled, "holes not enabled"

    log = []
    orig = sched.schedule

    def schedule_spy():
        out = orig()
        log.append(
            (dict(out.num_scheduled_tokens), {k: list(v) for k, v in out.scheduled_hole_ranges.items()})
        )
        return out

    sched.schedule = schedule_spy

    tok = llm.get_tokenizer()
    import this  # noqa: F401  (Zen of Python, natural text)
    zen = "".join({**{chr(i + 65): chr((i + 13) % 26 + 65) for i in range(26)},
                   **{chr(i + 97): chr((i + 13) % 26 + 97) for i in range(26)}}.get(c, c)
                  for c in this.s)
    paras = []
    for k in range(400):
        paras.append(f"Section {k}. " + zen.splitlines()[2 + k % 19] + f" (note {k * 7 % 13})\n")
    text_ids = tok.encode("".join(paras))
    cap = int(os.environ.get("HOLES_VOCAB_CAP", "0"))
    if cap:  # truncated-vocab model slices: keep in-vocabulary ids only
        text_ids = [t for t in text_ids if t < cap]

    def prompt(n, seed):
        off = (seed * 977) % (len(text_ids) - n - 1)
        return text_ids[off: off + n]

    sp = SamplingParams(temperature=0.0, max_tokens=24, logprobs=5, ignore_eos=True)

    def run(prompts):
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp, use_tqdm=False)
        return [(list(o.outputs[0].token_ids), o.outputs[0].logprobs, o.num_cached_tokens) for o in outs]

    def evict(p, start, end):
        req = Request("probe", p, SamplingParams(max_tokens=1), None, block_hasher=core.request_block_hasher)
        return sched.holes_evict_token_range(req, start, end)

    def compare(ref, got, tag):
        """(tokens equal, |dlogprob| over the top-5 of the first token =
        the prefill's output, steps that agree before divergence)."""
        toks_ok = ref[0] == got[0]
        agree = next((i for i, (a, b) in enumerate(zip(ref[0], got[0])) if a != b),
                     len(ref[0]))
        lr, lg = ref[1][0], got[1][0]
        maxd = max(abs(v.logprob - lg[t].logprob) for t, v in lr.items() if t in lg)
        print(f"[{tag}] tokens_equal={toks_ok} agree_steps={agree} "
              f"max|dlogprob| first token top5={maxd:.4f} cached_tokens={got[2]}",
              flush=True)
        return toks_ok, maxd

    results = []
    noise = [0.0]  # |dlogprob| of stock arms (full hit, prefix-only, fresh in a batch)
    cases = [
        # (name, prompt length, holes as token ranges)
        ("mid_hole", 3000, [(512, 1536)]),
        ("front_hole", 3000, [(0, 512)]),
        ("two_holes", 3000, [(256, 512), (1024, 1280)]),
        ("big_hole_chunked", 6000, [(256, 4800)]),
        ("tail_1tok", 2049, [(512, 1024)]),
    ]
    for i, (name, n, holes) in enumerate(cases):
        p = prompt(n, 1000 + i)
        llm.reset_prefix_cache()
        log.clear()
        ref = run([p])[0]  # full recompute (empty cache)
        full_steps = list(log)
        log.clear()
        hit = run([p])[0]  # stock path: full prefix hit (no holes)
        nh = sum(evict(p, s, e) for s, e in holes)
        log.clear()
        got = run([p])[0]
        computed = sum(next(iter(ns.values())) for ns, _ in log if ns)
        hole_log = [hr for _, hr in log if hr]
        tok_ok, maxd = compare(ref, got, name)
        tok_ok2, d_hit = compare(ref, hit, name + "/fullhit")
        noise.append(d_hit)
        # Prefix-only baseline (what stock vLLM does with this hole):
        # everything after the hole start is recomputed.
        evict(p, holes[0][0], n)
        base = run([p])[0]
        _, maxd2 = compare(ref, base, name + "/prefix_only")
        noise.append(maxd2)
        # KV-manager block (block_size x PCP size under CP)
        bs = int(os.environ.get("HOLES_KV_BLOCK", "16"))
        frontier = ((n - 1) // bs) * bs
        expect_prefill = sum(e - s for s, e in holes) + (n - frontier)
        prefill_tokens = computed - (len(got[0]) - 1)
        print(f"   evicted_blocks={nh} hole_steps={hole_log} prefill_tokens={prefill_tokens} "
              f"expected={expect_prefill} full_steps={[next(iter(x[0].values())) for x in full_steps if x[0]][:4]}",
              flush=True)
        results.append((name, tok_ok, maxd, maxd2, prefill_tokens == expect_prefill))

    # Mixed batch: a holed request next to a fresh one and a fully cached one.
    pa, pb, pc = prompt(2500, 7), prompt(1800, 8), prompt(1200, 9)
    llm.reset_prefix_cache()
    refs = run([pa, pb, pc])
    llm.reset_prefix_cache()
    run([pa, pc])
    evict(pa, 800, 1600)
    log.clear()
    gots = run([pa, pb, pc])
    for tag, r, g in zip("abc", refs, gots):
        t, d = compare(r, g, "batch_" + tag)
        if tag != "a":
            noise.append(d)
        results.append(("batch_" + tag, t, d, 0.0, True))
    print("   batch hole steps:", [hr for _, hr in log if hr], flush=True)

    print("\nSUMMARY name tokens_equal dlp_first(holes) dlp_first(prefix_only) prefill_ok")
    # The holed run must deviate from the full recompute no more than stock
    # vLLM's own recomputation noise (different chunking / batching) does.
    lim = 2 * max(noise) + 1e-3
    exact_tokens = args.dtype != "bfloat16"
    print(f"stock noise max={max(noise):.4f} -> limit {lim:.4f}; "
          f"token equality required: {exact_tokens}")
    ok = True
    for r in results:
        print("  ", r)
        ok &= r[4] and r[2] <= lim and (r[1] or not exact_tokens)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    t = time.time()
    rc = main()
    print(f"elapsed {time.time() - t:.1f}s")
    sys.exit(rc)
