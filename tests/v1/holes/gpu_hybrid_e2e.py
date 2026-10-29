# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/M GPU check: hybrid (Full + sliding-window) replay on a tiny model.

A tiny Llama-style config (4 layers, alternating full / sliding attention,
random dummy weights, bf16) is built in a temp directory; no download. Two
arms, each in its own child process, one GPU engine at a time:

* R reference: holes off, no connector.
* H holes on (hybrid), CPU offload connector with the ``rate`` policy.

Every prompt is run once (fills the GPU prefix cache), then cache pages are
evicted by hash (scenario 1: one Full page in the middle; scenario 2: the same
Full page plus the SWA page that breaks the nearer window state, so the replay
starts earlier and crosses a retained Full page whose writes are suppressed)
and the prompt is run again. Pass: per request the generated token ids equal
the reference; at least one replay span was scheduled (non-vacuity). A
near-tie divergence is reported with the logprob gap, never as a pass.

Usage: python gpu_hybrid_e2e.py [--prompts N] [--eager]
"""

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

BLOCK = 16
VOCAB = 2048


def write_mamba_config(d: str) -> None:
    """Tiny Granite-4.0-H style (Mamba2 + attention) hybrid, dense: layers 1
    and 3 are attention, 0 and 2 Mamba2."""
    cfg = {
        "architectures": ["GraniteMoeHybridForCausalLM"],
        "model_type": "granitemoehybrid",
        "vocab_size": VOCAB,
        "hidden_size": 128,
        "intermediate_size": 256,
        "shared_intermediate_size": 256,
        "num_local_experts": 0,
        "num_experts_per_tok": 0,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "layer_types": ["mamba", "attention", "mamba", "attention"],
        "position_embedding_type": "nope",
        "mamba_n_heads": 8,
        "mamba_d_head": 32,
        "mamba_n_groups": 1,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_chunk_size": 16,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)


def write_config(d: str) -> None:
    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": VOCAB,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "layer_types": [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        "sliding_window": 48,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)


MAMBA_UNIT = 48  # the hybrid page size of the tiny Granite-H config (checked at run time)


def make_prompts(n: int, seed: int, unit: int = BLOCK, pages: int = 8) -> list[list[int]]:
    rng = random.Random(seed)
    return [
        [rng.randrange(10, VOCAB) for _ in range(pages * unit + rng.randrange(1, 12))]
        for _ in range(n)
    ]


def run_arm(args) -> int:
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig
    from vllm.inputs import TokensPrompt

    model_dir = tempfile.mkdtemp(prefix="tiny_hybrid_")
    (write_mamba_config if args.kind == "mamba" else write_config)(model_dir)
    hybrid = args.arm == "H"
    add = {"holes": {"enabled": True, "test_hooks": True}} if hybrid else {}
    ktc = None
    if hybrid:
        pj = os.path.join(model_dir, "rate_params.json")
        with open(pj, "w") as f:  # synthetic rate parameters (no real trace)
            json.dump({"tool_gaps": [2.0, 3.0, 1.5], "decode_durs": [0.05],
                       "p_end": 0.125, "floor": 0.01}, f)
        ktc = KVTransferConfig(
            kv_connector="OffloadingConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "cpu_bytes_to_use": 1 << 26,
                "eviction_policy": "rate",
                "rate": {"params": pj},
            },
        )
    llm = LLM(
        model_dir,
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=1024,
        max_num_batched_tokens=MAMBA_UNIT if args.kind == "mamba" else 32,  # chunked prefill: SWA windows materialize at chunk ends
        max_num_seqs=1,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=True,
        enforce_eager=args.eager,
        seed=0,
        block_size=BLOCK,
        additional_config=add,
        **({"mamba_cache_mode": "align"} if args.kind == "mamba" else {}),
        kv_transfer_config=ktc,
    )
    sched = llm.llm_engine.engine_core.engine_core.scheduler
    if args.weight_scale != 1.0:
        # Dummy weights are uniform +-1e-3: logits are then so flat that a wrong
        # replay position or missing replay KV does not change greedy tokens
        # (code-r1 finding 10). Rescale in place (same addresses, so graphs stay
        # valid) in both arms; test-side only.
        core = llm.llm_engine.engine_core.engine_core
        model = core.model_executor.driver_worker.worker.model_runner.model
        import torch

        with torch.no_grad():
            for prm in model.parameters():
                if prm.is_floating_point():
                    prm.mul_(args.weight_scale)
        torch.cuda.synchronize()
    counters = {"replay_spans": 0, "replay_tokens": 0, "no_write_tokens": 0, "tails": 0}
    step_log = []
    if hybrid:
        assert sched.hybrid_holes_enabled, "hybrid holes not effective"
        orig = sched.update_from_output

        def upd(so, mro):
            for st in (so.hybrid_replay_steps or {}).values():
                step_log.append([st.start, st.end, st.restore, st.no_write])
                if st.end > st.start:
                    counters["replay_spans"] += 1
                    counters["replay_tokens"] += st.end - st.start
                    counters["no_write_tokens"] += sum(
                        e - s for v in st.no_write.values() for s, e in v
                    )
                else:
                    counters["tails"] += 1
            return orig(so, mro)

        sched.update_from_output = upd

        from vllm.utils.hashing import get_hash_fn_by_name
        from vllm.v1.core.kv_cache_utils import (
            get_request_block_hasher,
            make_block_hash_with_group_id,
            resolve_block_hashes,
        )
        from vllm.v1.request import Request

        cc = sched.vllm_config.cache_config
        hbs = sched.kv_cache_manager.coordinator.hash_block_size
        hasher = get_request_block_hasher(
            hbs, get_hash_fn_by_name(cc.prefix_caching_hash_algo)
        )
        kinds = sched.kv_cache_manager.coordinator.hybrid_kinds
        full_gid = kinds.index("full")
        swa_gid = kinds.index("mamba" if args.kind == "mamba" else "swa")
        cache = sched.kv_cache_manager.block_pool.cached_block_hash_to_block._cache

        def evict(prompt, pages_gids):
            r = Request(
                request_id="probe",
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_hasher=hasher,
            )
            for page, gid in pages_gids:
                bs = sched.kv_cache_manager.coordinator.single_type_managers[gid].block_size
                h = resolve_block_hashes(r.block_hashes, hbs, bs)[page]
                cache.pop(make_block_hash_with_group_id(h, gid), None)
            res = sched.kv_cache_manager.coordinator.hybrid_residency(
                r.block_hashes, r.num_tokens
            )
            print("RESIDENCY", pages_gids, res, flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=12, ignore_eos=True, logprobs=2)
    recs = []
    mamba = args.kind == "mamba"
    if hybrid and mamba:
        sizes = [m.block_size for m in sched.kv_cache_manager.coordinator.single_type_managers]
        assert set(sizes) == {MAMBA_UNIT}, f"group block sizes {sizes}"
    prompts = make_prompts(args.prompts, 7, MAMBA_UNIT, 4) if mamba else make_prompts(args.prompts, 7)
    for i, prompt in enumerate(prompts):
        runs = [("fresh", [])]
        if hybrid and mamba:
            # chunked prefill (budget = one page) leaves a Mamba state at every
            # page boundary: restore from 48, and from 48 across a lost state
            # (Mamba align keeps only the last-prompt-boundary state: a short
            # request ending after page 2 leaves the state at 96 to restore from)
            llm.generate(
                [TokensPrompt(prompt_token_ids=prompt[: 2 * MAMBA_UNIT + 1])],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
            runs += [
                ("restore", [(2, full_gid)]),
                ("earlier_state", [(3, full_gid), (1, swa_gid)]),
            ]
        elif hybrid:
            # a short request that ends exactly on page 2 leaves a retained SWA
            # window at 32, the state a replay can restore from
            llm.generate(
                [TokensPrompt(prompt_token_ids=prompt[: 2 * BLOCK])],
                SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True),
                use_tqdm=False,
            )
            runs += [("restore", [(2, full_gid)])]
            runs += [
                ("full_hole", [(5, full_gid)]),
                ("earlier_state", [(5, full_gid), (3, swa_gid)]),
            ]
        for tag, ev in runs:
            if hybrid and ev:
                if tag != "restore" or mamba:
                    llm.generate(
                        [TokensPrompt(prompt_token_ids=prompt)], sp, use_tqdm=False
                    )  # (re)fill the GPU cache of every page
                evict(prompt, ev)
            o = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt)], sp, use_tqdm=False
            )[0]
            out = list(o.outputs[0].token_ids)
            gaps = []
            for lp in o.outputs[0].logprobs or []:
                v = sorted((x.logprob for x in lp.values()), reverse=True)
                gaps.append(v[0] - v[1] if len(v) > 1 else float("inf"))
            recs.append(
                {"prompt_idx": i, "tag": tag, "out": out, "gaps": gaps,
                 "cached": o.num_cached_tokens or 0}
            )
    with open(args.out, "w") as f:
        json.dump({"arm": args.arm, "counters": counters, "steps": step_log,
                   "requests": recs}, f)
    print(f"arm={args.arm} counters={counters}", flush=True)
    print("CACHED", [(r["prompt_idx"], r["tag"], r["cached"]) for r in recs], flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["R", "H"])
    ap.add_argument("--out")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--kind", choices=["swa", "mamba"], default="swa")
    ap.add_argument("--weight-scale", type=float, default=300.0,
                    help="multiply the dummy weights (+-1e-3) by this; 1 = unscaled")
    args = ap.parse_args()
    if args.arm:
        return run_arm(args)
    tmp = tempfile.mkdtemp(prefix="hybrid_e2e_")
    outs = {}
    for arm in ("R", "H"):
        outs[arm] = os.path.join(tmp, f"{arm}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--arm", arm,
               "--out", outs[arm], "--prompts", str(args.prompts),
               "--weight-scale", str(args.weight_scale), "--kind", args.kind]
        if args.eager:
            cmd.append("--eager")
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"RESULT FAIL: arm {arm} exited {rc}")
            return 1
    ref = {r["prompt_idx"]: r["out"] for r in json.load(open(outs["R"]))["requests"]}
    h = json.load(open(outs["H"]))
    bad = [
        (r["prompt_idx"], r["tag"], min(r["gaps"]) if r["gaps"] else None)
        for r in h["requests"]
        if r["out"] != ref[r["prompt_idx"]]
    ]
    c = h["counters"]
    nonvacuous = c["replay_spans"] > 0 and (c["no_write_tokens"] > 0 or args.kind == "mamba")
    print(f"counters={c} mismatches={bad}")
    print("steps (start, end, restore, no_write):", h.get("steps"))
    ok = not bad and nonvacuous
    print("RESULT", "PASS" if ok else "FAIL",
          "(non-vacuous)" if nonvacuous else "(VACUOUS: no replay span)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
