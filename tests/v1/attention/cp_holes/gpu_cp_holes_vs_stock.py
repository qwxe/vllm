# SPDX-License-Identifier: Apache-2.0
"""Greedy outputs + first-token logprobs of CP_HOLES vs a stock backend.

    gpurun env PYTHONPATH=<wt> <venv>/bin/python gpu_cp_holes_vs_stock.py \
        --backend TRITON_ATTN --out /tmp/stock.json
    gpurun env PYTHONPATH=<wt> <venv>/bin/python gpu_cp_holes_vs_stock.py \
        --cp-scheme ulysses --cp-dense sharded --out /tmp/cp.json \
        --compare /tmp/stock.json
"""

import argparse
import json
import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--cp-scheme", default=None)
    ap.add_argument("--cp-dense", default="replicated")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--tol", type=float, default=0.02,
                    help="max |dlogprob| of the first token (fp32: 0.02; "
                    "bf16: compare with the stock FA-vs-Triton noise)")
    ap.add_argument("--allow-token-diff", action="store_true",
                    help="bf16: greedy near-ties may flip on any path")
    a = ap.parse_args()
    backend = a.backend
    if a.cp_scheme:
        os.environ["VLLM_CP_HOLES_SCHEME"] = a.cp_scheme
        os.environ["VLLM_CP_HOLES_DENSE"] = a.cp_dense
        from vllm.v1.attention.backends.cp_holes import register
        register()
        backend = "CUSTOM"
    from vllm import LLM, SamplingParams
    llm = LLM("Qwen/Qwen2.5-0.5B-Instruct", dtype=a.dtype, max_model_len=4096,
              gpu_memory_utilization=a.gpu_util, enforce_eager=True, seed=0,
              max_num_batched_tokens=1024, attention_backend=backend)
    long = " ".join(f"Item {i}: the value is {i * 37 % 101}." for i in range(300))
    prompts = ["The capital of France is",
               "Write a haiku about the sea.",
               long + " Summarize the list in one sentence."]
    outs = llm.generate(prompts, SamplingParams(max_tokens=24, temperature=0,
                                                logprobs=5))
    res = []
    for o in outs:
        c = o.outputs[0]
        lp0 = {int(k): v.logprob for k, v in c.logprobs[0].items()}
        res.append({"text": c.text, "ids": list(c.token_ids), "lp0": lp0})
    json.dump(res, open(a.out, "w"))
    for r in res:
        print("OUT:", repr(r["text"][:60]))
    if a.compare:
        ref = json.load(open(a.compare))
        ok = True
        for i, (x, y) in enumerate(zip(res, ref)):
            same = x["ids"] == y["ids"]
            common = set(x["lp0"]) & {int(k) for k in y["lp0"]}
            dlp = max(abs(x["lp0"][k] - y["lp0"][str(k)]) for k in common)
            print(f"prompt {i}: tokens_equal={same} max|dlogprob0|={dlp:.4f}")
            ok &= (same or a.allow_token_diff) and dlp < a.tol
        print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
