# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA_MLA vs stock TRITON_MLA on hole-free inputs (one backend per process).

    python gpu_msa_mla_vs_stock.py run BACKEND out.json   # MODEL env = model dir
    python gpu_msa_mla_vs_stock.py cmp a.json b.json

Prompts of 300-6000 tokens, chunked prefill at 1024 tokens (so later chunks run
with cached context), a second pass with full prefix-cache hits, and a batch.
Compares greedy tokens and first-token top-5 logprobs.
"""

import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def run(backend, out):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    cap = int(os.environ.get("VOCAB_CAP", "16384"))
    llm = LLM(os.environ["MODEL"], dtype=os.environ.get("DTYPE", "bfloat16"),
              max_model_len=8192, max_num_batched_tokens=1024, max_num_seqs=8,
              gpu_memory_utilization=0.6, enforce_eager=True, seed=0,
              enable_prefix_caching=True, attention_backend=backend,
              compilation_config={"custom_ops": ["-rotary_embedding"]})
    tok = llm.get_tokenizer()
    import this  # noqa: F401
    text = "".join(f"Item {k}: {this.s[(k * 37) % 700:(k * 37) % 700 + 60]}\n" for k in range(900))
    ids = [t for t in tok.encode(text) if t < cap]
    lens = [300, 1500, 3000, 6000]
    prompts = [ids[i * 50: i * 50 + n] for i, n in enumerate(lens)]
    sp = SamplingParams(temperature=0.0, max_tokens=16, logprobs=5, ignore_eos=True)
    res = {}
    for tag, batch in (("cold", True), ("hit", True), ("single", False)):
        if tag == "cold":
            llm.reset_prefix_cache()
        ps = prompts if batch else prompts[-1:]
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in ps], sp, use_tqdm=False)
        res[tag] = [dict(tokens=list(o.outputs[0].token_ids),
                         lp={str(t): v.logprob for t, v in o.outputs[0].logprobs[0].items()},
                         cached=o.num_cached_tokens) for o in outs]
    json.dump(res, open(out, "w"))
    print("saved", out)


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    worst, eq, n = 0.0, 0, 0
    for tag in A:
        for x, y in zip(A[tag], B[tag]):
            n += 1
            eq += x["tokens"] == y["tokens"]
            agree = next((i for i, (p, q) in enumerate(zip(x["tokens"], y["tokens"])) if p != q), 16)
            d = max(abs(v - y["lp"][t]) for t, v in x["lp"].items() if t in y["lp"])
            worst = max(worst, d)
            print(f"{tag}: tokens_equal={x['tokens'] == y['tokens']} agree={agree} "
                  f"dlp_first={d:.4f} cached={x['cached']}/{y['cached']}")
    print(f"RESULT tokens_equal {eq}/{n} max_dlp_first {worst:.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        cmp(sys.argv[2], sys.argv[3])
