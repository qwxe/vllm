# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-vLLM holes e2e with MLA CP (MSA_MLA_CP) on a DeepSeek-V2 model.

Runs holes/core's end-to-end check (`tests/v1/holes/gpu_holes_e2e.py`: holes
made in a cached prefix, recomputed, compared with a full recompute) with
`--attention-backend MSA_MLA_CP` and `--prefill-context-parallel-size d`
(interleaved KV placement, block size = interleave size = u). Needs d GPUs
(one PCP worker per GPU, NCCL); `--pcp 1` runs the CP code path at world 1.

    python gpu_mla_cp_e2e.py --model <path> --pcp 2 --u 128 \
        --eager --dtype float32 --gpu-util 0.6 [--vocab-cap 16384]

fp32 needs a model without fused MoE layers on SM 8.x (the 1-layer slice);
bf16 is judged by holes/core's noise rule. On the server, compare with
`--pcp 1` and with `--backend MSA_MLA` (no CP).
"""

import argparse
import json
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "..", "holes", "gpu_holes_e2e.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pcp", type=int, default=1)
    ap.add_argument("--u", type=int, default=128)
    ap.add_argument("--backend", default="MSA_MLA_CP")
    ap.add_argument("--vocab-cap", type=int, default=0)
    args, rest = ap.parse_known_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # scheduler in-process
    os.environ["HOLES_MODEL"] = args.model
    if args.vocab_cap:
        os.environ["HOLES_VOCAB_CAP"] = str(args.vocab_cap)
    # DeepSeek's rope otherwise JIT-compiles a FlashInfer kernel (needs nvcc)
    os.environ.setdefault("HOLES_CUSTOM_OPS", "-rotary_embedding")
    kw = {"block_size": args.u}
    if args.backend == "MSA_MLA_CP":
        kw.update(prefill_context_parallel_size=args.pcp,
                  cp_kv_cache_interleave_size=args.u, cp_kv_placement="interleave")
    os.environ["HOLES_EXTRA_KW"] = json.dumps(kw)
    os.environ["HOLES_KV_BLOCK"] = str(args.u * (args.pcp if args.backend == "MSA_MLA_CP" else 1))
    print(f"MLA CP e2e: backend={args.backend} pcp={args.pcp} u={args.u} kw={kw}", flush=True)
    sys.argv = [E2E] + rest + ["--backend", args.backend]
    runpy.run_path(E2E, run_name="__main__")


if __name__ == "__main__":
    main()
