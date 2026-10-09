# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-vLLM one-GPU check of the CP_HOLES backend (CP world size 1).

Runs holes/core's end-to-end check (`tests/v1/holes/gpu_holes_e2e.py`: holes
made in a cached prefix, recomputed, compared with a full recompute) with the
attention backend set to CP_HOLES. With one GPU the CP group has one rank, so
this exercises the backend inside vLLM (metadata builder on holes/core's
`hole_segments`, paged store, cache writes, decode steps) and, with
`--cp-dense sharded`, the model-runner path of the sharded dense mode
(input sharding and the logits-row gather are identities at world 1).

    gpurun env PYTHONPATH=<worktree> <venv>/bin/python gpu_cp_holes_e2e.py \
        --cp-scheme passq --cp-dense sharded --eager --dtype float32 \
        --gpu-util 0.6
"""

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "..", "..", "holes", "gpu_holes_e2e.py")


def main():
    argv = sys.argv[1:]
    scheme, dense = "adaptive", "replicated"
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--cp-scheme":
            scheme = argv[i + 1]
            i += 2
        elif argv[i] == "--cp-dense":
            dense = argv[i + 1]
            i += 2
        else:
            rest.append(argv[i])
            i += 1
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # in-process engine
    os.environ["VLLM_CP_HOLES_SCHEME"] = scheme
    os.environ["VLLM_CP_HOLES_DENSE"] = dense
    from vllm.v1.attention.backends.cp_holes import register
    register()
    print(f"CP_HOLES scheme={scheme} dense={dense} local_attn="
          f"{os.environ.get('CP_HOLES_LOCAL_ATTN', 'tiled')}", flush=True)
    sys.argv = [E2E] + rest + ["--backend", "CUSTOM"]
    runpy.run_path(E2E, run_name="__main__")


if __name__ == "__main__":
    main()
