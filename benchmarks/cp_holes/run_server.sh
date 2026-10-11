#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Correctness + per-scheme step time of the CP_HOLES schemes on a multi-GPU
# node (NCCL). Run from the vLLM fork root, inside the vLLM environment.
#   benchmarks/cp_holes/run_server.sh [out.jsonl]
# Env overrides for a smaller sweep (defaults = the full one):
#   CP_BENCH_D="1 2 4 8" CP_BENCH_E="32768 131072" CP_BENCH_M="2048 16384"
#   CP_BENCH_LAYOUTS="rate tail deep_small tiny8 hole_only multi"
# A d above the visible GPU count is skipped, so one GPU runs the d = 1
# timing only (the --check runs need d >= 2).
# Record `nvidia-smi topo -m` and `nvidia-smi topo -p2p r` next to the output:
# the link class (NVLink / PCIe P2P / host) decides the ring vs pass-Q choice.
set -eu
OUT=${1:-cp_bench_$(date +%Y%m%d_%H%M).jsonl}
NG=$(nvidia-smi -L | wc -l)
B=benchmarks/cp_holes/bench_cp_holes.py
DS=${CP_BENCH_D:-"1 2 4 8"}
ES=${CP_BENCH_E:-"32768 131072"}
MS=${CP_BENCH_M:-"2048 16384"}
LAYOUTS=${CP_BENCH_LAYOUTS:-"rate tail deep_small tiny8 hole_only multi"}
nvidia-smi topo -m > "${OUT%.jsonl}.topo.txt" || true
nvidia-smi topo -p2p r >> "${OUT%.jsonl}.topo.txt" 2>/dev/null || true
for d in $DS; do
  [ "$d" -ge 2 ] && [ "$d" -le "$NG" ] || continue
  torchrun --nproc-per-node "$d" $B --check
  torchrun --nproc-per-node "$d" $B --check --hkv 2   # Ulysses d > Hkv at d=4,8
  torchrun --nproc-per-node "$d" $B --check --local-attn msa --store paged
done
for d in $DS; do
  [ "$d" -le "$NG" ] || continue
  for E in $ES; do
    for m in $MS; do
      # the CP_HOLES backend path (MSA, paged in place) and the PyTorch
      # stand-in for comparison
      # shellcheck disable=SC2086  # LA and LAYOUTS are word lists
      for LA in "--local-attn msa --store paged" "--local-attn tiled"; do
        torchrun --nproc-per-node "$d" $B --E $E --m $((m)) $LA \
          --layouts $LAYOUTS --out "$OUT"
      done
    done
  done
done
echo "results: $OUT"
