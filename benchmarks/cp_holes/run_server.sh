#!/bin/bash
# Correctness + per-scheme step time of the CP_HOLES schemes on a multi-GPU
# node (NCCL). Run from the vLLM fork root, inside the vLLM environment.
#   benchmarks/cp_holes/run_server.sh [out.jsonl]
# Record `nvidia-smi topo -m` and `nvidia-smi topo -p2p r` next to the output:
# the link class (NVLink / PCIe P2P / host) decides the ring vs pass-Q choice.
set -eu
OUT=${1:-cp_bench_$(date +%Y%m%d_%H%M).jsonl}
NG=$(nvidia-smi -L | wc -l)
B=benchmarks/cp_holes/bench_cp_holes.py
nvidia-smi topo -m > "${OUT%.jsonl}.topo.txt" || true
nvidia-smi topo -p2p r >> "${OUT%.jsonl}.topo.txt" 2>/dev/null || true
for d in 2 4 8; do
  [ "$d" -le "$NG" ] || continue
  torchrun --nproc-per-node "$d" $B --check
  torchrun --nproc-per-node "$d" $B --check --hkv 2   # Ulysses d > Hkv at d=4,8
  torchrun --nproc-per-node "$d" $B --check --local-attn msa --store paged
done
for d in 1 2 4 8; do
  [ "$d" -le "$NG" ] || continue
  for E in 32768 131072; do
    for m in 2048 16384; do
      # the CP_HOLES backend path (MSA, paged in place) and the PyTorch
      # stand-in for comparison
      for LA in "--local-attn msa --store paged" "--local-attn tiled"; do
        torchrun --nproc-per-node "$d" $B --E $E --m $((m)) $LA \
          --layouts rate tail deep_small tiny8 hole_only multi --out "$OUT"
      done
    done
  done
done
echo "results: $OUT"
