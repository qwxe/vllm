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
# MLA CP (DeepSeek-V2-Lite, MSA_MLA_CP) runs after the GQA sweep; env:
#   CP_MLA_MODEL (default deepseek-ai/DeepSeek-V2-Lite; dummy weights, 1 layer)
#   CP_MLA_SLICE (a real-weight 1-layer slice dir for the correctness runs;
#                 skipped when unset), CP_MLA_U (default 128), CP_SKIP_GQA=1
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
if [ "${CP_SKIP_GQA:-0}" != 1 ]; then
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
fi

# ---- MLA CP (MSA_MLA_CP, pass-Q over the interleaved latent) ----
MLA_MODEL=${CP_MLA_MODEL:-deepseek-ai/DeepSeek-V2-Lite}
MLA_U=${CP_MLA_U:-128}
MLA_OUT=${OUT%.jsonl}_mla.jsonl
T=tests/v1/attention/cp_holes
for d in $DS; do
  [ "$d" -le "$NG" ] || continue
  if [ "$d" -ge 2 ]; then
    # scheme math (gloo, ranks on GPU 0) at u = 128
    python $T/test_mla_cp_gloo.py --world "$d" --steps 8 --u 128
  fi
  if [ -n "${CP_MLA_SLICE:-}" ]; then
    # real-weight slice, fp32: no-CP MSA_MLA reference, then CP over d GPUs
    # (gloo harness, one process per GPU) and the real engine (NCCL)
    [ -f mla_ref.pt ] || python $T/test_mla_cp_model_gloo.py --model "$CP_MLA_SLICE" \
      --world 1 --backend MSA_MLA --u "$MLA_U" --out mla_ref.pt
    python $T/test_mla_cp_model_gloo.py --model "$CP_MLA_SLICE" --world "$d" \
      --u "$MLA_U" --ref mla_ref.pt
    python $T/gpu_mla_cp_e2e.py --model "$CP_MLA_SLICE" --pcp "$d" --u "$MLA_U" \
      --vocab-cap 16384 --eager --dtype float32 --gpu-util 0.6
  fi
  # step time, bf16, one dense layer, dummy weights
  # shellcheck disable=SC2086  # ES is a word list
  python benchmarks/cp_holes/bench_mla_cp.py --model "$MLA_MODEL" --world "$d" \
    --u "$MLA_U" --E $ES --out "$MLA_OUT"
done
# no-CP baseline for the same layouts
# shellcheck disable=SC2086
python benchmarks/cp_holes/bench_mla_cp.py --model "$MLA_MODEL" --world 1 \
  --backend MSA_MLA --u "$MLA_U" --E $ES --out "$MLA_OUT"
echo "results: $OUT $MLA_OUT"
