# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare measured segmented launches with sum-of-segments pricing (CPU only).

Reads a profiler bundle's tp<N>/segmented_attention.csv (feat/prof-segments) and
attention.csv, and for every multi-segment shot reports, per backend:
  sum_t1   sum of that backend's own single-segment times T1(q, s)
  sum_attn sum of attention.csv lookups (the simulator's default per-segment
           pricing, `_lookup_batch_attention`, FLASH_ATTN profile)
  roof     the meta's fitted roofline (all rows, not held out)
Run with PYTHONPATH=<profiler repo>.
"""
import argparse
import collections
import csv
import json
import sys

import numpy as np
import yaml

from profiler.core import segmented as S


def attn_table(path):
    pts = {}
    for r in csv.DictReader(open(path)):
        if int(r["n_decode"]) == 0 and int(r["prefill_chunk"]) > 0:
            pts.setdefault(int(r["prefill_chunk"]), {})[int(r["kv_prefill"])] = float(r["time_us"])
    return S.T1Table((q, s, t) for q, d in pts.items() for s, t in d.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--tp", default="1")
    args = ap.parse_args()
    root = f"{args.bundle}/tp{args.tp}"
    rows = S.read_rows(__import__("pathlib").Path(f"{root}/segmented_attention.csv"))
    meta = yaml.safe_load(open(f"{args.bundle}/meta.yaml"))
    fit = meta.get("segmented_attention_fit", {})
    at = attn_table(f"{root}/attention.csv")
    by_b = collections.defaultdict(list)
    for r in rows:
        by_b[r["backend"]].append(r)
    t1 = {b: S.T1Table((r["layout"][0][0], r["layout"][0][1], r["time_us"]) for r in rs if len(r["layout"]) == 1)
          for b, rs in by_b.items()}
    meas = {b: {tuple(r["layout"]): r["time_us"] for r in rs} for b, rs in by_b.items()}
    out = {}
    for b, rs in by_b.items():
        f = fit.get(b, {}).get("per_tp", {}).get(args.tp)
        cls = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rs:
            lay = r["layout"]
            if len(lay) < 2:
                continue
            small = any(q <= 256 for q, s in lay)
            c = f"n_seg {'2' if len(lay) == 2 else '3-17'}, {'has q<=256' if small else 'all q>256'}"
            m = r["time_us"]
            for name, pred in (("sum_t1", sum(t1[b](q, s) for q, s in lay)),
                               ("sum_attn", sum(at(q, s) for q, s in lay)),
                               ("roof", S.roofline(lay, t1[b], max(0.0, f["t0_us"]), f["pair_rate"], f["read_rate"])
                                if f else float("nan"))):
                for k in (c, "all multi"):
                    cls[k][name].append(pred / m - 1)
        out[b] = {k: {n: dict(n=len(v), signed_mean=float(np.mean(v)), p50_abs=float(np.median(np.abs(v))),
                                p90_abs=float(np.quantile(np.abs(v), .9))) for n, v in d.items()}
                  for k, d in cls.items()}
    # backend vs backend on identical shots
    bs = sorted(meas)
    pairs = {}
    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            a, c = bs[i], bs[j]
            common = [k for k in meas[a] if k in meas[c]]
            rr = collections.defaultdict(list)
            for k in common:
                small = any(q <= 256 for q, s in k)
                cl = "single" if len(k) == 1 else ("multi, has q<=256" if small else "multi, all q>256")
                rr[cl].append(meas[c][k] / meas[a][k])
                rr["all"].append(meas[c][k] / meas[a][k])
            pairs[f"{c} / {a}"] = {k: dict(n=len(v), median=float(np.median(v)), p10=float(np.quantile(v, .1)),
                                           p90=float(np.quantile(v, .9)), total_ratio=None) for k, v in rr.items()}
            tot_a = sum(meas[a][k] for k in common)
            tot_c = sum(meas[c][k] for k in common)
            pairs[f"{c} / {a}"]["all"]["total_ratio"] = tot_c / tot_a
    json.dump(dict(pricing=out, backends=pairs), sys.stdout, indent=1)


if __name__ == "__main__":
    main()
