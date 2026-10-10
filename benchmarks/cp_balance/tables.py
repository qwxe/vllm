# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Markdown tables from analyze.py metrics (CPU only)."""

import argparse
import collections
import json

import numpy as np

PARTS = ["contig", "zigzag", "zigzag_abs", "balanced", "balanced_cal", "balanced_attn", "c2e", "snake"]
FREE = [("passq", "interleave"), ("passq_contig", "contig"), ("ulysses", "heads")]


def load(p):
    return [json.loads(line) for line in open(p)]


def agg(ms, metric):
    """Work-weighted eta over a set of configs: sum(ideal) / sum(T)."""
    num = den = 0.0
    for r in ms:
        if metric == "attn":
            num += r["mean"]
            den += r.get("Tlock", r["Tatt"])
        elif metric == "attn_rank":
            num += r["mean"]
            den += r["Tatt"]
        elif metric == "split":
            num += r["ideal_layer"]
            den += r["split"]
        elif metric == "fused":
            num += r["ideal_layer"]
            den += r["fused"]
        elif metric == "eff":
            num += r["T1"] / r["d"]
            den += r.get("Tlock", r["Tatt"])
        elif metric == "eff_split":
            num += r["base_layer"] / r["d"]
            den += r["split"]
    return num / den if den else float("nan")


def cols():
    c = [("allgather", p) for p in PARTS] + [("ring", p) for p in PARTS] + FREE
    return c


def table_sets(ms, sets, metric, title):
    by = collections.defaultdict(list)
    for r in ms:
        for sname, pred in sets:
            if pred(r):
                by[(sname, r["d"], r["scheme"], r["part"])].append(r)
    out = [f"**{title}**", ""]
    hdr = "| set | d | " + " | ".join(f"{s[:4]}:{p}" for s, p in cols()) + " |"
    out.append(hdr)
    out.append("|" + "---|" * (2 + len(cols())))
    for sname, _ in sets:
        for d in (2, 4, 8):
            vals = []
            for s, p in cols():
                rs = by.get((sname, d, s, p))
                vals.append(f"{agg(rs, metric):.3f}" if rs else "")
            if any(vals):
                out.append(f"| {sname} | {d} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def table_compact(ms, sets, metric, title, scheme_parts):
    by = collections.defaultdict(list)
    for r in ms:
        for sname, pred in sets:
            if pred(r):
                by[(sname, r["d"], r["scheme"], r["part"])].append(r)
    out = [f"**{title}**", ""]
    out.append("| set | d | " + " | ".join(f"{s} {p}" for s, p in scheme_parts) + " |")
    out.append("|" + "---|" * (2 + len(scheme_parts)))
    for sname, _ in sets:
        for d in (2, 4, 8):
            vals = []
            for s, p in scheme_parts:
                rs = by.get((sname, d, s, p))
                vals.append(f"{agg(rs, metric):.3f}" if rs else "")
            if any(vals):
                out.append(f"| {sname} | {d} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def per_layout(ms, d, scheme, metric_key, title, layouts):
    idx = {(r["layout"], r["d"], r["scheme"], r["part"]): r for r in ms}
    sp = [(scheme, p) for p in PARTS] + FREE
    out = [f"**{title}**", ""]
    out.append("| layout | m | E | " + " | ".join(p if s == scheme else s for s, p in sp) + " |")
    out.append("|" + "---|" * (3 + len(sp)))
    for lay in layouts:
        vals = []
        mm = EE = ""
        for s, p in sp:
            r = idx.get((lay, d, s, p))
            if r:
                mm, EE = r["m"], r["E"]
                v = r.get(metric_key) if metric_key in r else r["imb"]
                if s == "ring" and metric_key == "imb_lock":
                    v = r["imb_lock"]
                vals.append(f"{v:.2f}")
            else:
                vals.append("")
        out.append(f"| {lay} | {mm} | {EE} | " + " | ".join(vals) + " |")
    return "\n".join(out)


def pred_quality(ms):
    by = collections.defaultdict(list)
    for r in ms:
        if "imb_pred" in r and r["scheme"] in ("allgather", "ring"):
            m = r["imb_lock"] if r["scheme"] == "ring" else r["imb"]
            by[(r["scheme"], r["part"])].append((r["imb_pred"], m))
    out = ["| scheme | partitioner | n | mean abs err (pred - meas imbalance) | p90 abs | mean signed |",
           "|---|---|---|---|---|---|"]
    for k in sorted(by):
        v = np.array(by[k])
        e = v[:, 0] - v[:, 1]
        out.append(f"| {k[0]} | {k[1]} | {len(v)} | {np.abs(e).mean():.3f} | {np.quantile(np.abs(e), .9):.3f} | "
                   f"{e.mean():+.3f} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", nargs="+")
    ap.add_argument("--order", help="layouts file giving the row order of the per-layout tables")
    args = ap.parse_args()
    ms = []
    for p in args.metrics:
        ms += load(p)
    synth = [("synth all", lambda r: r["set"] == "synth"),
             ("synth holed", lambda r: r["set"] == "synth" and not r["layout"].startswith("ctrl"))]
    real = [("rate112 holed", lambda r: r["set"] == "rate112:holed"),
            ("head448 holed", lambda r: r["set"] == "head448:holed"),
            ("holed steps >1 range", lambda r: r["set"].endswith(":holed") and r["nseg"] > 1),
            ("plain (hit + tail)", lambda r: r["set"].endswith(":plain"))]
    sets = synth + real
    print(table_sets(ms, sets, "attn", "Attention-kernel eta = mean / max rank time (ring: lockstep, sum over ring steps of the max)"))
    print()
    print(table_sets(ms, sets, "split", "Layer eta, split sync (max dense + max attention; dense measured per rank token count)"))
    print()
    print(table_sets(ms, sets, "fused", "Layer eta, fused sync (max over ranks of dense + attention)"))
    print()
    print(table_sets(ms, sets, "eff", "Attention efficiency T1 / (d T_d) (balance x shard efficiency; laptop clocks bias it)"))
    print()
    print(pred_quality(ms))
    print()
    lays = []
    for r in ms:
        if r["set"] == "synth" and r["layout"] not in lays:
            lays.append(r["layout"])
    if args.order:
        names = [x["name"] for x in json.load(open(args.order))]
        lays = [x for x in names if x in lays]
    for d in (2, 4, 8):
        print(per_layout(ms, d, "ring", "imb_lock", f"Synthetic, d={d}: ring lockstep attention imbalance (max/mean)", lays))
        print()
    print(per_layout(ms, 8, "allgather", "imb", "Synthetic, d=8: all-gather attention imbalance", lays))


if __name__ == "__main__":
    main()
