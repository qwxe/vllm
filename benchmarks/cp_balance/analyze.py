# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Imbalance / eta tables from bench_balance.py sweep output (CPU only).

Per (step, d, scheme, partitioner):
  A_r      attention kernel time of rank r (ring: sum over its d steps)
  T_att    max_r A_r; ring also reports the lockstep time sum_t max_r T(r, t)
  imb      T_att / mean_r A_r, eta = 1 / imb               (balance only)
  eff      T1 / (d * T_att)                                 (balance x shard efficiency)
  layer    dense_r = dense(n_r) (measured curve) + attention;
           split  = max_r dense_r + max_r A_r, fused = max_r (dense_r + A_r);
           eff_layer = (dense(m) + T1) / (d * layer time)
Set aggregates are work-weighted: eta = sum(ideal) / sum(T).
"""

import argparse
import collections
import json

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(line) for line in open(p)]
    return rows


class Dense:
    def __init__(self, path):
        D = [json.loads(line) for line in open(path)]
        self.n = np.array([0] + [r["n"] for r in D], float)
        self.t = np.array([0.0] + [r["t_us"] for r in D])

    def __call__(self, n):
        if n <= 0:
            return 0.0
        return float(np.interp(n, self.n, self.t))


def group(rows):
    g = collections.defaultdict(list)
    t1 = {}
    for r in rows:
        key = (r["layout"], r.get("req"), r.get("stepi", 0), r["d"]) if r["scheme"] != "t1" else None
        if r["scheme"] == "t1":
            # t1 rows carry d = 1; a real step differs per d, so key on its shape too
            t1.setdefault((r["layout"], r.get("req"), r.get("stepi", 0), r["m"], r["E"]), []).append(r["t_us"])
            continue
        g[key + (r["scheme"], r["part"])].append(r)
    t1 = {k: float(np.median(v)) for k, v in t1.items()}
    return g, t1


def metrics(g, t1, dense, cal):
    out = []
    for key, rs in g.items():
        layout, req, stepi, d, scheme, part = key
        T1 = t1[(layout, req, stepi, rs[0]["m"], rs[0]["E"])]
        m, s = rs[0]["m"], rs[0]["set"]
        A = np.zeros(d)
        P = np.zeros(d)  # model prediction (attention a + b * pairs)
        nrows = np.zeros(d)
        per_step = collections.defaultdict(list)
        for r in rs:
            A[r["rank"]] += r["t_us"]
            if scheme == "ring":
                per_step[r["rstep"]].append(r["t_us"])
                P[r["rank"]] += cal["attn_b_us_per_pair"] * r["n_vis"]
                if r["rstep"] == 0:
                    P[r["rank"]] += cal["attn_a_us"] * r["n_rows"]
            elif scheme in ("allgather",):
                P[r["rank"]] = cal["attn_a_us"] * r["n_rows"] + cal["attn_b_us_per_pair"] * (r["sum_pos"] + r["n_rows"])
            if scheme in ("allgather", "ring"):
                nrows[r["rank"]] = r["n_rows"]
        if scheme in ("allgather", "ring"):
            # ranks without rows have no calls
            pass
        else:
            nrows[:] = [m * (i + 1) // d - m * i // d for i in range(d)]
        Tatt = A.max()
        mean = A.sum() / d
        rec = dict(set=s, layout=layout, req=req, stepi=stepi, d=d, scheme=scheme, part=part, m=m, E=rs[0]["E"],
                   nseg=rs[0]["nseg"], T1=T1, Tatt=Tatt, mean=mean, imb=Tatt / mean, eff=T1 / (d * Tatt))
        if scheme == "ring":
            lock = sum(max(v) for v in per_step.values())
            rec.update(Tlock=lock, imb_lock=lock / mean, eff_lock=T1 / (d * lock))
        if P.sum() > 0:
            rec["imb_pred"] = P.max() / (P.sum() / d)
        dn = np.array([dense(x) for x in nrows])
        split = dn.max() + Tatt
        fused = (dn + A).max()
        ideal_layer = (dn.sum() + A.sum()) / d
        base = dense(m) + T1
        rec.update(dense_max=dn.max(), split=split, fused=fused, ideal_layer=ideal_layer,
                   eta_split=ideal_layer / split, eta_fused=ideal_layer / fused,
                   eff_split=base / (d * split), eff_fused=base / (d * fused), base_layer=base)
        out.append(rec)
    return out


def agg(ms, keyf):
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in ms:
        k = keyf(r)
        a = acc[k]
        T = r.get("Tlock", r["Tatt"])
        a["mean"] += r["mean"]
        a["T"] += T
        a["Tmax"] += r["Tatt"]
        a["T1d"] += r["T1"] / r["d"]
        a["ideal_layer"] += r["ideal_layer"]
        a["split"] += r["split"]
        a["fused"] += r["fused"]
        a["base_d"] += r["base_layer"] / r["d"]
        a["n"] += 1
        if "imb_pred" in r:
            a["pred_err"] += abs(r["imb_pred"] - r["imb"])
            a["npred"] += 1
    res = {}
    for k, a in acc.items():
        res[k] = dict(eta=a["mean"] / a["T"], eta_rank=a["mean"] / a["Tmax"], eff=a["T1d"] / a["T"],
                      eta_split=a["ideal_layer"] / a["split"], eta_fused=a["ideal_layer"] / a["fused"],
                      eff_split=a["base_d"] / a["split"], eff_fused=a["base_d"] / a["fused"], n=int(a["n"]),
                      pred_err=(a["pred_err"] / a["npred"]) if a["npred"] else None)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--dense", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--out", required=True, help="per-config metrics jsonl")
    args = ap.parse_args()
    rows = load(args.files)
    g, t1 = group(rows)
    ms = metrics(g, t1, Dense(args.dense), json.load(open(args.cal)))
    with open(args.out, "w") as fh:
        for r in ms:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(ms)} configs -> {args.out}")


if __name__ == "__main__":
    main()
