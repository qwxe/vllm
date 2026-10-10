# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sum-of-segments pricing vs one measured MSA launch (CPU only).

T1(q, s) comes from the calib phase's single-piece grid (MSA auto, same harness,
bilinear in the simulator's bracket blend). Every all-gather rank call of the
sweeps is one launch over several pieces; its pieces are rebuilt from the
layout and the partitioner, then priced as sum_i T1(q_i, s_i) (feat/sim-segments
`--attn-seg-combine sum`) and as max_i T1 and the a + b i model.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import partitioners as P  # noqa: E402


def bracket(vals, x):
    n = len(vals)
    if n == 1 or x <= vals[0]:
        return 0, 0, 0.0
    i = 0
    while i < n and vals[i] <= x:
        i += 1
    lo, hi = (n - 2, n - 1) if i >= n else (i - 1, i)
    return lo, hi, (x - vals[lo]) / (vals[hi] - vals[lo])


class T1:
    def __init__(self, pts):
        rows = collections.defaultdict(dict)
        for q, s, t in pts:
            rows[q][s] = t
        self.qs = sorted(rows)
        self.lines = {q: (sorted(r), [r[s] for s in sorted(r)]) for q, r in rows.items()}

    def line(self, q, s):
        ss, ts = self.lines[q]
        lo, hi, t = bracket(ss, s)
        return ts[lo] + t * (ts[hi] - ts[lo])

    def __call__(self, q, s):
        lo, hi, t = bracket(self.qs, q)
        a = self.line(self.qs[lo], s)
        return a if hi == lo else a + t * (self.line(self.qs[hi], s) - a)


def pieces(qpos):
    if len(qpos) == 0:
        return []
    brk = np.nonzero(np.diff(qpos) != 1)[0] + 1
    st = np.concatenate([[0], brk])
    en = np.concatenate([brk, [len(qpos)]])
    return [(int(e - s), int(qpos[s])) for s, e in zip(st, en)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--cal", required=True)
    ap.add_argument("--sweep", nargs="+", required=True)
    ap.add_argument("--layouts", nargs="+", required=True)
    args = ap.parse_args()
    cal = json.load(open(args.cal))
    t1 = T1([(r["q"], r["s"], r["t_us"]) for r in map(json.loads, open(args.calib)) if r["kind"] == "single"])
    lay = {}
    for f in args.layouts:
        for s in json.load(open(f)):
            for d in (s.get("d_list") or [2, 4, 8]):
                lay[(s["name"], s.get("stepi", 0), d)] = s
    cache = {}
    res = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in args.sweep:
        for line in open(f):
            r = json.loads(line)
            if r["scheme"] != "allgather":
                continue
            key = (r["layout"], r.get("stepi", 0), r["d"])
            st = lay[key]
            ck = key + (r["part"],)
            if ck not in cache:
                segs = [tuple(x) for x in st["segs"]]
                qpos = P.computed_positions(segs)
                cache[ck] = (qpos, P.partition(r["part"], segs, r["d"], cal=cal))
            qpos, rows = cache[ck]
            pc = pieces(qpos[rows[r["rank"]]])
            if len(pc) < 2:
                continue
            m = r["t_us"]
            small = any(q <= 256 for q, s in pc)
            cls = f"{'2' if len(pc) == 2 else '3+'} pieces, {'has q<=256' if small else 'all q>256'}"
            preds = dict(sum=sum(t1(q, s) for q, s in pc), max=max(t1(q, s) for q, s in pc),
                         ab=cal["attn_a_us"] * r["n_rows"] + cal["attn_b_us_per_pair"] * (r["sum_pos"] + r["n_rows"]))
            for k in (cls, "all multi-piece launches"):
                for n, p in preds.items():
                    res[k][n].append(p / m - 1)
    out = {}
    for k, d in sorted(res.items()):
        out[k] = {n: dict(n=len(v), signed_mean=round(float(np.mean(v)), 3), p50_abs=round(float(np.median(np.abs(v))), 3),
                          p90_abs=round(float(np.quantile(np.abs(v), .9)), 3)) for n, v in d.items()}
        print(k.ljust(28), "  ".join(f"{n}: {x['signed_mean']:+.3f} / {x['p50_abs']:.3f} / {x['p90_abs']:.3f} (n {x['n']})"
                                      for n, x in out[k].items()))
    json.dump(out, open(os.environ.get("OUT", "/dev/null"), "w"), indent=1)


if __name__ == "__main__":
    main()
