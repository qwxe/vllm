# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Holed prefill steps for bench_balance.py (CPU only).

synth: single steps used at every d (hole count, position, context, small pieces,
       hole-only steps, no-hole controls).
real:  requests sampled from the csim layouts (`holes/layouts_*.json`), chunked
       front first at 2048 * d tokens per step, one step list per d.
"""

import argparse
import json

import numpy as np

A128 = 128


def holes_in(lo, hi, h, H):
    """h holes of H/h tokens evenly spaced in [lo, hi), starts aligned to 128."""
    size = H // h
    span = hi - lo
    out = []
    for i in range(h):
        c = lo + span * (2 * i + 1) // (2 * h)
        s = max(lo, (c - size // 2) // A128 * A128)
        out.append((int(s), int(size)))
    return out


def synth():
    K = 1024
    st = []

    def add(name, segs):
        segs = sorted(segs)
        for (a, n), (b, _) in zip(segs, segs[1:]):
            assert a + n <= b, (name, segs)
        st.append(dict(name=name, set="synth", segs=[list(x) for x in segs]))

    E, T, H = 64 * K, 2048, 6144
    for h in (1, 2, 4, 8, 16):
        add(f"count_h{h}_spread_E64K", holes_in(0, E - T, h, H) + [(E - T, T)])
    for h in (1, 4):
        for nm, lo, hi in (("front", 0, int(0.3 * E)), ("middle", int(0.35 * E), int(0.65 * E)),
                           ("back", int(0.7 * E), E - T)):
            add(f"pos_{nm}_h{h}_E64K", holes_in(lo, hi, h, H) + [(E - T, T)])
    for Ek in (16, 32, 100):
        Ee = Ek * K if Ek != 100 else 100000
        add(f"ctx_E{Ek}K_h4_spread", holes_in(0, Ee - T, 4, H) + [(Ee - T, T)])
    add("small_h8x64_E64K", holes_in(0, E - T, 8, 8 * 64) + [(E - T, T)])
    add("small_h32x64_E64K", holes_in(0, E - T, 32, 32 * 64) + [(E - T, T)])
    add("small_h16x16_E64K", holes_in(0, E - T, 16, 16 * 16) + [(E - T, T)])
    add("holeonly_h1_8K_at32K", [(32 * K, 8192)])
    add("holeonly_h4_spread_to32K", holes_in(0, 32 * K, 4, 8192))
    add("ctrl_tail8K_E64K", [(E - 8192, 8192)])
    add("ctrl_fresh8K", [(0, 8192)])
    add("rate_like_h512at20K_tail8K_E100K", [(20 * K, 512), (100000 - 8192, 8192)])
    return st


def chunk(comp, B):
    """Front-first greedy steps of at most B tokens over the ranges [[s, e), ...]."""
    steps, cur, room = [], [], B
    for s, e in comp:
        while s < e:
            take = min(room, e - s)
            cur.append((s, take))
            s += take
            room -= take
            if room == 0:
                steps.append(cur)
                cur, room = [], B
    if cur:
        steps.append(cur)
    return steps


def real(files, n_holed, n_plain, seed, ds, m_cap):
    rng = np.random.default_rng(seed)
    st = []
    for f, tag in files:
        R = json.load(open(f))["rec"]
        holed = [r for r in R if len(r["comp"]) >= 2 and sum(e - s for s, e in r["comp"]) <= m_cap]
        plain = [r for r in R if len(r["comp"]) == 1 and r["comp"][0][0] > 0
                 and r["comp"][0][1] - r["comp"][0][0] <= m_cap]
        picks = []
        for pool, n, kind in ((holed, n_holed, "holed"), (plain, n_plain, "plain")):
            if not n:
                continue
            pool = sorted(pool, key=lambda r: r["L"])
            # stratified by context length: one pick per equal-size bin
            bins = np.array_split(np.arange(len(pool)), n)
            for b in bins:
                picks.append((pool[int(rng.choice(b))], kind))
        for r, kind in picks:
            for d in ds:
                for i, segs in enumerate(chunk(r["comp"], 2048 * d)):
                    st.append(dict(name=f"{tag}:{kind}:{r['i']}", set=f"{tag}:{kind}", req=r["i"], stepi=i,
                                   L=r["L"], d_list=[d], segs=[list(x) for x in segs]))
    return st


def work(steps, ds_default):
    tot = 0.0
    for s in steps:
        nd = len(s.get("d_list") or ds_default)
        pairs = sum(n * st + n * (n + 1) / 2 for st, n in s["segs"])
        tot += pairs * nd
    return tot * 32 * 512 / 21.8e12  # seconds of attention, one pass at d = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holes-dir", required=True)
    ap.add_argument("--out-synth", required=True)
    ap.add_argument("--out-real", required=True)
    ap.add_argument("--n-holed", type=int, default=8)
    ap.add_argument("--n-plain", type=int, default=2)
    ap.add_argument("--m-cap", type=int, default=24576)
    args = ap.parse_args()
    s = synth()
    json.dump(s, open(args.out_synth, "w"), indent=0)
    files = [(f"{args.holes_dir}/layouts_rate_c112_fcfs.json", "rate112"),
             (f"{args.holes_dir}/layouts_head_rate_c448_W1_s0.json", "head448")]
    r = real(files, args.n_holed, args.n_plain, 0, [2, 4, 8], args.m_cap)
    json.dump(r, open(args.out_real, "w"), indent=0)
    print(f"synth steps {len(s)}, est. one-pass attention {work(s, [2, 4, 8]):.1f} s (x3 d)")
    print(f"real steps {len(r)}, est. one-pass attention {work(r, None):.1f} s")
    multi = sum(1 for x in r if len(x["segs"]) > 1)
    print(f"real steps with >1 range: {multi}")


if __name__ == "__main__":
    main()
