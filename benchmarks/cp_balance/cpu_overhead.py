# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host (CPU) cost of the CP_HOLES per-step planning, no GPU needed.

Times, per step shape (E, segments, d):
- StepPlan construction;
- the row partitioners (rows_c2e as built, balanced from the simulator);
- each scheme's constructor (Ulysses, pass-Q, ring, all-gather; comm = no group);
- plan_msa_rows for pass-Q / Ulysses (once per step and micro-batch, reused by
  every layer through the store's step cache);
- the host part of MsaLocalAttention for one ring buffer (the ring and the
  all-gather cached pass call it per layer and per ring step: key sort, tables,
  plan_msa_rows).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import partitioners as P  # noqa: E402

from vllm.v1.attention.cp_holes.plan import StepPlan, rows_c2e, home_rank  # noqa: E402
from vllm.v1.attention.cp_holes import schemes as S  # noqa: E402
from vllm.v1.attention.ops.msa.planner import plan_msa_rows  # noqa: E402

torch.set_num_threads(1)


def tmin(fn, n=5):
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best * 1e3


def ring_host(qpos, kp, sms):
    """MsaLocalAttention's host work for one call (keys already sorted per request here)."""
    korder = np.lexsort((kp, np.zeros_like(kp)))
    kp_s = kp[korder]
    bs = 16
    nblk = -(-len(kp_s) // bs)
    tab = np.full((1, nblk * bs), 1 << 30, dtype=np.int64)
    tab[0, :len(kp_s)] = kp_s
    plan_msa_rows(qpos, np.zeros(len(qpos), dtype=np.int64), np.asarray([len(kp_s)]), 8, 4, block_size=bs,
                  key_pos=[kp_s], device="cpu", num_sms=sms, elem_size=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sms", type=int, default=188)
    args = ap.parse_args()
    steps = json.load(open(args.layouts))
    fh = open(args.out, "w")
    for st in steps:
        segs = [tuple(x) for x in st["segs"]]
        qpos = P.computed_positions(segs)
        E = int(qpos[-1]) + 1
        for d in (st.get("d_list") or [2, 4, 8]):
            rec = dict(layout=st["name"], E=E, m=len(qpos), d=d, nseg=len(segs))
            rec["stepplan"] = tmin(lambda: StepPlan(segments=[list(segs)], d=d))
            plan = StepPlan(segments=[list(segs)], d=d)
            rec["c2e"] = tmin(lambda: rows_c2e(plan))
            rec["balanced"] = tmin(lambda: P.partition("balanced", segs, d))
            comm = S.CPComm(None, 0, d)
            for name, cls in (("ulysses", S.UlyssesCP), ("passq", S.PassQCP), ("ring", S.RingCP),
                              ("allgather", S.AllGatherKVCP)):
                try:
                    rec[f"init_{name}"] = tmin(lambda: cls(plan, comm, 32, 8, 128, local_attn=lambda *a: None),
                                               n=3)
                except Exception as e:  # noqa: BLE001
                    rec[f"init_{name}"] = f"err {type(e).__name__}: {e}"
            # pass-Q: all rows of a micro-batch (n_mb = 2 -> half the rows) over interleaved blocks
            half = qpos[: len(qpos) // 2]
            blocks = np.arange(-(-E // 16))
            mine = blocks[((blocks * 16) // 128) % d == 0]
            kb = (mine * 16)[None, :]
            kvl = int(np.clip(E - mine * 16, 0, 16).sum())
            rec["msa_plan_passq_mb"] = tmin(lambda: plan_msa_rows(
                half, np.zeros(len(half), dtype=np.int64), np.asarray([kvl]), 8, 4, block_size=16, kblk_pos=kb,
                device="cpu", num_sms=args.sms, elem_size=2))
            # ring: rank 0's rows (c2e) over rank 0's buffer
            rows = rows_c2e(plan)
            r0 = rows[0].numpy()
            comp = np.zeros(E, dtype=bool)
            comp[qpos] = True
            res = np.nonzero(~comp)[0]
            kp = np.sort(np.concatenate([res[(res // 128) % d == 0], qpos[r0]]))
            rec["ring_host_per_call"] = tmin(lambda: ring_host(qpos[r0], kp, args.sms))
            rec["ring_host_per_layer"] = rec["ring_host_per_call"] * d
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in rec.items()}),
                  flush=True)


if __name__ == "__main__":
    main()
