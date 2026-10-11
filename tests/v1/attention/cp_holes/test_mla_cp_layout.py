# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the MSA_MLA_CP pieces that need no GPU.

1. The local key layout (`cp_local_len`, `cp_local_pos`) and the 16-key page
   views (`page_starts`, `page_table`) agree with vLLM's PCP slot mapping
   (torch port `vllm_slot_mapping`, itself checked against the Triton kernel
   in test_kv_placement.py): local key j of a request sits in cache slot
   bt[j // B] * B + j % B and holds position cp_local_pos(j).
2. `reduce_replicated` and `allgather_merge` over gloo (world 2 / 3 / 4) give
   every rank the LSE merge of all ranks' partials.

    python tests/v1/attention/cp_holes/test_mla_cp_layout.py
"""

from __future__ import annotations

import math
import os
import socket
import sys
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_layout() -> int:
    from vllm.v1.attention.backends.mla.msa_mla_cp import PAGE, page_starts, page_table
    from vllm.v1.attention.cp_holes.kv_store import vllm_slot_mapping
    from vllm.v1.attention.cp_holes.mla import cp_local_len, cp_local_pos

    n = 0
    g = torch.Generator().manual_seed(0)
    for d in (1, 2, 3, 4, 8):
        for u, B in ((16, 16), (16, 64), (32, 64), (64, 64), (128, 128)):
            for E in (1, 15, 16, 200, 1000, 3333):
                nb = -(-E // (B * d))
                bt = torch.randperm(nb + 5, generator=g)[:nb].to(torch.int32)
                pos = torch.arange(E)
                for r in range(d):
                    sm = vllm_slot_mapping(bt, pos, B, r, d, u)
                    mine = sm >= 0
                    L = int(cp_local_len(E, d, r, u))
                    assert L == int(mine.sum()), (d, u, B, E, r)
                    j = np.arange(L)
                    p = cp_local_pos(j, d, r, u)
                    assert (p == pos[mine].numpy()).all()
                    slot = bt.long()[j // B] * B + torch.from_numpy(j % B)
                    assert torch.equal(slot, sm[mine]), (d, u, B, E, r)
                    # 16-key pages: start positions and page ids
                    for s0 in (0, 16 * (L // 48)):
                        ln = L - s0
                        kbp = page_starts([s0], [ln], d, r, u)[0]
                        npg = -(-ln // PAGE)
                        assert (kbp[npg:] >= 1 << 30).all()
                        for k in range(npg):
                            jj = np.arange(s0 + PAGE * k, min(L, s0 + PAGE * (k + 1)))
                            assert (cp_local_pos(jj, d, r, u) == kbp[k] + jj - jj[0]).all()
                    pt = page_table(bt[None], -(-L // PAGE), B)[0].long()
                    ppb = B // PAGE
                    for k in range(-(-L // PAGE)):
                        jj = np.arange(PAGE * k, min(L, PAGE * (k + 1)))
                        assert torch.equal(pt[k] * PAGE + torch.from_numpy(jj % PAGE),
                                           sm[mine][jj]), (d, u, B, E, r, k, ppb)
                    n += 1
    return n


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank, world, port, q):
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world)
        from vllm.v1.attention.cp_holes.local_attn import merge_attn_states_torch
        from vllm.v1.attention.cp_holes.mla import allgather_merge, reduce_replicated
        from vllm.v1.attention.cp_holes.schemes import CPComm

        comm = CPComm(dist.group.WORLD, rank, world)
        m, H, V = 37, 4, 8
        parts_o, parts_l = [], []
        for x in range(world):
            gx = torch.Generator().manual_seed(100 + x)
            o = torch.randn(m, H, V, generator=gx)
            lse = torch.randn(m, H, generator=gx) * 3
            lse[x::5] = -math.inf  # rows with no local key on rank x
            o[x::5] = 0
            parts_o.append(o)
            parts_l.append(lse)
        ref_o, ref_l = merge_attn_states_torch(parts_o, parts_l)
        cnt = [(x + 1) * m // world - x * m // world for x in range(world)]
        got = reduce_replicated(comm, cnt, parts_o[rank], parts_l[rank], torch.float32)
        e1 = float((got - ref_o).abs().max())
        go, gl = allgather_merge(comm, parts_o[rank], parts_l[rank])
        fin = torch.isfinite(ref_l)
        e2 = max(float((go - ref_o).abs().max()), float((gl[fin] - ref_l[fin]).abs().max()))
        q.put((rank, max(e1, e2), None))
        dist.barrier()
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001
        q.put((rank, None, traceback.format_exc()))


def check_reduce(world: int) -> float:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=_worker, args=(r, world, port, q)) for r in range(world)]
    for p in ps:
        p.start()
    res = [q.get() for _ in range(world)]
    for p in ps:
        p.join()
    for _, _, tb in res:
        if tb:
            raise AssertionError(tb)
    return max(e for _, e, _ in res)


def test_layout():
    assert check_layout() > 0


def test_reduce():
    for w in (2, 3, 4):
        assert check_reduce(w) < 1e-5


if __name__ == "__main__":
    n = check_layout()
    print(f"layout: {n} (d, u, B, E, rank) cases OK", flush=True)
    ok = True
    for w in (2, 3, 4):
        e = check_reduce(w)
        print(f"reduce world={w}: max err {e:.2e}", flush=True)
        ok &= e < 1e-5
    print("RESULT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
