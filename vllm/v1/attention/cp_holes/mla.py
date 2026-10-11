# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-split CP for MLA prefills with holes: pass-Q over an interleaved
latent cache (MLA-MSA.md section 4).

The MLA KV cache is one latent per token (kv_lora_rank + rope dims) shared by
all heads, so head-split CP (Ulysses) gives no KV-capacity gain; only a
position split shards it. Placement is CP_HOLES' interleave: position p lives
on rank (p // u) % d. Per step and layer:

1. one all-to-all sends Q (nope + rope, model dtype) of the rank's own rows to
   every rank, and each new latent row to its home rank only;
2. every rank writes the latent it homes, gathers its homed keys of
   [0, E_step) per request, up-projects them through kv_b_proj (E / d keys per
   rank: the up-projection is split d ways, unlike ring / all-gather, which
   would up-project every key on every rank), and runs the MSA-MLA compute
   path for all rows of the step over those keys (per-key positions,
   mask kpos <= qpos);
3. a second all-to-all returns (O in the model dtype, LSE fp32) to the row
   owners, which merge the d partials by LSE.

Interleaving balances keys and causal work for any hole layout (BALANCE-4050:
pass-Q interleaved eta .98-1.00), and contiguous equal-token rows balance the
dense layers, so no partitioner is needed. The plan (row ownership, key
positions, MSA work list) is built once per step on the host; `forward` runs
per layer.

`MlaLatentStore` is a reference store (dense per-request buffers indexed by
position). In vLLM the store is the rank's paged latent cache under PCP's
interleaved slot mapping (KPOS_MODE 1 over interleaved blocks); see
MLA-MSA.md section 4 for what is left to wire.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.distributed as dist

from vllm.v1.attention.ops.msa.api_mla import (COMPUTE, msa_mla_attention,
                                               plan_msa_mla_rows)

from .comm import all_to_all_bytes, pack_rows, row_bytes, unpack_rows, wait_all
from .local_attn import merge_attn_states
from .plan import StepPlan, home_rank, rows_contiguous
from .schemes import CPComm

PAGE = 16


class MlaLatentStore:
    """Reference latent store of one rank: per request a dense buffer over
    positions [0, cap); only the positions this rank homes are meaningful."""

    def __init__(self, caps: Sequence[int], dim: int, dtype: torch.dtype,
                 device: torch.device):
        self.buf = [torch.zeros(int(c), dim, dtype=dtype, device=device)
                    for c in caps]

    def write(self, lat: torch.Tensor, pos: torch.Tensor,
              req: torch.Tensor) -> None:
        for r in torch.unique(req).tolist():
            sel = (req == r).nonzero(as_tuple=True)[0]
            self.buf[r][pos[sel]] = lat[sel].to(self.buf[r].dtype)

    def gather(self, idx: Sequence[torch.Tensor]) -> torch.Tensor:
        """idx[r]: positions to read for request r (padding may repeat 0)."""
        return torch.cat([self.buf[r][i] for r, i in enumerate(idx)])


class MlaPassQCP:
    """MLA pass-Q CP over interleaved latent placement (one micro-batch)."""

    name = "mla_passq"

    def __init__(self, plan: StepPlan, comm: CPComm, num_heads: int,
                 kv_lora_rank: int = 512, rope_dim: int = 64,
                 nope_dim: int = 128, v_dim: int = 128,
                 device: torch.device | str = "cuda", mode: str = "auto"):
        assert plan.d == comm.world, (plan.d, comm.world)
        self.plan, self.comm = plan, comm
        self.H, self.L, self.R, self.P, self.V = (num_heads, kv_lora_rank,
                                                  rope_dim, nope_dim, v_dim)
        d, me, u = comm.world, comm.rank, plan.u
        self.rows_by_rank = [x[0] for x in rows_contiguous(plan.m, d, 1)]
        self.own_rows = self.rows_by_rank[me]
        self.cnt = [int(x.numel()) for x in self.rows_by_rank]
        hm = home_rank(plan.qpos[self.own_rows], d, u)
        self.send_sel = [(hm == x).nonzero(as_tuple=True)[0] for x in range(d)]
        recv_rows = []
        for x in range(d):
            rx = self.rows_by_rank[x]
            recv_rows.append(rx[home_rank(plan.qpos[rx], d, u) == me])
        self.recv_new = [int(t.numel()) for t in recv_rows]
        rr = torch.cat(recv_rows) if recv_rows else torch.empty(0, dtype=torch.int64)
        self.new_pos, self.new_req = plan.qpos[rr], plan.qreq[rr]
        # local keys: positions < E_step(i) homed on me, per request, padded
        # to PAGE rows in the dense gather buffer
        self.key_pos, self.gather_idx, kv_lens = [], [], []
        for i in range(plan.num_reqs):
            p = torch.arange(plan.e_step[i], dtype=torch.int64)
            p = p[home_rank(p, d, u) == me]
            n = int(p.numel())
            pad = -(-n // PAGE) * PAGE
            self.key_pos.append(p.numpy())
            self.gather_idx.append(torch.cat([p, torch.zeros(pad - n, dtype=torch.int64)]))
            kv_lens.append(n)
        self.kv_lens = np.asarray(kv_lens, dtype=np.int64)
        nblk = [-(-n // PAGE) for n in kv_lens]
        maxb = max(1, max(nblk) if nblk else 1)
        dbt = np.zeros((plan.num_reqs, maxb), dtype=np.int32)
        off = 0
        for i, nb in enumerate(nblk):
            dbt[i, :nb] = off + np.arange(nb)
            off += nb
        maxk = max(1, maxb * PAGE)
        kp = np.full((plan.num_reqs, maxk), 1 << 30, dtype=np.int32)
        for i, p in enumerate(self.key_pos):
            kp[i, : len(p)] = p
        dev = torch.device(device)
        self.dense_bt = torch.from_numpy(dbt).to(dev)
        self.kpos_t = torch.from_numpy(kp).to(dev)
        self.gather_idx = [g.to(dev) for g in self.gather_idx]
        self.new_pos_d, self.new_req_d = self.new_pos.to(dev), self.new_req.to(dev)
        self.send_sel_d = [s.to(dev) for s in self.send_sel]
        # one MSA plan per step: every row of the step over the local keys
        self.attn_plan = plan_msa_mla_rows(
            plan.qpos.numpy(), plan.qreq.numpy(), self.kv_lens, COMPUTE, num_heads,
            block_size=PAGE, key_pos=self.key_pos, mode=mode, device=dev)

    def _comm_dev(self, t: torch.Tensor) -> torch.device:
        g = self.comm.group
        if g is not None and dist.get_backend(g) == "gloo":
            return torch.device("cpu")
        return t.device

    def forward(self, q: torch.Tensor, lat: torch.Tensor, store: MlaLatentStore,
                up_proj: Callable[[torch.Tensor], torch.Tensor],
                scale: float) -> torch.Tensor:
        """q: [n_own, H, P + R], lat: [n_own, L + R] (this step's new latent of
        the own rows, i.e. kv_c_normed | k_pe). Returns O [n_own, H, V]."""
        d, me, dt = self.comm.world, self.comm.rank, q.dtype
        H, L, R, P, V = self.H, self.L, self.R, self.P, self.V
        cd = self._comm_dev(q)
        rq = row_bytes((H, P + R), dt)
        rl = row_bytes((L + R,), lat.dtype)
        qb = pack_rows([q]).reshape(-1)
        send = [torch.cat([qb, pack_rows([lat[self.send_sel_d[x]]]).reshape(-1)]).to(cd)
                for x in range(d)]
        recv = [self.cnt[x] * rq + self.recv_new[x] * rl for x in range(d)]
        bufs, w = all_to_all_bytes(send, recv, self.comm.group)
        wait_all([w])
        qs, ls = [], []
        for x, b in enumerate(bufs):
            b = b.to(q.device)
            qs.append(b[: self.cnt[x] * rq])
            ls.append(b[self.cnt[x] * rq:])
        (q_all, ) = unpack_rows(torch.cat(qs), [((H, P + R), dt)])
        lat_new = torch.cat(ls)
        if lat_new.numel():
            (lat_new, ) = unpack_rows(lat_new, [((L + R, ), lat.dtype)])
            store.write(lat_new, self.new_pos_d, self.new_req_d)
        # local attention over the homed keys (gather + up-project E / d keys)
        ws = store.gather(self.gather_idx)
        m = q_all.shape[0]
        o = torch.zeros(m, H, V, device=q.device, dtype=dt)
        lse = torch.full((m, H), float("-inf"), device=q.device)
        if ws.shape[0]:
            nb = ws.shape[0] // PAGE
            kvn = up_proj(ws[:, :L]).view(nb, PAGE, H, P + V)
            kpe = ws.view(nb, PAGE, 1, L + R)[..., L:].expand(-1, -1, H, -1)
            msa_mla_attention(q_all[..., :P], q_all[..., P:], kvn[..., :P], kpe,
                              kvn[..., P:], self.dense_bt, self.attn_plan,
                              kblk_pos=self.kpos_t, softmax_scale=scale,
                              out=o, lse=lse)
        # return partial (O, LSE) of every rank's rows to their owner
        send, off = [], 0
        for x in range(d):
            n = self.cnt[x]
            send.append(pack_rows([o[off:off + n], lse[off:off + n]]).reshape(-1).to(cd))
            off += n
        ro = row_bytes((H, V), dt) + row_bytes((H, ), torch.float32)
        bufs, w = all_to_all_bytes(send, [self.cnt[me] * ro] * d, self.comm.group)
        wait_all([w])
        os_, ls_ = [], []
        for b in bufs:
            o_p, l_p = unpack_rows(b.to(q.device), [((H, V), dt), ((H, ), torch.float32)])
            os_.append(o_p)
            ls_.append(l_p)
        return merge_attn_states(os_, ls_)[0].to(dt)


def homed_resident_positions(plan: StepPlan, i: int, rank: int) -> torch.Tensor:
    """Resident (not computed this step) positions of request i < E_step that
    `rank` homes: what its latent store must hold before the step."""
    return plan.resident_homed(i, rank)
