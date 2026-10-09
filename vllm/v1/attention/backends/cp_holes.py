# SPDX-License-Identifier: Apache-2.0
"""CP_HOLES: a context-parallel attention backend for holed prefills.

Plugs the schemes of `vllm.v1.attention.cp_holes` into vLLM's attention
backend interface. The CP group is vLLM's prefill context parallel group
(`get_pcp_group()`, `--prefill-context-parallel-size`), which vLLM 0.19 creates
but for which no GPU backend sets `supports_pcp`.

Registration (no change to vLLM's registry file):

    from vllm.v1.attention.backends.cp_holes import register
    register()                         # AttentionBackendEnum.CUSTOM
    # then --attention-backend CUSTOM (or VLLM_ATTENTION_BACKEND=CUSTOM)

Environment knobs (read once per step by the metadata builder):
- VLLM_CP_HOLES_SCHEME: ulysses | passq | ring | allgather | adaptive
  (default adaptive)
- VLLM_CP_HOLES_DENSE: replicated (default) | sharded.
  replicated: every CP rank runs the dense layers on all tokens (stock model
  runner, no input sharding); attention runs the scheme on the rank's own rows
  and all-gathers O, so logits are identical on every rank. This is the
  correctness mode and needs no model-runner change.
  sharded: the model runner feeds each rank only `metadata.scheme.own_rows`
  (see `shard_inputs` / `gather_rows`); attention returns O for those rows.
  This is the performance mode and needs the runner patch.
- VLLM_CP_HOLES_PASSQ_RATIO: the adaptive threshold (default 0.2).
- VLLM_CP_HOLES_MB: micro-batches for Ulysses / pass-Q (default 2).

KV placement:
- passq / ring / adaptive use vLLM's PCP placement unchanged: run with
  block_size = cp_kv_cache_interleave_size = u (KERNEL-ALGO 4.0 wants
  u = 128). The KV manager then treats a logical block as block_size * d
  tokens, exactly the interleave the schemes assume.
- ulysses (head slices of every position) and allgather (replicated) need the
  KV manager to NOT multiply the block size by the PCP size; ulysses also
  needs the KV cache spec built with Hkv / d heads (Hkv replicated over d / Hkv
  ranks when d > Hkv). Both are small core patches, listed in CP-IMPL.md.

Hole ranges: the hole-aware scheduler (branch holes/core) is expected to put
each request's computed ranges on the common metadata as `cp_segments`
(list of [(start, length)] per request). Without it, every request has the
single stock segment [seq_len - query_len, seq_len).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torch.distributed as dist

from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionLayer,
                                       AttentionMetadataBuilder,
                                       AttentionType, CommonAttentionMetadata,
                                       MultipleOf)
from vllm.v1.attention.cp_holes import StepPlan, make_scheme
from vllm.v1.attention.cp_holes.comm import (all_gather_bytes, pack_rows,
                                             unpack_rows)
from vllm.v1.attention.cp_holes.kv_store import (PagedKVStore,
                                                 vllm_local_layout)
from vllm.v1.attention.cp_holes.schemes import PASSQ_RATIO, CPComm

_POSITION_SHARDED = ("passq", "ring", "adaptive")


def _cp_comm() -> CPComm:
    from vllm.distributed import get_pcp_group
    try:
        return CPComm.from_coordinator(get_pcp_group())
    except AssertionError:
        return CPComm(None, 0, 1)


class CPHolesBackend(AttentionBackend):
    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return CPHolesImpl

    @staticmethod
    def get_builder_cls():
        return CPHolesMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int,
                           num_kv_heads: int, head_size: int,
                           cache_dtype_str: str = "auto") -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


@dataclass
class CPHolesMetadata:
    num_actual_tokens: int
    plan: StepPlan
    scheme: object
    block_tables: list[torch.Tensor]
    local_pos: list[torch.Tensor]
    present: list[torch.Tensor]
    dense: str


class CPHolesMetadataBuilder(AttentionMetadataBuilder[CPHolesMetadata]):
    reorder_batch_threshold: ClassVar[Optional[int]] = None

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        mc, pc = vllm_config.model_config, vllm_config.parallel_config
        self.hq = mc.get_num_attention_heads(pc)
        self.hkv = mc.get_num_kv_heads(pc)
        self.dh = mc.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.interleave = pc.cp_kv_cache_interleave_size
        self.comm = _cp_comm()
        self.scheme_name = os.environ.get("VLLM_CP_HOLES_SCHEME", "adaptive")
        self.dense = os.environ.get("VLLM_CP_HOLES_DENSE", "replicated")
        self.ratio = float(os.environ.get("VLLM_CP_HOLES_PASSQ_RATIO",
                                          PASSQ_RATIO))
        self.n_mb = int(os.environ.get("VLLM_CP_HOLES_MB", "2"))
        if self.scheme_name in _POSITION_SHARDED and self.comm.world > 1:
            assert self.block_size == self.interleave, (
                "CP_HOLES position-sharded schemes need block_size == "
                "cp_kv_cache_interleave_size (= u)")

    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> CPHolesMetadata:
        cam = common_attn_metadata
        n = cam.num_reqs
        seq_lens = cam.seq_lens_cpu[:n].tolist()
        qsl = cam.query_start_loc_cpu[:n + 1].tolist()
        segs = getattr(cam, "cp_segments", None)
        if segs is None:
            segs = [[(seq_lens[i] - (qsl[i + 1] - qsl[i]),
                      qsl[i + 1] - qsl[i])] for i in range(n)]
        d = self.comm.world
        sharded = self.scheme_name in _POSITION_SHARDED
        plan = StepPlan([list(s) for s in segs], d=d,
                        u=self.interleave if sharded else 128)
        kw = {}
        if self.scheme_name == "adaptive":
            kw["threshold"] = self.ratio
        elif self.scheme_name in ("ulysses", "passq"):
            kw["n_mb"] = self.n_mb
        scheme = make_scheme(self.scheme_name, plan, self.comm, self.hq,
                             self.hkv, self.dh, **kw)
        bt_cpu = cam.block_table_tensor[:n].cpu()
        w, r = (d, self.comm.rank) if sharded else (1, 0)
        tables, lpos, present = [], [], []
        for i in range(n):
            e = max(seq_lens[i], plan.e_step[i])
            nb = -(-e // (self.block_size * w))
            bt, lp = vllm_local_layout(bt_cpu[i], nb, self.block_size, r, w,
                                       self.interleave if sharded else
                                       self.block_size)
            comp = torch.zeros(e + 1, dtype=torch.bool)
            for s, ln in plan.segments[i]:
                comp[s:s + ln] = True
            ok = (lp >= 0) & (lp < e)
            pr = ok.clone()
            pr[ok] = ~comp[lp[ok]]
            dev = self.device
            tables.append(bt.to(dev))
            lpos.append(torch.where(ok, lp, torch.full_like(lp, -1)).to(dev))
            present.append(pr.to(dev))
        return CPHolesMetadata(num_actual_tokens=cam.num_actual_tokens,
                               plan=plan, scheme=scheme, block_tables=tables,
                               local_pos=lpos, present=present,
                               dense=self.dense)


class CPHolesImpl(AttentionImpl[CPHolesMetadata]):
    supports_pcp: bool = True

    def __init__(self, num_heads: int, head_size: int, scale: float,
                 num_kv_heads: int, alibi_slopes=None, sliding_window=None,
                 kv_cache_dtype: str = "auto", logits_soft_cap=None,
                 attn_type: str = AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, **kwargs) -> None:
        assert alibi_slopes is None and sliding_window is None
        assert kv_cache_dtype == "auto", "CP_HOLES: fp8 KV not supported yet"
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads

    def forward(self, layer: AttentionLayer, query: torch.Tensor,
                key: torch.Tensor, value: torch.Tensor,
                kv_cache: torch.Tensor, attn_metadata: CPHolesMetadata,
                output: Optional[torch.Tensor] = None,
                output_scale=None, output_block_scale=None) -> torch.Tensor:
        assert output is not None
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        md = attn_metadata
        n = md.num_actual_tokens
        scheme = md.scheme
        q, k, v = query[:n], key[:n], value[:n]
        q = q.view(n, -1, self.head_size)
        k = k.view(n, -1, self.head_size)
        v = v.view(n, -1, self.head_size)
        store = PagedKVStore(kv_cache, md.block_tables, md.local_pos,
                             md.present)
        out = output[:n].view(n, -1, self.head_size)
        if md.dense == "sharded":
            out.copy_(scheme.forward(q, k, v, store, self.scale))
            return output
        own = scheme.own_rows.to(q.device)
        o_own = scheme.forward(q[own], k[own], v[own], store, self.scale)
        out.copy_(gather_rows(scheme, o_own, n))
        return output


def gather_rows(scheme, x_own: torch.Tensor, m: int) -> torch.Tensor:
    """All-gather per-rank rows (scheme.own_rows order) into all m rows."""
    comm = scheme.comm
    tail = tuple(x_own.shape[1:])
    if comm.world == 1:
        out = x_own.new_empty((m, ) + tail)
        out[scheme.own_rows.to(x_own.device)] = x_own
        return out
    rows_all = scheme.rows_by_rank
    from vllm.v1.attention.cp_holes.comm import row_bytes
    rb = row_bytes(tail, x_own.dtype)
    bufs, w = all_gather_bytes(pack_rows([x_own]).reshape(-1),
                               [int(r.numel()) * rb for r in rows_all],
                               comm.group)
    out = x_own.new_empty((m, ) + tail)
    for r, b in zip(rows_all, bufs):
        (part, ) = unpack_rows(b, [(tail, x_own.dtype)]) if r.numel() else \
            (x_own.new_empty((0, ) + tail), )
        out[r.to(x_own.device)] = part
    return out


def shard_inputs(scheme, *tensors: torch.Tensor) -> list[torch.Tensor]:
    """Model-runner hook (sharded dense mode): keep the rank's own rows of
    per-token inputs (input_ids, positions, ...)."""
    own = scheme.own_rows
    return [t[own.to(t.device)] for t in tensors]


def register() -> None:
    """Register CP_HOLES as AttentionBackendEnum.CUSTOM."""
    from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                     register_backend)
    register_backend(AttentionBackendEnum.CUSTOM,
                     "vllm.v1.attention.backends.cp_holes.CPHolesBackend")
