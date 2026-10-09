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
- ulysses runs with `--cp-kv-placement head` (head slices of every position;
  the KV cache spec has Hkv / d heads, 1 replicated head when d > Hkv) and
  allgather with `--cp-kv-placement replicate`. With either, the KV manager,
  slot mapping and capacity see no PCP interleave (ParallelConfig.
  kv_interleave_pcp_size == 1).

Hole ranges come from holes/core: `common_attn_metadata.hole_segments`
(`vllm.v1.worker.hole_segments.HoleSegments`), see `request_segments`. A step
without holes uses the stock single segment [seq_len - query_len, seq_len)
per request.
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
# ParallelConfig.cp_kv_placement each scheme's KV layout requires.
REQUIRED_PLACEMENT = {
    "ulysses": "head",
    "allgather": "replicate",
    "passq": "interleave",
    "ring": "interleave",
    "adaptive": "interleave",
}


def _cp_comm() -> CPComm:
    from vllm.distributed import get_pcp_group
    try:
        return CPComm.from_coordinator(get_pcp_group())
    except AssertionError:
        return CPComm(None, 0, 1)


class CPHolesBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16, torch.float32
    ]
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
        if self.comm.world > 1:
            want = REQUIRED_PLACEMENT[self.scheme_name]
            got = getattr(pc, "cp_kv_placement", "interleave")
            assert got == want, (
                f"CP_HOLES scheme {self.scheme_name} needs "
                f"--cp-kv-placement {want}, got {got}")
        if self.scheme_name in _POSITION_SHARDED and self.comm.world > 1:
            assert self.block_size == self.interleave, (
                "CP_HOLES position-sharded schemes need block_size == "
                "cp_kv_cache_interleave_size (= u)")

    def build(self, common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> CPHolesMetadata:
        segs, bt_rows, seq_lens, missing = request_segments(
            common_attn_metadata)
        n = len(segs)
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
        bt_cpu = bt_rows.cpu()
        w, r = (d, self.comm.rank) if sharded else (1, 0)
        tables, lpos, present = [], [], []
        for i in range(n):
            e = max(seq_lens[i], plan.e_step[i])
            nb = -(-e // (self.block_size * w))
            bt, lp = vllm_local_layout(bt_cpu[i], nb, self.block_size, r, w,
                                       self.interleave if sharded else
                                       self.block_size)
            pr = kv_slot_validity(lp, e, plan.segments[i], missing[i],
                                  plan.e_step[i])
            ok = (lp >= 0) & (lp < e)
            dev = self.device
            tables.append(bt.to(dev))
            lpos.append(torch.where(ok, lp, torch.full_like(lp, -1)).to(dev))
            present.append(pr.to(dev))
        return CPHolesMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
                               plan=plan, scheme=scheme, block_tables=tables,
                               local_pos=lpos, present=present,
                               dense=self.dense)


def kv_slot_validity(local_pos: torch.Tensor, e: int,
                     segments: list[tuple[int, int]],
                     missing: list[tuple[int, int]] | None,
                     e_step: int) -> torch.Tensor:
    """Which local slots hold valid KV at the start of the step.

    A slot is valid when its position is below the request's end, is not
    computed in this step (written during the step), and is not in the
    scheduler's missing-KV ranges (holes/core's block hit-mask misses not
    computed yet; `SchedulerOutput.missing_kv_ranges`). Without that list
    (`missing is None`, e.g. a step without holes) every non-computed position
    below the end counts as resident, which is exact for stock prefix caching.
    A missing range below E_step would be read by a query of this step: that
    is a scheduling bug and raises.
    """
    bad = torch.zeros(e + 1, dtype=torch.bool)
    for s0, ln in segments:
        bad[s0:s0 + ln] = True
    for a, b in missing or ():
        if a < e_step:
            raise ValueError(
                f"CP_HOLES: KV of [{a}, {b}) is missing but the step computes "
                f"queries up to {e_step}")
        bad[min(a, e + 1):min(b, e + 1)] = True
    ok = (local_pos >= 0) & (local_pos < e)
    pr = ok.clone()
    pr[ok] = ~bad[local_pos[ok]]
    return pr


def request_segments(cam):
    """Per-request computed segments of the step, request block-table rows and
    request sequence ends.

    With holes/core, `cam.hole_segments` (a `HoleSegments`) is set when the
    step has holes, and the metadata entries are segments (query_start_loc /
    seq_lens per segment, the request's block-table row repeated). Segments
    are in token order: per request, hole pieces in position order, then the
    tail; this is the StepPlan's computed-list order, so the plan's row
    indices are the model's token indices. Without holes every entry is one
    request with the single segment [seq_len - q_len, seq_len).

    Returns (segments [[(start, length)]], block-table rows [n, max_blocks],
    seq_ends [n], missing-KV ranges per request or None each).
    """
    hs = getattr(cam, "hole_segments", None)
    if hs is None:
        n = cam.num_reqs
        seq = cam.seq_lens_cpu[:n].tolist()
        qsl = cam.query_start_loc_cpu[:n + 1].tolist()
        segs = [[(seq[i] - (qsl[i + 1] - qsl[i]), qsl[i + 1] - qsl[i])]
                for i in range(n)]
        return segs, cam.block_table_tensor[:n], seq, [None] * n
    req = hs.seg_req_idx.tolist()
    q = hs.seg_query_lens.tolist()
    ends = hs.seg_ends.tolist()
    segs, first, seq = [], [], []
    for k, r in enumerate(req):
        if not first or req[first[-1]] != r:
            assert not first or r > req[first[-1]], \
                "hole segments must be grouped by request"
            first.append(k)
            segs.append([])
            seq.append(0)
        segs[-1].append((ends[k] - q[k], q[k]))
        seq[-1] = max(seq[-1], ends[k])
    rows = torch.tensor(first, dtype=torch.int64,
                        device=cam.block_table_tensor.device)
    rm = getattr(hs, "req_missing_ranges", None)
    missing = [None if rm is None else list(rm[req[k]]) for k in first]
    return segs, cam.block_table_tensor.index_select(0, rows), seq, missing


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
        scheme = md.scheme
        # sharded: the model runner fed this rank only its own rows
        n = int(scheme.own_rows.numel()) if md.dense == "sharded" else \
            md.num_actual_tokens
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


def sharded_scheme(attn_metadata):
    """The step's CP_HOLES scheme if the sharded dense mode is active, else
    None. `attn_metadata` is the model runner's per-layer dict."""
    if not isinstance(attn_metadata, dict):
        return None
    for md in attn_metadata.values():
        if isinstance(md, CPHolesMetadata):
            return md.scheme if md.dense == "sharded" else None
    return None


def shard_inputs(scheme, *tensors):
    """Model-runner hook (sharded dense mode): keep the rank's own rows of
    per-token inputs (input_ids, positions, inputs_embeds; None passes)."""
    out = []
    for t in tensors:
        out.append(None if t is None else t[scheme.own_rows.to(t.device)])
    return out


def gather_logits_hidden(scheme, hidden_own: torch.Tensor,
                         logits_indices: torch.Tensor) -> torch.Tensor:
    """Model-runner hook (sharded dense mode): the hidden states of the rows
    the logits are taken at (each request's tail-last token), from whichever
    rank owns them, on every rank.

    Each logits row lives on exactly one rank; the others contribute zeros to
    one all-reduce over the CP group (num_logits x hidden, small). Every rank
    then computes the logits and samples exactly as without CP, so the
    per-rank request state stays identical (as under TP).
    """
    own = scheme.own_rows.to(hidden_own.device)
    li = logits_indices.to(hidden_own.device, torch.int64)
    out = hidden_own.new_zeros((li.numel(), ) + tuple(hidden_own.shape[1:]))
    if own.numel():
        pos = torch.searchsorted(own, li).clamp(max=own.numel() - 1)
        hit = own[pos] == li
        out[hit] = hidden_own[pos[hit]]
    if scheme.comm.world > 1:
        dist.all_reduce(out, group=scheme.comm.group)
    return out


def register() -> None:
    """Register CP_HOLES as AttentionBackendEnum.CUSTOM."""
    from vllm.v1.attention.backends.registry import (AttentionBackendEnum,
                                                     register_backend)
    register_backend(AttentionBackendEnum.CUSTOM,
                     "vllm.v1.attention.backends.cp_holes.CPHolesBackend")
