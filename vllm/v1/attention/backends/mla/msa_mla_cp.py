# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA_MLA_CP: MSA_MLA with prefill context parallelism (PCP) for MLA models.

vLLM 0.19 creates the PCP group (`--prefill-context-parallel-size d`), the
interleaved KV placement (KV manager blocks of block_size * d tokens, slot
mapping `-1` for positions another rank holds) and fused-MoE support for it,
but no MLA backend sets `supports_pcp`. This backend runs the pass-Q scheme of
`vllm.v1.attention.cp_holes.mla` (MLA-MSA.md section 4) inside vLLM's MLA
layer, in the replicated-dense mode: the model runner gives every PCP rank the
whole step (all rows), so

* Q needs no all-to-all (every rank has every row);
* the new latent reaches its home rank through vLLM's own cache write (the
  PCP slot mapping stores only the positions the rank holds), so nothing
  travels for it either;
* each rank attends all prefill rows of the step over the keys it holds:
  the latent of its local keys of [0, E) per request is gathered (local key j
  of a request is position ((j // u) * d + r) * u + j % u; pages of 16 keys
  are contiguous in position when u % 16 == 0, so the MSA kernel reads them
  with KPOS_MODE 1 page starts), up-projected through kv_b_proj (E / d keys
  per rank) and attended with the compute path; small request-steps take the
  absorbed path over the same keys in place;
* the (O, LSE) partials go to the row owners (all-to-all), are merged by LSE
  and all-gathered, so every rank continues with the full output
  (`cp_holes.mla.reduce_replicated`).

Decode tokens attend the rank's local keys with the absorbed MQA kernel (local
sequence length = keys the rank holds below the end, as vLLM's DCP does) and
the partials are all-gathered and merged over PCP (`allgather_merge`). The
default decode kernel is MSA's absorbed path (fp32 LSE, empty partials give
LSE = -inf); `MSA_MLA_DECODE=triton` selects TRITON_MLA's kernel, whose LSE is
in the model dtype.

Selection: `--attention-backend MSA_MLA_CP --prefill-context-parallel-size d
--cp-kv-placement interleave --cp-kv-cache-interleave-size u --block-size B`
with u % 16 == 0 and B % u == 0 (u = B = 128 is the CP_HOLES default). The
MSA_MLA env knobs apply (MSA_MODE, MSA_MLA_PATH, MSA_MLA_ABSORB_MAX, ...).
At d = 1 it computes what MSA_MLA computes, through the CP code path.
Not supported: the sharded dense mode (VLLM_CP_HOLES_DENSE=sharded; DeepSeek's
fused MoE all-gathers equal-size PCP shards), spec decode under d > 1, fp8 KV
cache, DCP, CUDA graphs (the decode merge is a collective).
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadataBuilder
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.mla.msa_mla import (
    MsaMLABackend,
    MsaMLAImpl,
    MsaMLAMetadataBuilder,
    MsaMLAPass,
    MsaMLAPrefillMetadata,
    mla_prefill_groups,
)
from vllm.v1.attention.cp_holes.mla import (
    allgather_merge,
    cp_local_len,
    cp_local_pos,
    reduce_replicated,
)
from vllm.v1.attention.cp_holes.schemes import CPComm
from vllm.v1.attention.ops.msa import merge_states
from vllm.v1.attention.ops.msa.api_mla import (
    ABSORB,
    COMPUTE,
    context_index,
    msa_mla_attention,
    plan_msa_mla,
)

logger = init_logger(__name__)

PAGE = 16
_FAR = 1 << 30  # key position of padding pages (never visible)


def _cp_comm() -> CPComm:
    from vllm.distributed import get_pcp_group

    try:
        return CPComm.from_coordinator(get_pcp_group())
    except AssertionError:
        return CPComm(None, 0, 1)


@dataclass
class MlaCPInfo:
    """Per-step CP data shared by every layer."""

    comm: CPComm
    cnt: list[int]  # prefill rows owned by each rank (contiguous)


def page_starts(starts, lens, d: int, r: int, u: int) -> np.ndarray:
    """[G, max pages] absolute position of the first key of every 16-key page
    of local keys [starts[g], starts[g] + lens[g]) of rank r (padding pages:
    _FAR). starts must be multiples of 16 and u a multiple of 16."""
    starts = np.asarray(starts, dtype=np.int64)
    lens = np.asarray(lens, dtype=np.int64)
    nblk = -(-lens // PAGE)
    mx = max(1, int(nblk.max()) if len(nblk) else 1)
    kbp = np.full((len(lens), mx), _FAR, dtype=np.int64)
    for g, nb in enumerate(nblk):
        kbp[g, :nb] = cp_local_pos(starts[g] + PAGE * np.arange(nb), d, r, u)
    return kbp


def page_table(bt_rows: torch.Tensor, n_pages: int, block_size: int) -> torch.Tensor:
    """[G, n_pages] int32: the paged cache viewed as 16-key pages; page k of a
    request's local keys lives in cache block bt_rows[g, 16 k // block_size]."""
    ppb = block_size // PAGE
    k = torch.arange(max(1, n_pages), device=bt_rows.device)
    return (bt_rows[:, k // ppb].to(torch.int64) * ppb + k % ppb).to(torch.int32)


class MsaMLACPMetadataBuilder(MsaMLAMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kw):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kw)
        pc = vllm_config.parallel_config
        self.comm = _cp_comm()
        self.d, self.r = self.comm.world, self.comm.rank
        self.u = max(1, int(pc.cp_kv_cache_interleave_size))
        # decode: MSA's absorbed kernel by default (fp32 LSE for the merge)
        self.decode_impl = os.environ.get("MSA_MLA_DECODE", "msa")
        dense = os.environ.get("VLLM_CP_HOLES_DENSE", "replicated")
        if dense != "replicated":
            raise NotImplementedError(
                "MSA_MLA_CP runs the replicated dense mode only "
                f"(VLLM_CP_HOLES_DENSE={dense})")
        if self.d > 1:
            placement = getattr(pc, "cp_kv_placement", "interleave")
            assert placement == "interleave", (
                f"MSA_MLA_CP needs --cp-kv-placement interleave, got {placement}")
            assert self.u % PAGE == 0, (
                f"MSA_MLA_CP needs cp_kv_cache_interleave_size % {PAGE} == 0, got {self.u}")
            assert self.block_size % self.u == 0 and self.block_size % PAGE == 0, (
                self.block_size, self.u)
            assert pc.decode_context_parallel_size == 1, "MSA_MLA_CP: no DCP"
        logger.info_once(
            "MSA_MLA_CP: pcp rank %d of %d, interleave %d, block %d, decode %s",
            self.r, self.d, self.u, self.block_size, self.decode_impl)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = MLACommonMetadataBuilder.build(
            self, common_prefix_len, common_attn_metadata, fast_build)
        cm = common_attn_metadata
        d, r, u = self.d, self.r, self.u
        dev = self.device
        H = self.num_heads
        kw = dict(mode=self.mode, num_sms=self.num_sms, device=dev, eps=self.eps, c=self.c,
                  elem_size=self.elem_size)
        if md.decode is not None:
            nd = md.num_decodes
            qsl = cm.query_start_loc_cpu[: nd + 1].numpy().astype(np.int64)
            if cm._seq_lens_cpu is not None:
                sl = cm._seq_lens_cpu[:nd].numpy().astype(np.int64)
            else:
                sl = cm.seq_lens[:nd].cpu().numpy().astype(np.int64)
            ql = qsl[1:] - qsl[:-1]
            loc = cp_local_len(sl, d, r, u)
            if d > 1:
                assert (ql == 1).all(), "MSA_MLA_CP: one decode token per request"
                md.decode.seq_lens = torch.from_numpy(loc.astype(np.int32)).to(
                    dev, non_blocking=True)
            if self.decode_impl == "msa":
                # local index order is position order; the decode token sees
                # every local key, so local indices serve as positions
                segs = np.stack([np.arange(nd), qsl[:-1], ql, np.maximum(sl - ql, loc - ql)],
                                axis=1)
                md.decode.msa_plan = plan_msa_mla(
                    segs, loc, ABSORB, H, block_size=self.block_size,
                    num_rows=int(qsl[-1]), **kw)
            md.decode.cp_comm = self.comm
        if md.prefill is None:
            return md
        groups, n_rows = mla_prefill_groups(cm, md.num_decodes, self.group_aliased)
        comp, absb = [], []
        for g in groups:
            m = sum(n for _, n, _ in g[2])
            use_abs = self.path == ABSORB or (self.path == "auto" and m <= self.absorb_max)
            (absb if use_abs else comp).append(g)
        pf = md.prefill
        meta = MsaMLAPrefillMetadata(
            block_table=pf.block_table,
            query_start_loc=pf.query_start_loc,
            max_query_len=pf.max_query_len,
            output_dtype=pf.output_dtype,
            q_data_type=pf.q_data_type,
        )
        bt = cm.block_table_tensor
        if comp:
            G = len(comp)
            Es = np.array([g[1] for g in comp], dtype=np.int64)
            Ls = cp_local_len(Es, d, r, u)
            rows_t = torch.tensor([g[0] for g in comp], dtype=torch.int64)
            bt_rows = bt.index_select(0, rows_t.to(bt.device, non_blocking=True))
            W = int(os.environ.get("MSA_MLA_WORKSPACE", self.chunked_prefill_workspace_size))
            wg = max(PAGE, (W // G) // PAGE * PAGE)
            n_pass = int(-(-Ls.max() // wg)) if Ls.sum() > W else 1
            if n_pass == 1:
                wg = max(1, int(Ls.max()))
            for p in range(n_pass):
                starts = np.minimum(Ls, p * wg)
                lens = np.clip(Ls - p * wg, 0, wg)
                segs = [(gi, row, n, s) for gi, g in enumerate(comp) for row, n, s in g[2]
                        if lens[gi] > 0]
                if not segs:
                    continue
                src, dbt, _, _ = context_index(bt_rows, starts, lens, self.block_size)
                kbp = page_starts(starts, lens, d, r, u)
                plan = plan_msa_mla(segs, lens, COMPUTE, H, block_size=PAGE, kblk_pos=kbp,
                                    num_rows=n_rows, **kw)
                kbp_t = torch.from_numpy(kbp.astype(np.int32)).to(dev, non_blocking=True)
                meta.passes.append(MsaMLAPass(src, dbt, kbp_t, plan))
            meta.stats.update(compute_groups=G, passes=len(meta.passes),
                              ctx_tokens=int(Es.sum()), local_ctx_tokens=int(Ls.sum()))
        if absb:
            rows, segs, k = [], [], 0
            for gi, g in enumerate(absb):
                for row, n, s in g[2]:
                    rows.append(np.arange(row, row + n))
                    segs.append((gi, k, n, s))
                    k += n
            Ls = cp_local_len(np.array([g[1] for g in absb], dtype=np.int64), d, r, u)
            meta.absorb_rows = torch.from_numpy(np.concatenate(rows)).to(dev, non_blocking=True)
            rows_t = torch.tensor([g[0] for g in absb], dtype=torch.int64)
            bt_rows = bt.index_select(0, rows_t.to(bt.device, non_blocking=True))
            kbp = page_starts(np.zeros_like(Ls), Ls, d, r, u)
            meta.absorb_bt = page_table(bt_rows, kbp.shape[1], self.block_size)
            meta.absorb_kbp = torch.from_numpy(kbp.astype(np.int32)).to(dev, non_blocking=True)
            meta.absorb_plan = plan_msa_mla(segs, Ls, ABSORB, H, block_size=PAGE,
                                            kblk_pos=kbp, num_rows=k, **kw)
            meta.stats.update(absorb_groups=len(absb), absorb_rows=k)
        meta.cp = MlaCPInfo(self.comm, [(x + 1) * n_rows // d - x * n_rows // d
                                        for x in range(d)])
        md.prefill = meta
        return md


class MsaMLACPImpl(MsaMLAImpl):
    supports_pcp: bool = True

    def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
                    k_scale, output) -> None:
        pf = attn_metadata.prefill
        assert isinstance(pf, MsaMLAPrefillMetadata)
        H, P, V, L = self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank
        out = output.view(-1, H, V)
        n = out.shape[0]
        q_nope, q_pe = q[..., :P], q[..., P:]
        D = kv_c_and_k_pe_cache.shape[-1]
        flat = kv_c_and_k_pe_cache.view(-1, D)
        # this rank's partial over the keys it holds, for every prefill row
        o = torch.zeros(n, H, V, device=q.device, dtype=torch.float32)
        lse = torch.full((n, H), float("-inf"), device=q.device)
        for i, ps in enumerate(pf.passes):
            ws = flat.index_select(0, ps.src_slots)
            nb = ws.shape[0] // PAGE
            kvn = self.kv_b_proj(ws[:, :L])[0].view(nb, PAGE, H, P + V)
            k_nope, v = kvn[..., :P], kvn[..., P:]
            kpe = ws.view(nb, PAGE, 1, D)[..., L:].expand(-1, -1, H, -1)
            o_p = o if i == 0 else torch.zeros_like(o)
            l_p = lse if i == 0 else torch.full_like(lse, float("-inf"))
            msa_mla_attention(q_nope, q_pe, k_nope, kpe, v, ps.dense_bt, ps.plan,
                              kblk_pos=ps.kblk_pos, softmax_scale=self.scale,
                              out=o_p, lse=l_p)
            if i:
                o, lse = merge_states(o, lse, o_p, l_p)
        if pf.absorb_plan is not None:
            w_uk_t, w_uv = self._absorb_weights(q.dtype)
            idx = pf.absorb_rows
            qn = q_nope.index_select(0, idx).transpose(0, 1)  # [H, r, P]
            q_lat = torch.bmm(qn, w_uk_t).transpose(0, 1).contiguous()  # [r, H, L]
            qpe = q_pe.index_select(0, idx)
            pages = kv_c_and_k_pe_cache.view(-1, PAGE, 1, D)
            o_lat, l_a = msa_mla_attention(q_lat, qpe, pages[..., :L], pages[..., L:], None,
                                           pf.absorb_bt, pf.absorb_plan,
                                           kblk_pos=pf.absorb_kbp,
                                           softmax_scale=self.scale, return_lse=True)
            o_v = torch.bmm(o_lat.transpose(0, 1), w_uv).transpose(0, 1)  # [r, H, V]
            o.index_copy_(0, idx, o_v.float())
            lse.index_copy_(0, idx, l_a)
        out.copy_(reduce_replicated(pf.cp.comm, pf.cp.cnt, o, lse, out.dtype))

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        o, lse = super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        comm = attn_metadata.decode.cp_comm
        if comm.world == 1:
            return o, lse
        # TRITON_MLA's kernel leaves O = nan for a request with no local key
        o = torch.where(torch.isinf(lse).unsqueeze(-1), torch.zeros_like(o), o)
        return allgather_merge(comm, o, lse)


class MsaMLACPBackend(MsaMLABackend):
    @staticmethod
    def get_name() -> str:
        return "MSA_MLA_CP"

    @staticmethod
    def get_impl_cls() -> type["MsaMLACPImpl"]:
        return MsaMLACPImpl

    @staticmethod
    def get_builder_cls() -> type["MsaMLACPMetadataBuilder"]:
        return MsaMLACPMetadataBuilder
