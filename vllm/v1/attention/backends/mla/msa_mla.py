# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA_MLA: hole-aware multi-segment prefill for MLA models (Triton).

TRITON_MLA with the prefill (`forward_mha`) replaced by MSA (MLA-MSA.md):

* With holes (branch holes/core) every varlen entry of the common metadata is a
  segment (hole piece or tail; seq_len = segment end) and `hole_segments` maps it
  back to its request. Stock MLA then gathers and up-projects the context
  [0, s_i) once per *segment*. MSA_MLA groups the prefill segments by request,
  gathers the latent of [0, E) (E = the request's deepest segment end; the step's
  own KV is already in the cache) once, up-projects it once through kv_b_proj,
  and runs one MSA launch over all segments: MHA with qk 128 + 64 and v 128,
  mask kpos <= qpos (the "compute" path).
* A request-step whose segments are all tiny (sum of query tokens <=
  MSA_MLA_ABSORB_MAX) instead runs the "absorbed" path: MQA over the paged latent
  (qk 512 + 64, v = latent), no gather and no up-projection.
* Decode tokens use TRITON_MLA's absorbed Triton kernel unchanged.
* Without holes every request is one segment [seq_len - q, seq_len), so MSA_MLA
  is a drop-in MLA backend.

Select with `--attention-backend MSA_MLA`. Env knobs: MSA_MODE (a | b | c | bc |
auto, as for MSA), MSA_EPS, MSA_C, MSA_MLA_PATH (auto | compute | absorb),
MSA_MLA_ABSORB_MAX (default 128 tokens per request-step, from the RTX 4050 bench), MSA_MLA_DECODE (triton |
msa), MSA_MLA_WORKSPACE (context tokens per pass; tests), MSA_MLA_GROUP_ALIASED=1
(profiler only: entries whose block tables start with the same block are the
segments of one request, as in the profiler's aliased segmented shots).
The up-projected context is bounded by vLLM's MLA prefill workspace
(`chunked_prefill_workspace_size`); a longer context runs in several passes
merged by LSE. Not supported: fp8 KV cache, DCP.
"""

import os
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonPrefillMetadata,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_and_maybe_dequant_weights,
)
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend, TritonMLAImpl
from vllm.v1.attention.ops.msa import merge_states
from vllm.v1.attention.ops.msa.api_mla import (
    ABSORB,
    COMPUTE,
    context_index,
    msa_mla_attention,
    plan_msa_mla,
)
from vllm.v1.attention.ops.msa.planner import MsaPlan


@dataclass
class MsaMLAPass:
    """One context pass of the compute path: gather index, dense block table,
    key positions (None = from 0) and the MSA plan."""

    src_slots: torch.Tensor
    dense_bt: torch.Tensor
    kblk_pos: torch.Tensor | None
    plan: MsaPlan


@dataclass
class MsaMLAPrefillMetadata(MLACommonPrefillMetadata):
    passes: list[MsaMLAPass] = field(default_factory=list)
    absorb_rows: torch.Tensor | None = None  # prefill-local rows of absorbed groups
    absorb_plan: MsaPlan | None = None
    absorb_bt: torch.Tensor | None = None
    stats: dict = field(default_factory=dict)


def mla_prefill_groups(cm, num_decodes: int, group_aliased: bool = False):
    """Prefill segments grouped by request.

    Returns (groups, num_prefill_rows): groups is a list of
    (varlen entry whose block-table row covers the group's context, E,
    [(row, n, pos_start)]), rows relative to the first prefill token.
    Requests come from `hole_segments` (holes/core), else every entry is its
    own request; with `group_aliased` entries whose block tables start with the
    same block are one request (the profiler's aliased segmented shots).
    """
    n = cm.num_reqs
    qsl = cm.query_start_loc_cpu[: n + 1].numpy().astype(np.int64)
    if cm._seq_lens_cpu is not None:
        seq_lens = cm._seq_lens_cpu[:n].numpy().astype(np.int64)
    else:  # H<>D sync; the GPU model runner always passes the CPU copy
        seq_lens = cm.seq_lens[:n].cpu().numpy().astype(np.int64)
    hs = getattr(cm, "hole_segments", None)
    if hs is not None:
        req = hs.seg_req_idx.astype(np.int64)
    elif group_aliased:  # H<>D sync, profiling only
        req = cm.block_table_tensor[:n, 0].cpu().numpy().astype(np.int64)
    else:
        req = np.arange(n)
    base = qsl[num_decodes]
    order: dict[int, list] = {}
    for i in range(num_decodes, n):
        q = int(qsl[i + 1] - qsl[i])
        if q <= 0:
            continue
        g = order.setdefault(int(req[i]), [i, 0, []])
        if int(seq_lens[i]) > g[1]:
            g[0], g[1] = i, int(seq_lens[i])
        g[2].append((int(qsl[i] - base), q, int(seq_lens[i]) - q))
    return [tuple(v) for v in order.values()], int(qsl[n] - base)


class MsaMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kw):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kw)
        self.mode = os.environ.get("MSA_MODE", "auto")
        self.eps = float(os.environ.get("MSA_EPS", "0.05"))
        self.c = float(os.environ.get("MSA_C", "2.0"))
        self.path = os.environ.get("MSA_MLA_PATH", "auto")
        assert self.path in ("auto", COMPUTE, ABSORB), self.path
        self.absorb_max = int(os.environ.get("MSA_MLA_ABSORB_MAX", "128"))
        self.group_aliased = os.environ.get("MSA_MLA_GROUP_ALIASED", "0") == "1"
        # decode: TRITON_MLA's kernel (default) or MSA's absorbed path ("msa";
        # forced for fp32, which the stock decode kernel cannot run on SM 8.x)
        dflt = "msa" if self.model_config.dtype == torch.float32 else "triton"
        self.decode_impl = os.environ.get("MSA_MLA_DECODE", dflt)
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.block_size = kv_cache_spec.block_size
        self.elem_size = torch.empty((), dtype=self.model_config.dtype).element_size()

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cm = common_attn_metadata
        if md.decode is not None and self.decode_impl == "msa":
            nd = md.num_decodes
            qsl = cm.query_start_loc_cpu[: nd + 1].numpy().astype(np.int64)
            if cm._seq_lens_cpu is not None:
                sl = cm._seq_lens_cpu[:nd].numpy().astype(np.int64)
            else:
                sl = cm.seq_lens[:nd].cpu().numpy().astype(np.int64)
            ql = qsl[1:] - qsl[:-1]
            segs = np.stack([np.arange(nd), qsl[:-1], ql, sl - ql], axis=1)
            md.decode.msa_plan = plan_msa_mla(
                segs, sl, ABSORB, self.num_heads, block_size=self.block_size,
                mode=self.mode, num_sms=self.num_sms, device=self.device, eps=self.eps,
                c=self.c, num_rows=int(qsl[-1]), elem_size=self.elem_size)
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
        dev = self.device
        H = self.num_heads
        kw = dict(mode=self.mode, num_sms=self.num_sms, device=dev, eps=self.eps, c=self.c,
                  elem_size=self.elem_size)
        if comp:
            G = len(comp)
            Es = np.array([g[1] for g in comp], dtype=np.int64)
            rows_t = torch.tensor([g[0] for g in comp], dtype=torch.int64)
            bt_rows = bt.index_select(0, rows_t.to(bt.device, non_blocking=True))
            W = int(os.environ.get("MSA_MLA_WORKSPACE", self.chunked_prefill_workspace_size))
            wg = max(16, (W // G) // 16 * 16)
            n_pass = int(-(-Es.max() // wg)) if Es.sum() > W else 1
            if n_pass == 1:
                wg = int(Es.max())
            qpos_dev = None
            for p in range(n_pass):
                starts = np.minimum(Es, p * wg)
                lens = np.clip(Es - p * wg, 0, wg)
                segs = [(gi, r, n, s) for gi, g in enumerate(comp) for r, n, s in g[2]
                        if lens[gi] > 0]
                src, dbt, kv_lens, kbp = context_index(bt_rows, starts, lens, self.block_size)
                plan = plan_msa_mla(segs, kv_lens, COMPUTE, H, block_size=16, kblk_pos=kbp,
                                    num_rows=n_rows, qpos=qpos_dev, **kw)
                qpos_dev = plan.qpos
                kbp_t = None
                if kbp is not None:
                    kbp_t = torch.from_numpy(kbp.astype(np.int32)).to(dev, non_blocking=True)
                meta.passes.append(MsaMLAPass(src, dbt, kbp_t, plan))
            meta.stats.update(compute_groups=G, passes=n_pass, ctx_tokens=int(Es.sum()))
        if absb:
            rows, segs, k = [], [], 0
            for gi, g in enumerate(absb):
                for r, n, s in g[2]:
                    rows.append(np.arange(r, r + n))
                    segs.append((gi, k, n, s))
                    k += n
            kv_lens = np.array([g[1] for g in absb], dtype=np.int64)
            meta.absorb_rows = torch.from_numpy(np.concatenate(rows)).to(dev, non_blocking=True)
            rows_t = torch.tensor([g[0] for g in absb], dtype=torch.int64)
            meta.absorb_bt = bt.index_select(0, rows_t.to(bt.device, non_blocking=True))
            meta.absorb_plan = plan_msa_mla(segs, kv_lens, ABSORB, H,
                                            block_size=self.block_size, num_rows=k, **kw)
            meta.stats.update(absorb_groups=len(absb), absorb_rows=k)
        md.prefill = meta
        return md


class MsaMLAImpl(TritonMLAImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("MSA_MLA: fp8 KV cache is not supported")
        self._w_uk_t = None
        self._w_uv = None

    def _absorb_weights(self, dtype):
        if self._w_uk_t is None:
            w = get_and_maybe_dequant_weights(self.kv_b_proj, out_dtype=dtype).T
            w = w.view(self.kv_lora_rank, self.num_heads,
                       self.qk_nope_head_dim + self.v_head_dim)
            w_uk, w_uv = w.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            self._w_uk_t = w_uk.permute(1, 2, 0).contiguous()  # [H, P, L]
            self._w_uv = w_uv.transpose(0, 1).contiguous()  # [H, L, V]
        return self._w_uk_t, self._w_uv

    def forward_mha(self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata,
                    k_scale, output) -> None:
        pf = attn_metadata.prefill
        assert isinstance(pf, MsaMLAPrefillMetadata)
        H, P, V, L = self.num_heads, self.qk_nope_head_dim, self.v_head_dim, self.kv_lora_rank
        out = output.view(-1, H, V)
        q_nope, q_pe = q[..., :P], q[..., P:]
        flat = kv_c_and_k_pe_cache.view(-1, kv_c_and_k_pe_cache.shape[-1])
        acc = None
        for ps in pf.passes:
            ws = flat.index_select(0, ps.src_slots)
            kvn = self.kv_b_proj(ws[:, :L])[0].view(-1, H, P + V)
            nb = ws.shape[0] // 16
            kvn = kvn.view(nb, 16, H, P + V)
            k_nope, v = kvn[..., :P], kvn[..., P:]
            kpe = ws.view(nb, 16, 1, -1)[..., L:].expand(-1, -1, H, -1)
            if len(pf.passes) == 1:
                msa_mla_attention(q_nope, q_pe, k_nope, kpe, v, ps.dense_bt, ps.plan,
                                  kblk_pos=ps.kblk_pos, softmax_scale=self.scale, out=out)
                continue
            o_p = torch.zeros(out.shape, device=q.device, dtype=torch.float32)
            l_p = torch.full(out.shape[:2], float("-inf"), device=q.device)
            msa_mla_attention(q_nope, q_pe, k_nope, kpe, v, ps.dense_bt, ps.plan,
                              kblk_pos=ps.kblk_pos, softmax_scale=self.scale,
                              out=o_p, lse=l_p)
            acc = (o_p, l_p) if acc is None else merge_states(acc[0], acc[1], o_p, l_p)
        if acc is not None:
            rows = pf.passes[0].plan.qpos >= 0
            out[rows] = acc[0][rows].to(out.dtype)
        if pf.absorb_plan is not None:
            w_uk_t, w_uv = self._absorb_weights(q.dtype)
            idx = pf.absorb_rows
            qn = q_nope.index_select(0, idx).transpose(0, 1)  # [H, r, P]
            q_lat = torch.bmm(qn, w_uk_t).transpose(0, 1).contiguous()  # [r, H, L]
            qpe = q_pe.index_select(0, idx)
            cache4 = kv_c_and_k_pe_cache.unsqueeze(2)
            o_lat = msa_mla_attention(q_lat, qpe, cache4[..., :L], cache4[..., L:], None,
                                      pf.absorb_bt, pf.absorb_plan,
                                      softmax_scale=self.scale)
            o_v = torch.bmm(o_lat.transpose(0, 1), w_uv).transpose(0, 1)  # [r, H, V]
            out.index_copy_(0, idx, o_v.to(out.dtype))


    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        plan = getattr(attn_metadata.decode, "msa_plan", None)
        if plan is None:
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)
        if type(q) is tuple:
            q_lat, q_pe = q
        else:
            q_lat, q_pe = q[..., : self.kv_lora_rank], q[..., self.kv_lora_rank :]
        L = self.kv_lora_rank
        cache4 = kv_c_and_k_pe_cache.unsqueeze(2)
        o, lse = msa_mla_attention(q_lat, q_pe, cache4[..., :L], cache4[..., L:], None,
                                   attn_metadata.decode.block_table, plan,
                                   softmax_scale=self.scale, return_lse=True)
        return o, lse


class MsaMLABackend(TritonMLABackend):
    # fp32 is for strict end-to-end tests (decode runs TRITON_MLA's kernel in fp32)
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16, torch.bfloat16, torch.float32
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "MSA_MLA"

    @staticmethod
    def get_impl_cls() -> type["MsaMLAImpl"]:
        return MsaMLAImpl

    @staticmethod
    def get_builder_cls() -> type["MsaMLAMetadataBuilder"]:
        return MsaMLAMetadataBuilder
