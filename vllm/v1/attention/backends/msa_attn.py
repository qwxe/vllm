# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MSA: hole-aware, position-explicit paged attention backend (Triton).

Derived from the TRITON_ATTN backend: same KV cache layout and KV write
(`triton_reshape_and_cache_flash` on `slot_mapping`); only the attention call
changes. One work-list kernel (`vllm/v1/attention/ops/msa`) runs three host
schedules (KERNEL-ALGO.md 2.2-2.5): (a) one tile chain per segment, (b)
position-packed q tiles, (c) split-KV with an LSE merge. The planner picks
per step (MSA_MODE=auto): pack only >= 2 sub-tile pieces of one request within
W_pack = max(BN, eps * depth); split key ranges when the launch would leave
the GPU under-filled (chain cap = total keys / (c * SMs)).

Segments: with holes (branch holes/core) the common metadata has one varlen
entry per segment (seq_lens = segment end, the request's block-table row
repeated) and `hole_segments` maps segments back to requests; MSA uses it to
put a request's segments on one block-table row, so they can share packed
tiles. Without holes every request is one segment (stock semantics), so MSA is
a drop-in backend for ordinary prefill / decode / mixed batches.

Select with `--attention-backend MSA`. Knobs (env): MSA_MODE in
{a, b, c, bc, auto} (default auto), MSA_EPS (0.05), MSA_C (2.0).
Not supported: non-causal attention, R-SWA, async speculative decode,
alibi, sliding window, softcap, sinks, fp8 KV, encoder
attention, CUDA graphs for attention (the work list changes per step;
piecewise graphs still cover the rest of the model).
"""

import os
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.ops.msa import MsaPlan, msa_attention, plan_msa


@dataclass
class MsaAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    plan: MsaPlan
    use_cascade: bool = False


def msa_segments(cm: CommonAttentionMetadata) -> tuple[np.ndarray, np.ndarray]:
    """CPU (segments [n, 4] = (block-table row, row_start, n_rows, pos_start),
    kv_lens [rows of the block table]) for one step."""
    hs = getattr(cm, "hole_segments", None)
    if hs is not None:
        cu = hs.cu_query_lens().astype(np.int64)
        q = hs.seg_query_lens.astype(np.int64)
        ends = hs.seg_ends.astype(np.int64)
        req = hs.seg_req_idx.astype(np.int64)
        S = len(q)
        # first segment (= block-table row) of every request; rows of one
        # request are identical copies of its table
        first = np.full(int(req.max()) + 1 if S else 0, -1, dtype=np.int64)
        for i in range(S - 1, -1, -1):
            first[req[i]] = i
        row = first[req]
        kv_lens = np.zeros(S, dtype=np.int64)
        np.maximum.at(kv_lens, row, ends)
        seg = np.stack([row, cu[:-1], q, ends - q], axis=1)
        return seg[q > 0], kv_lens
    n = cm.num_reqs
    seq_lens = cm.seq_lens[:n].cpu().numpy().astype(np.int64)
    qsl = cm.query_start_loc_cpu[: n + 1].numpy().astype(np.int64)
    qlen = qsl[1:] - qsl[:-1]
    seg = np.stack([np.arange(n), qsl[:-1], qlen, seq_lens - qlen], axis=1)
    return seg[qlen > 0], seq_lens


class MsaAttentionMetadataBuilder(AttentionMetadataBuilder[MsaAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if (
            vllm_config.scheduler_config.async_scheduling
            and vllm_config.speculative_config is not None
            and vllm_config.speculative_config.num_speculative_tokens > 0
        ):
            raise NotImplementedError("MSA: async speculative decode is not supported")
        self.block_size = kv_cache_spec.block_size
        mc = vllm_config.model_config
        self.num_heads_q = mc.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = mc.get_num_kv_heads(vllm_config.parallel_config)
        self.mode = os.environ.get("MSA_MODE", "auto")
        self.eps = float(os.environ.get("MSA_EPS", "0.05"))
        self.c = float(os.environ.get("MSA_C", "2.0"))
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.elem_size = torch.empty((), dtype=mc.dtype).element_size()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MsaAttentionMetadata:
        cm = common_attn_metadata
        causal = cm.causal
        if not (bool(causal.all()) if isinstance(causal, torch.Tensor) else causal):
            raise NotImplementedError("MSA: non-causal attention is not supported")
        if cm.rswa_prefix_lens is not None:
            raise NotImplementedError("MSA: R-SWA is not supported")
        segs, kv_lens = msa_segments(cm)
        plan = plan_msa(
            segs,
            kv_lens,
            self.num_heads_kv,
            self.num_heads_q // self.num_heads_kv,
            block_size=self.block_size,
            mode=self.mode,
            num_sms=self.num_sms,
            device=self.device,
            eps=self.eps,
            c=self.c,
            num_rows=cm.num_actual_tokens,
            elem_size=self.elem_size,
        )
        return MsaAttentionMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
            plan=plan,
        )


class MsaAttentionImpl(TritonAttentionImpl):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            self.alibi_slopes is not None
            or self.sliding_window != (-1, -1)
            or self.logits_soft_cap
            or self.sinks is not None
        ):
            raise NotImplementedError(
                "MSA: alibi / sliding window / softcap / sinks are not supported"
            )
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("MSA: fp8 KV cache is not supported")
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError("MSA: decoder self-attention only")

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata: MsaAttentionMetadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        assert output is not None, "Output tensor must be provided."
        if attn_metadata is None:  # profiling run
            return output.fill_(0)
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("MSA: fused output quantization")
        n = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        msa_attention(
            query[:n],
            key_cache,
            value_cache,
            attn_metadata.block_table,
            attn_metadata.plan,
            softmax_scale=self.scale,
            out=output[:n],
        )
        return output


class MsaAttentionBackend(TritonAttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "MSA"

    @staticmethod
    def get_impl_cls() -> type["MsaAttentionImpl"]:
        return MsaAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MsaAttentionMetadataBuilder"]:
        return MsaAttentionMetadataBuilder

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_rswa(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER
