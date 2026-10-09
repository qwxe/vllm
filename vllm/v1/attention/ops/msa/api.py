# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Synced from the standalone repo /home/asd/vllm-dev-kernels (msa/).
"""Public API: msa_attention (plan + launch) and merge_states (for CP callers)."""

import math

import torch
import triton

from .kernel import _merge_states, _msa_fwd, _msa_merge
from .planner import MsaPlan, plan_msa


def msa_attention(
    q: torch.Tensor,             # [rows, Hq, D]
    k_cache: torch.Tensor,       # [num_blocks, block_size, Hkv, D]
    v_cache: torch.Tensor,       # [num_blocks, block_size, Hkv, D]
    block_table: torch.Tensor,   # [num_reqs, max_blocks] int32 (local blocks)
    plan: MsaPlan,
    kblk_pos: torch.Tensor | None = None,   # int32; [num_reqs, max_blocks] block starts (plan.kpos_mode 1)
                                            # or [num_reqs, max_keys] per-key positions (kpos_mode 2)
    softmax_scale: float | None = None,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,        # [rows, Hq] fp32, natural log
    return_lse: bool = False,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """Position-explicit paged attention. Returns O (and LSE if asked).

    Rows not covered by the plan's segments are left untouched in `out`. A row whose
    visible key set is empty gets O = 0 and LSE = -inf (merge-neutral).
    """
    rows, Hq, D = q.shape
    Hkv = k_cache.shape[2]
    G = plan.num_q_per_kv
    assert Hq == Hkv * G, (Hq, Hkv, G)
    assert k_cache.shape[1] == plan.block_size
    assert k_cache.stride(3) == 1 and v_cache.stride(3) == 1 and q.stride(2) == 1
    if plan.kpos_mode:
        assert kblk_pos is not None and kblk_pos.dtype == torch.int32
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)
    if out is None:
        out = torch.empty_like(q)
    write_lse = return_lse or lse is not None
    if write_lse and lse is None:
        lse = torch.empty(rows, Hq, device=q.device, dtype=torch.float32)
    D_PAD = triton.next_power_of_2(D)
    M = plan.m_rows
    if plan.n_slots:
        op = torch.empty(plan.n_slots, M, D_PAD, device=q.device, dtype=torch.float32)
        lsep = torch.empty(plan.n_slots, M, device=q.device, dtype=torch.float32)
    else:
        op = lsep = q  # unused dummies
    lse_arg = lse if write_lse else q
    kp = kblk_pos if kblk_pos is not None else block_table
    num_warps = num_warps or plan.num_warps
    num_stages = num_stages or plan.num_stages
    n_items = plan.items.shape[0]
    if n_items:
        _msa_fwd[(n_items,)](
            q, k_cache, v_cache, out, lse_arg, op, lsep,
            plan.qpos, plan.tile_rows, plan.items, block_table, kp,
            softmax_scale * 1.4426950408889634,
            q.stride(0), q.stride(1), out.stride(0), out.stride(1),
            lse.stride(0) if write_lse else 0, lse.stride(1) if write_lse else 0,
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_table.stride(0), kp.stride(0),
            G=G, G_PAD=plan.g_pad, BQ=plan.bq, D=D, D_PAD=D_PAD, BN=plan.bn,
            BLOCK_SIZE=plan.block_size, KPOS_MODE=plan.kpos_mode, WRITE_LSE=write_lse,
            num_warps=num_warps, num_stages=num_stages,
        )
    n_groups = plan.groups.shape[0]
    if n_groups:
        _msa_merge[(n_groups,)](
            out, lse_arg, op, lsep, plan.tile_rows, plan.groups,
            out.stride(0), out.stride(1),
            lse.stride(0) if write_lse else 0, lse.stride(1) if write_lse else 0,
            G=G, G_PAD=plan.g_pad, BQ=plan.bq, D=D, D_PAD=D_PAD, WRITE_LSE=write_lse,
            num_warps=4,
        )
    if write_lse:
        return out, lse
    return out


def merge_states(o_a, lse_a, o_b, lse_b, out=None, lse_out=None):
    """Merge two partial attention results over disjoint key sets.

    o_*: [rows, H, D] (any float dtype), lse_*: [rows, H] fp32 natural log (-inf = empty).
    Returns (O, LSE). This is the ring-step / pass-Q reduction (same math as vLLM's
    merge_attn_states / FlashInfer merge_state).
    """
    rows, H, D = o_a.shape
    assert lse_a.is_contiguous() and lse_b.is_contiguous()
    if out is None:
        out = torch.empty_like(o_a)
    if lse_out is None:
        lse_out = torch.empty_like(lse_a)
    BR = 16
    _merge_states[(triton.cdiv(rows, BR), H)](
        out, lse_out, o_a, lse_a, o_b, lse_b, rows, H,
        out.stride(0), out.stride(1), o_a.stride(0), o_a.stride(1), o_b.stride(0), o_b.stride(1),
        lse_out.stride(0), lse_a.stride(0), lse_b.stride(0),
        D=D, D_PAD=triton.next_power_of_2(D), BR=BR, num_warps=4,
    )
    return out, lse_out


def msa(q, k_cache, v_cache, block_table, segments, kv_lens, *, kblk_pos=None,
        kblk_pos_cpu=None, mode="auto", softmax_scale=None, return_lse=False, **plan_kw):
    """One-shot convenience: plan + launch. CP callers that reuse a plan across layers
    should call plan_msa once per step and msa_attention per layer."""
    Hkv = k_cache.shape[2]
    G = q.shape[1] // Hkv
    if kblk_pos is not None and kblk_pos_cpu is None:
        kblk_pos_cpu = kblk_pos.cpu().numpy()
    plan = plan_msa(segments, kv_lens, Hkv, G, block_size=k_cache.shape[1], elem_size=q.element_size(),
                    kblk_pos=kblk_pos_cpu, mode=mode, device=q.device,
                    num_rows=q.shape[0], **plan_kw)
    return msa_attention(q, k_cache, v_cache, block_table, plan, kblk_pos=kblk_pos,
                         softmax_scale=softmax_scale, return_lse=return_lse)
