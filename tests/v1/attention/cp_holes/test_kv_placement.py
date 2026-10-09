# SPDX-License-Identifier: Apache-2.0
"""ParallelConfig.cp_kv_placement: KV accounting, KV spec heads, slot mapping.

CPU part: the config property, per-rank capacity, KV heads per rank.
GPU part (skipped without CUDA): vLLM's real `BlockTable.compute_slot_mapping`
Triton kernel under a faked PCP group of 2 / 4 ranks, for "interleave" vs
"head" / "replicate", checked against the layout the CP_HOLES store uses
(`vllm_slot_mapping`).

    python tests/v1/attention/cp_holes/test_kv_placement.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm.config import ParallelConfig
from vllm.v1.attention.cp_holes.kv_store import vllm_slot_mapping
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker import cp_utils


def test_config_and_capacity():
    assert ParallelConfig().cp_kv_placement == "interleave"
    for placement, want in (("interleave", 4), ("head", 1), ("replicate", 1)):
        # (a real ParallelConfig with pcp = 4 needs 4 visible GPUs)
        pc = SimpleNamespace(prefill_context_parallel_size=4,
                             cp_kv_placement=placement)
        assert ParallelConfig.kv_interleave_pcp_size.fget(pc) == want
        spec = FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=8,
                                 dtype=torch.bfloat16)
        cfg = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=4096),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                prefill_context_parallel_size=4,
                kv_interleave_pcp_size=want))
        # each rank stores max_model_len / 4 tokens only when interleaved
        assert spec.max_memory_usage_bytes(cfg) == \
            (4096 // want // 16) * spec.page_size_bytes
    # placement is part of the config hash (it changes the KV layout)
    h = {ParallelConfig(cp_kv_placement=p).compute_hash()
         for p in ("interleave", "head", "replicate")}
    assert len(h) == 3


def test_kv_heads_per_rank():
    f = cp_utils.cp_kv_heads_per_rank
    assert f(8, 1) == 8 and f(8, 2) == 4 and f(8, 8) == 1
    assert f(8, 16) == 1 and f(2, 4) == 1
    try:
        f(8, 3)
        raise AssertionError("expected failure")
    except AssertionError as e:
        assert "expected" not in str(e)


def test_kv_spec_heads():
    """Attention.get_kv_cache_spec divides the KV heads under "head"."""
    from vllm.model_executor.layers.attention.attention import Attention
    for pcp, placement, want in ((4, "head", 2), (16, "head", 1),
                                 (4, "interleave", 8), (4, "replicate", 8)):
        layer = SimpleNamespace(attn_type="decoder", sliding_window=None,
                                num_kv_heads=8, head_size=64,
                                head_size_v=64,
                                kv_cache_torch_dtype=torch.bfloat16)
        cfg = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16),
            model_config=SimpleNamespace(use_mla=False),
            parallel_config=SimpleNamespace(
                cp_kv_placement=placement,
                prefill_context_parallel_size=pcp))
        spec = Attention.get_kv_cache_spec(layer, cfg)
        assert spec.num_kv_heads == want, (pcp, placement, spec.num_kv_heads)


def _gpu_slot_mapping(world, rank, placement, bs, positions, row):
    """Run vLLM's BlockTable slot mapping with a faked PCP group."""
    from vllm.v1.worker import block_table as bt_mod
    fake = SimpleNamespace(world_size=world, rank_in_group=rank)
    orig = cp_utils.get_pcp_group
    cp_utils.get_pcp_group = lambda: fake
    cp_utils.set_cp_kv_placement(placement)
    try:
        bt = bt_mod.BlockTable(bs, 4, 64, 4096, False, torch.device("cuda"),
                               bs, bs)
    finally:
        cp_utils.get_pcp_group = orig
        cp_utils.set_cp_kv_placement("interleave")
    bt.add_row(row.tolist(), 0)
    bt.commit_block_table(1)
    n = positions.numel()
    qsl = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    bt.compute_slot_mapping(1, qsl, positions.cuda())
    return bt.slot_mapping.gpu[:n].cpu()


def test_slot_mapping_gpu():
    if not torch.cuda.is_available():
        print("skip: no CUDA")
        return
    bs = 16
    g = torch.Generator().manual_seed(0)
    row = torch.randperm(60, generator=g)[:40].to(torch.int32)
    pos = torch.randperm(600, generator=g)[:300].to(torch.int64)
    for world in (2, 4):
        for rank in range(world):
            got = _gpu_slot_mapping(world, rank, "interleave", bs, pos, row)
            want = vllm_slot_mapping(row, pos, bs, rank, world, bs)
            assert torch.equal(got, want), ("interleave", world, rank)
            for placement in ("head", "replicate"):
                got = _gpu_slot_mapping(world, rank, placement, bs, pos, row)
                want = vllm_slot_mapping(row, pos, bs, 0, 1, bs)
                assert torch.equal(got, want), (placement, world, rank)
                assert bool((got >= 0).all())


if __name__ == "__main__":
    test_config_and_capacity()
    test_kv_heads_per_rank()
    test_kv_spec_heads()
    test_slot_mapping_gpu()
    print("kv placement tests OK")
