# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any, cast

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group, get_pcp_group

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


def check_attention_cp_compatibility(vllm_config: VllmConfig) -> None:
    pcp_size = vllm_config.parallel_config.prefill_context_parallel_size
    dcp_size = vllm_config.parallel_config.decode_context_parallel_size
    interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
    if pcp_size * dcp_size > 1:
        layer_type = cast(type[Any], AttentionLayerBase)
        layers = get_layers_from_vllm_config(vllm_config, layer_type)
        for layer in layers.values():
            layer_impl = getattr(layer, "impl", None)
            if layer_impl is None:
                continue
            if vllm_config.speculative_config is not None and interleave_size > 1:
                assert layer_impl.supports_mtp_with_cp_non_trivial_interleave_size, (
                    "MTP with cp_kv_cache_interleave_size > 1 is not "
                    f"supported in {layer_impl.__class__.__name__}."
                )
            if dcp_size > 1:
                assert layer_impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer_impl.__class__.__name__} "
                    "does not return the softmax lse for decode."
                )

            if pcp_size > 1:
                assert layer_impl.supports_pcp, (
                    "PCP requires attention impls' support, "
                    f"but the impl {layer_impl.__class__.__name__} "
                    "does not support PCP."
                )


# KV placement over the PCP group (ParallelConfig.cp_kv_placement). Set by the
# model runner in each worker; "head" / "replicate" mean the KV positions are
# not interleaved over PCP, so slot mapping and block counts ignore PCP.
_CP_KV_PLACEMENT = "interleave"


def set_cp_kv_placement(placement: str) -> None:
    global _CP_KV_PLACEMENT
    assert placement in ("interleave", "head", "replicate"), placement
    _CP_KV_PLACEMENT = placement


def get_cp_kv_placement() -> str:
    return _CP_KV_PLACEMENT


def get_kv_pcp_world_size_and_rank() -> tuple[int, int]:
    """(world size, rank) of the PCP group as seen by the KV layout."""
    if _CP_KV_PLACEMENT != "interleave":
        return 1, 0
    try:
        return get_pcp_group().world_size, get_pcp_group().rank_in_group
    except AssertionError:
        # PCP might not be initialized in testing
        return 1, 0


def cp_kv_heads_per_rank(num_kv_heads: int, pcp_world_size: int) -> int:
    """KV heads stored per rank under the "head" placement (Ulysses):
    num_kv_heads / pcp, or 1 (replicated heads) when pcp > num_kv_heads."""
    if pcp_world_size <= 1:
        return num_kv_heads
    if pcp_world_size <= num_kv_heads:
        assert num_kv_heads % pcp_world_size == 0, (num_kv_heads, pcp_world_size)
        return num_kv_heads // pcp_world_size
    assert pcp_world_size % num_kv_heads == 0, (num_kv_heads, pcp_world_size)
    return 1


def get_total_cp_world_size():
    pcp_world_size, _ = get_kv_pcp_world_size_and_rank()
    try:
        dcp_world_size = get_dcp_group().world_size
    except AssertionError:
        # DCP might not be initialized in testing
        dcp_world_size = 1
    return dcp_world_size * pcp_world_size
