# SPDX-License-Identifier: Apache-2.0
"""Helpers for the holes/core CPU tests (no GPU, no network)."""

import os

import torch

from vllm.config import (
    CacheConfig,
    ModelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

BS = 16
TINY = os.path.join(os.path.dirname(__file__), "tiny_qwen2")
init_none_hash(sha256)
HASHER = get_request_block_hasher(BS, sha256)


def make_request(rid: str, prompt: list[int], max_tokens: int = 4) -> Request:
    return Request(
        request_id=rid,
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=HASHER,
    )


def kv_cache_config(num_blocks: int) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=1, dtype=torch.float32))
        ],
    )


def make_scheduler(
    holes: dict | None,
    num_blocks: int = 1000,
    max_num_batched_tokens: int = 2048,
    long_prefill_token_threshold: int = 0,
) -> Scheduler:
    model_config = ModelConfig(
        model=TINY, skip_tokenizer_init=True, max_model_len=8192, dtype="bfloat16"
    )
    sched = SchedulerConfig(
        max_num_seqs=16,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=8192,
        enable_chunked_prefill=True,
        long_prefill_token_threshold=long_prefill_token_threshold,
        is_encoder_decoder=False,
    )
    cache = CacheConfig(
        block_size=BS,
        gpu_memory_utilization=0.9,
        cache_dtype="auto",
        enable_prefix_caching=True,
    )
    cache.num_gpu_blocks = num_blocks
    vc = VllmConfig(
        model_config=model_config,
        cache_config=cache,
        scheduler_config=sched,
        additional_config={"holes": holes} if holes is not None else {},
    )
    return Scheduler(
        vllm_config=vc,
        kv_cache_config=kv_cache_config(num_blocks),
        structured_output_manager=StructuredOutputManager(vc),
        block_size=BS,
        log_stats=True,
    )


def fake_output(sched_out, token: int = 7) -> ModelRunnerOutput:
    """Sample `token` for every request whose prefill is complete after the
    step (as the GPU runner's discard mask does)."""
    req_ids = list(sched_out.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={r: i for i, r in enumerate(req_ids)},
        sampled_token_ids=[[token] for _ in req_ids],
    )


def step(scheduler: Scheduler):
    out = scheduler.schedule()
    mro = fake_output(out)
    # Mirror the runner: discard samples of partial prefills.
    for i, rid in enumerate(mro.req_ids):
        req = scheduler.requests[rid]
        if req.num_computed_tokens < req.num_tokens or req.pending_hole_ranges:
            mro.sampled_token_ids[i] = []
    scheduler.update_from_output(out, mro)
    return out
