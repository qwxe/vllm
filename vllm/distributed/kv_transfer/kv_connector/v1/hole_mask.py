# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes/core: KV-connector API for block-granular (non-prefix) hits.

The stock scheduler-side API, ``get_num_new_matched_tokens``, returns a token
count: the connector can serve a *prefix* [num_computed, num_computed + n).
With holes the GPU prefix cache already holds blocks after a miss, and a
lower tier (CPU pool, peer GPU) may hold any subset of the prompt's blocks.
A connector that implements this mixin is asked for a per-block hit mask
instead, and is told which block ranges to load into the blocks the
scheduler allocated in place.

Scheduler protocol (``Scheduler.schedule``, waiting requests, holes enabled):

1. ``local, bs = kv_cache_manager.get_cache_hits_with_holes(request)``
2. ``mask, load_async = connector.get_block_hit_mask(request, local_hit)``
   where ``local_hit[i]`` tells whether block i hits in the GPU cache.
   ``mask`` is None: not ready, ask again next step (as with
   ``get_num_new_matched_tokens`` returning None).
3. Blocks hit by either tier form the cached set; the frontier is the end of
   the last hit block; blocks that miss in both before the frontier are the
   holes (recomputed); blocks that only the connector has are loaded.
4. After allocation: ``connector.update_state_after_alloc_with_ranges(
   request, blocks, load_ranges)`` with the token ranges to load; ``blocks``
   is the request's full, position-ordered block list. The scheduler caches
   the loaded blocks at schedule time, as stock vLLM does for synchronously
   loaded external tokens, so the load must complete before the layer's
   attention runs (``start_load_kv`` / ``wait_for_layer_load``).

Not supported yet (the scheduler falls back to treating connector-only blocks
as misses): asynchronous loads (``load_async=True``), i.e. the
WAITING_FOR_REMOTE_KVS path. The CPU-pool connector with the ``rate``
eviction policy is a later deliverable; this file only fixes the interface.

A connector that does not implement the mixin keeps prefix semantics: with
such a connector configured, the scheduler uses the stock prefix-only lookup
(no holes), so its behaviour is unchanged.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


class SupportsBlockHitMask(ABC):
    """Mixin for a scheduler-side KV connector that can serve arbitrary
    blocks of a prompt, not only a prefix."""

    @abstractmethod
    def get_block_hit_mask(
        self, request: "Request", local_hit: list[bool]
    ) -> tuple[list[bool] | None, bool]:
        """Which full blocks of the request's prompt the connector can load.

        Args:
            request: the request (``request.block_hashes`` are the chained
                block hashes; block i covers tokens [i*bs, (i+1)*bs)).
            local_hit: per-block GPU prefix-cache hits up to the last local
                hit (may be shorter than the prompt).

        Returns:
            (mask, load_async). ``mask[i]`` is True if block i can be loaded;
            entries where ``local_hit`` is True are ignored. The mask may be
            shorter than the prompt (missing entries = False). At most
            ``(request.num_tokens - 1) // block_size`` entries are used.
            ``mask`` None means "not determined yet, ask again".
        """
        raise NotImplementedError

    @abstractmethod
    def update_state_after_alloc_with_ranges(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        load_ranges: list[tuple[int, int]],
    ) -> None:
        """Record the loads for this request after allocation.

        Args:
            request: the request.
            blocks: all blocks of the request, position ordered.
            load_ranges: block-aligned token ranges to load into ``blocks``.
        """
        raise NotImplementedError
