# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Synced from the standalone repo /home/asd/vllm-dev-kernels (msa/).
"""MSA: position-explicit multi-segment paged attention (hole-aware prefill / CP local kernel)."""

from .api import merge_states, msa, msa_attention
from .planner import MsaPlan, plan_msa, plan_msa_rows
from .cp_adapter import MsaLocalAttention, msa_paged_local

__all__ = ["msa", "msa_attention", "merge_states", "plan_msa", "plan_msa_rows", "MsaPlan",
           "MsaLocalAttention", "msa_paged_local"]
