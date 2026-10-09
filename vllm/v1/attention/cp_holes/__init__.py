# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Context parallelism for prefills whose cached context has holes.

See CP-IMPL.md at the repository root for the design as built.
"""

from .local_attn import (KeySet, LocalAttention, RefLocalAttention,
                         get_local_attention, merge_attn_states,
                         set_local_attention)
from .plan import StepPlan, home_rank, rows_c2e, rows_contiguous
from .schemes import (AdaptiveCP, AllGatherKVCP, CPComm, PassQCP, RingCP,
                      UlyssesCP, make_scheme)

__all__ = [
    "KeySet", "LocalAttention", "RefLocalAttention", "get_local_attention",
    "set_local_attention", "merge_attn_states", "StepPlan", "home_rank",
    "rows_c2e", "rows_contiguous", "CPComm", "UlyssesCP", "PassQCP",
    "RingCP", "AllGatherKVCP", "AdaptiveCP", "make_scheme"
]
