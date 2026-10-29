# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""holes-v2 stage 1-2: the L2 (CPU OffloadingConnector) side of holes.

CPU only: a real Scheduler and OffloadingConnector over the mock spec and mock
manager of ``tests/v1/kv_connector/unit/offloading_connector/utils.py``.
"""

import pytest

from tests.v1.kv_connector.unit.offloading_connector.utils import RequestRunner


def _runner(holes: dict | None, **kw) -> RequestRunner:
    add = {"holes": holes} if holes is not None else None
    return RequestRunner(
        block_size=16, num_gpu_blocks=64, additional_config=add, **kw
    )


@pytest.mark.parametrize(
    "holes,expected",
    [
        (None, 1),  # holes not configured: min_reuse_blocks default 1
        ({"enabled": True}, 1),
        ({"enabled": True, "min_reuse_blocks": 3}, 3),
    ],
)
def test_min_reuse_reaches_manager(holes, expected):
    """Stage 1 (builder-r4 residual): the scheduler-side connector hands the
    hole config's ``min_reuse_blocks`` and the group kinds to the manager once,
    right after the manager is built."""
    r = _runner(holes)
    r.manager.set_group_kinds.assert_called_once_with(["full"], expected)
