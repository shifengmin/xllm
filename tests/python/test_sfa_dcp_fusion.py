# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TileLang SFA DCP remap kernel precision vs naive torch golden."""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp_ref import remap_sparse_indices

TOPK = 2048


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _make_logical_slots(
    num_tokens: int,
    layout: KVShardLayout,
    *,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(0)
    slots = torch.randint(
        0,
        8 * layout.logical_block_size,
        (num_tokens, TOPK),
        device=device,
        dtype=torch.int32,
    )
    mask = torch.rand((num_tokens, TOPK), device=device) < 0.25
    return torch.where(mask, torch.full_like(slots, KVShardLayout.INVALID_SLOT), slots)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_fused_remap_matches_naive() -> None:
    from xllm.python.kernels_npu.tilelang.sfa_dcp_remap import fused_remap_sparse_indices

    device = torch.device("npu")
    cases = (
        (128, 4, 2, 1),
        (128, 4, 2, 8),
        (64, 2, 1, 8),
        (128, 32, 7, 8),
    )
    for physical_block_size, dcp_size, dcp_rank, num_tokens in cases:
        layout = KVShardLayout(
            physical_block_size=physical_block_size,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
        )
        slots = _make_logical_slots(num_tokens, layout, device=device)
        out = torch.empty_like(slots)
        scratch = torch.empty(num_tokens * TOPK, dtype=torch.int32, device=device)
        fused = fused_remap_sparse_indices(
            slots,
            layout.physical_block_size,
            layout.dcp_size,
            layout.dcp_rank,
            out,
            scratch,
        )
        torch.npu.synchronize()
        naive = remap_sparse_indices(slots, layout, index_topk=TOPK)
        assert torch.equal(fused, naive), f"remap mismatch pb={physical_block_size} dcp={dcp_size} T={num_tokens}"
