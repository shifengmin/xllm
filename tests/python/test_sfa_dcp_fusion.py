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

"""SFA DCP remap precision: CPU golden and optional AOT kernel vs naive torch."""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp_ref import remap_sparse_indices

TOPK = 2048
# GLM-5.3: index_kpool=4 with always-select-tail, so the tail is index_kpool - 1.
KPOOL_TAIL_COLS = 3


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _aot_remap_available() -> bool:
    ops = getattr(torch.ops, "xllm_ops", None)
    return ops is not None and hasattr(ops, "sfa_dcp_remap_out")


def _make_logical_slots(
    num_tokens: int,
    layout: KVShardLayout,
    *,
    device: torch.device,
    width: int = TOPK,
) -> torch.Tensor:
    torch.manual_seed(0)
    slots = torch.randint(
        0,
        8 * layout.logical_block_size,
        (num_tokens, width),
        device=device,
        dtype=torch.int32,
    )
    mask = torch.rand((num_tokens, width), device=device) < 0.25
    return torch.where(mask, torch.full_like(slots, KVShardLayout.INVALID_SLOT), slots)


def test_remap_sparse_indices_packs_owned_slots() -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    slots = torch.tensor(
        [
            [0, 5, -1, 8],
            [1, 4, 9, -1],
        ],
        dtype=torch.int32,
    )
    remapped = remap_sparse_indices(slots, layout, index_topk=4)
    assert remapped.shape == slots.shape
    owned = remapped >= 0
    assert owned[0].tolist() == [True, True, False, False]
    assert owned[1].tolist() == [True, True, False, False]


def test_remap_sparse_indices_accepts_kpool_tail_wider_than_index_topk() -> None:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    slots = torch.tensor([[0, 5, -1, 8, 12, 1, 4]], dtype=torch.int32)
    remapped = remap_sparse_indices(slots, layout, index_topk=4)
    assert remapped.shape == slots.shape
    prefix = remap_sparse_indices(slots[..., :4], layout, index_topk=4)
    tail = layout.localize_slots(slots[..., 4:])
    assert torch.equal(remapped[..., :4], prefix)
    assert torch.equal(remapped[..., 4:], tail)
    owned = remapped[remapped >= 0]
    assert set(owned.tolist()) == {0, 4, 1}


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
@pytest.mark.skipif(
    not _aot_remap_available(),
    reason="TileLang AOT sfa_dcp_remap_out is not registered",
)
def test_fused_remap_matches_naive() -> None:
    """Both widths the indexer emits: exactly ``index_topk``, and the wider
    kPool tail (``topk + index_kpool - 1``, what GLM-5.3 ships). For the wide
    case the kernel only remaps the configured prefix and the tail is localized
    by the layout, so the assembled result is what the model actually feeds SFA.
    """
    device = torch.device("npu")
    cases = (
        (128, 4, 2, 1),
        (128, 4, 2, 8),
        (64, 2, 1, 8),
        (128, 32, 7, 8),
    )
    for physical_block_size, dcp_size, dcp_rank, num_tokens in cases:
        for width in (TOPK, TOPK + KPOOL_TAIL_COLS):
            layout = KVShardLayout(
                physical_block_size=physical_block_size,
                dcp_size=dcp_size,
                dcp_rank=dcp_rank,
            )
            slots = _make_logical_slots(num_tokens, layout, device=device, width=width)
            prefix = slots[..., :TOPK].contiguous()
            out = torch.empty_like(prefix)
            scratch = torch.empty(num_tokens * TOPK, dtype=torch.int32, device=device)
            fused = torch.ops.xllm_ops.sfa_dcp_remap_out(
                prefix,
                layout.physical_block_size,
                layout.dcp_size,
                layout.dcp_rank,
                out,
                scratch,
            )
            torch.npu.synchronize()
            if width > TOPK:
                fused = torch.cat([fused, layout.localize_slots(slots[..., TOPK:])], dim=-1)
            naive = remap_sparse_indices(slots, layout, index_topk=TOPK)
            assert torch.equal(fused, naive), (
                f"remap mismatch pb={physical_block_size} dcp={dcp_size} T={num_tokens} width={width}"
            )
