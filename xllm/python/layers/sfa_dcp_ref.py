# Copyright 2026 The xLLM Authors.
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
"""Torch reference ops for DCP SFA decode remap."""

from __future__ import annotations

import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout


def _pack_owned_slots(local_table: torch.Tensor) -> torch.Tensor:
    topk_count = local_table.shape[-1]
    owned_entries = local_table >= 0
    original_order = torch.arange(
        topk_count,
        dtype=torch.float32,
        device=local_table.device,
    ).expand_as(local_table)
    pack_keys = original_order + (~owned_entries).to(torch.float32) * topk_count
    _, pack_order = torch.sort(pack_keys, dim=-1)
    return torch.gather(local_table, dim=-1, index=pack_order.to(torch.int32))


def remap_sparse_indices(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
) -> torch.Tensor:
    """Pack this rank's owned sparse slots to the front of the SFA width.

    ``index_topk`` is the configured SFA width. A kPool indexer may emit a
    wider last dim (``topk + index_kpool - 1`` when always-select-tail is on).
    The prefix of ``index_topk`` is packed for SFA; the remaining tail columns
    are localized (not packed) and concatenated after it, so the fused kernel
    can still remap the prefix.
    """
    last_dim = int(topk_indices.shape[-1])
    if last_dim < index_topk:
        raise RuntimeError(
            f"sparse indices last dim ({last_dim}) is narrower than the configured index_topk ({index_topk})."
        )
    prefix = topk_indices[..., :index_topk] if last_dim > index_topk else topk_indices
    remapped = _pack_owned_slots(layout.localize_slots(prefix))
    if last_dim == index_topk:
        return remapped
    tail = layout.localize_slots(topk_indices[..., index_topk:])
    return torch.cat([remapped, tail], dim=-1)
