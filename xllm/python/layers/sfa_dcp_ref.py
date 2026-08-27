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
"""Golden torch implementations of DCP SFA pre/post compute.

These functions are the extracted decode-path naive torch ops from
``sfa_dcp.py``. Communication (AllGather / AllToAll) stays outside.

Fusion groups (comm barriers kept independent):

* pack_query_for_allgather  — AllGather send layout
* unpack_query_after_allgather — AllGather recv layout
* remap_sparse_indices — localize + compact owned top-k
* pack_a2a_payloads — LSE = max+log(sum) + AllToAll send layout
* merge_dcp_outputs — LSE softmax merge + restore TND
"""

from __future__ import annotations

import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout


def pack_query_for_allgather(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
) -> torch.Tensor:
    """Pack Q fragments and move heads to dim0 for dim0 AllGather.

    ``ql_nope, q_pe``: ``[T, H, D_*]`` -> send ``[H, T, D_nope + D_pe]``.
    """
    if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
        raise RuntimeError(
            "Cannot pack DCP query gather for ql_nope/q_pe with "
            f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
            f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
        )
    fused_q = torch.cat([ql_nope, q_pe], dim=-1)
    return fused_q.permute(1, 0, 2).contiguous()


def unpack_query_after_allgather(
    gathered: torch.Tensor,
    nope_dim: int,
    pe_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Restore AllGather output ``[H_all, T, D_fused]`` to TND Q fragments."""
    if gathered.shape[-1] != nope_dim + pe_dim:
        raise RuntimeError(f"DCP query unpack expected fused last-dim {nope_dim + pe_dim}, got {gathered.shape[-1]}.")
    restored = gathered.permute(1, 0, 2).contiguous()
    ql_nope, q_pe = torch.split(restored, (nope_dim, pe_dim), dim=-1)
    return ql_nope, q_pe


def remap_sparse_indices(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
) -> torch.Tensor:
    """Localize logical top-k slots and compact owned entries to the front.

    Equivalent to the original sort-based pack: owned keep original order,
    unowned become ``-1`` in the tail. ``index_topk`` is the configured
    maximum last-dim; runtime last-dim may be smaller.
    """
    if layout.shard_size <= 1:
        return topk_indices

    topk_count = topk_indices.shape[-1]
    if topk_count > index_topk:
        raise RuntimeError(f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({index_topk}).")

    local_table = layout.localize_slots(topk_indices)
    owned_entries = local_table >= 0
    original_order = torch.arange(
        topk_count,
        dtype=torch.float32,
        device=topk_indices.device,
    ).expand_as(topk_indices)
    pack_keys = original_order + (~owned_entries).to(torch.float32) * topk_count
    _, pack_order = torch.sort(pack_keys, dim=-1)
    return torch.gather(local_table, dim=-1, index=pack_order.to(torch.int32))


def remap_sparse_indices_compact(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
) -> torch.Tensor:
    """Same semantics as ``remap_sparse_indices`` without ``torch.sort``.

    Used as the TileLang kernel golden: owned entries keep relative order.
    """
    if layout.shard_size <= 1:
        return topk_indices

    topk_count = topk_indices.shape[-1]
    if topk_count > index_topk:
        raise RuntimeError(f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({index_topk}).")

    local_table = layout.localize_slots(topk_indices)
    owned_entries = local_table >= 0
    owned_count = owned_entries.to(torch.int32).sum(dim=-1, keepdim=True)
    # Stable compact: valid items first, original order preserved.
    order = torch.arange(topk_count, device=topk_indices.device, dtype=torch.int64)
    # False sorts before True in ascending, so invert owned to send them first.
    sort_key = (~owned_entries).to(torch.int64) * topk_count + order
    pack_order = torch.argsort(sort_key, dim=-1, stable=True)
    packed = torch.gather(local_table, dim=-1, index=pack_order.to(torch.int64))
    tail = torch.arange(topk_count, device=topk_indices.device).expand_as(packed)
    packed = torch.where(
        tail >= owned_count,
        torch.full_like(packed, KVShardLayout.INVALID_SLOT),
        packed,
    )
    return packed


def pack_a2a_payloads(
    sfa_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse LSE compute with AllToAll send layouts.

    ``sfa_output``: ``[T, H, D]``
    ``softmax_max/sum``: ``[N2, T, G]`` with ``N2 * G == H``
    Returns ``out_send [H, T, D]``, ``lse_send [H, T, 1]`` (fp32).
    """
    softmax_lse = softmax_max + torch.log(softmax_sum)
    softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
    if softmax_lse.shape[1] != sfa_output.shape[1]:
        raise RuntimeError(
            "DCP A2A pack expects LSE heads to match SFA output heads, "
            f"got lse {tuple(softmax_lse.shape)} and out {tuple(sfa_output.shape)}."
        )
    out_send = sfa_output.movedim(1, 0).contiguous()
    lse_send = softmax_lse.movedim(1, 0).contiguous()
    return out_send, lse_send


def merge_dcp_outputs(
    output_recv: torch.Tensor,
    lse_recv: torch.Tensor,
) -> torch.Tensor:
    """Merge AllToAll shards with LSE softmax and restore TND.

    ``output_recv``: ``[dcp, H_local, T, D]``
    ``lse_recv``: ``[dcp, H_local, T]``
    Returns ``[T, H_local, D]`` in the output dtype of ``output_recv``.
    """
    if output_recv.ndim != 4 or lse_recv.ndim != 3 or output_recv.shape[:3] != lse_recv.shape:
        raise RuntimeError(
            "DCP output merge expects matching rank/token/head dimensions, "
            f"got {tuple(output_recv.shape)} and {tuple(lse_recv.shape)}."
        )
    output_dtype = output_recv.dtype
    lse_recv = lse_recv.masked_fill(~torch.isfinite(lse_recv), float("-inf"))
    weights = torch.softmax(lse_recv, dim=0)
    weights = torch.nan_to_num(weights, nan=0.0)

    output = (output_recv.to(lse_recv.dtype) * weights.unsqueeze(-1)).sum(dim=0)
    return output.movedim(1, 0).contiguous().to(output_dtype)
