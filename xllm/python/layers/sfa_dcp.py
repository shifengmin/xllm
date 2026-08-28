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
"""DCP sparse-flash-attention decode path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, NamedTuple, Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp_ref import merge_dcp_outputs, remap_sparse_indices
from xllm.python.model_executor.forward_context import get_execution_buffer

_REMAP_VEC_LEN = 64
_SFA_DCP_FUSION_ENV = "XLLM_SFA_DCP_FUSION"
_SFA_DCP_FUSION_FUSED = "fused"
_SFA_DCP_FUSION_NAIVE = "naive"
_MAX_ATTENTION_UPDATE_SP = 16
_ATTENTION_UPDATE_HEAD_DIM_ALIGN = 8


def sfa_dcp_fusion_mode() -> str:
    raw = os.getenv(_SFA_DCP_FUSION_ENV, _SFA_DCP_FUSION_FUSED).strip().lower()
    if raw not in {_SFA_DCP_FUSION_FUSED, _SFA_DCP_FUSION_NAIVE}:
        raise RuntimeError(
            f"{_SFA_DCP_FUSION_ENV} must be '{_SFA_DCP_FUSION_FUSED}' or '{_SFA_DCP_FUSION_NAIVE}', got {raw!r}."
        )
    return raw


def merge_dcp_outputs_with_attention_update(
    output_recv: torch.Tensor,
    lse_recv: torch.Tensor,
    merged: torch.Tensor,
) -> torch.Tensor:
    """Merge DCP shards with ``npu_attention_update`` and restore TND.

    ``output_recv``: ``[dcp, H, T, D]``, ``lse_recv``: ``[dcp, H, T]``.
    ``merged`` is caller-owned ``[T, H, D]`` so ACL graph replay can reuse it.
    Rank slices are views; no extra device copy on the inputs.
    """
    import torch_npu

    if output_recv.ndim != 4 or lse_recv.ndim != 3 or output_recv.shape[:3] != lse_recv.shape:
        raise RuntimeError(
            "DCP output merge expects matching rank/token/head dimensions, "
            f"got {tuple(output_recv.shape)} and {tuple(lse_recv.shape)}."
        )
    dcp_size, num_heads, num_tokens, head_dim = output_recv.shape
    if dcp_size <= 1 or dcp_size > _MAX_ATTENTION_UPDATE_SP:
        raise RuntimeError(f"npu_attention_update expects 1 < dcp_size <= {_MAX_ATTENTION_UPDATE_SP}, got {dcp_size}.")
    if head_dim % _ATTENTION_UPDATE_HEAD_DIM_ALIGN != 0:
        raise RuntimeError(
            f"npu_attention_update expects head_dim divisible by {_ATTENTION_UPDATE_HEAD_DIM_ALIGN}, got {head_dim}."
        )
    row_count = num_heads * num_tokens
    lse_list = [lse_recv[rank].reshape(row_count) for rank in range(dcp_size)]
    local_out_list = [output_recv[rank].reshape(row_count, head_dim) for rank in range(dcp_size)]
    updated, _ = torch_npu.npu_attention_update(
        lse_list,
        local_out_list,
        0,
    )
    merged.copy_(updated.view(num_heads, num_tokens, head_dim).permute(1, 0, 2))
    return merged


class GroupCoordinator(Protocol):
    world_size: int
    rank_in_group: int
    device_group: Any


def all_gather_async(
    input: torch.Tensor,
    group: GroupCoordinator,
    output: torch.Tensor | None = None,
    async_op: bool = True,
) -> tuple[torch.Tensor, torch.distributed.Work | None]:
    if group.world_size == 1:
        return input, None
    if output is None:
        input_size = input.size()
        output_size = (input_size[0] * group.world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input.dtype, device=input.device)
    return output, dist.all_gather_into_tensor(output, input, group=group.device_group, async_op=async_op)


def get_dcp_local_seq_lens(
    seq_lens: torch.Tensor,
    layout: KVShardLayout,
) -> torch.Tensor:
    """Return this rank's local KV sequence lengths."""
    return layout.local_seq_lens(seq_lens)


def _count_prefills(
    query_start_loc_cpu: torch.Tensor,
    num_reqs: int,
    decode_threshold: int,
    is_prefilling: torch.Tensor | None,
    query_lens_cpu: torch.Tensor | None,
) -> int:
    """Count prefill requests. Short extends that are still prefilling count as prefills."""
    if num_reqs == 0:
        return 0
    query_lens_sharded = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    query_lens = query_lens_sharded if query_lens_cpu is None else query_lens_cpu
    if query_lens[0].item() > decode_threshold:
        return num_reqs
    is_prefill = query_lens > decode_threshold
    if is_prefilling is not None:
        is_prefilling_row = is_prefilling[: query_lens.shape[0]]
        if is_prefilling_row.shape[0] < query_lens.shape[0]:
            is_prefilling_row = F.pad(
                is_prefilling_row,
                (0, query_lens.shape[0] - is_prefilling_row.shape[0]),
                value=False,
            )
        is_prefill = is_prefill | is_prefilling_row
    if not torch.any(is_prefill):
        return 0
    first_prefill = int(is_prefill.int().argmax(dim=-1).item())
    return num_reqs - first_prefill


class DCPGatherContext(NamedTuple):
    """State needed to finish an async fused DCP all-gather."""

    gathered: torch.Tensor
    handle: torch.distributed.Work | None
    restore_perm: tuple[int, ...] | None
    split_sizes: tuple[int, ...]


@dataclass
class DCPContext:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    kv_gather_block_ids: torch.Tensor | None = None
    kv_gather_block_table: torch.Tensor | None = None
    gather_context: DCPGatherContext | None = None


@dataclass
class AscendSFADCPMetadata:
    num_prefills: int
    dcp_context: DCPContext


class AscendSFADCPMetadataBuilder:
    def __init__(
        self,
        dcp_group: GroupCoordinator,
        *,
        layout: KVShardLayout,
        device: torch.device,
        max_num_reqs: int,
        decode_threshold: int = 1,
    ) -> None:
        if dcp_group.world_size <= 1:
            raise RuntimeError("AscendSFADCPMetadataBuilder requires DCP world size > 1.")
        if layout.dcp_size != dcp_group.world_size:
            raise RuntimeError(
                "KVShardLayout.dcp_size must match the DCP group size, "
                f"got layout.dcp_size={layout.dcp_size}, "
                f"dcp_group.world_size={dcp_group.world_size}."
            )
        if max_num_reqs <= 0:
            raise RuntimeError(f"Invalid max_num_reqs: {max_num_reqs}")
        self.layout = layout
        self.dcp_size = layout.dcp_size
        self.dcp_rank = layout.dcp_rank
        self.device = device
        self.decode_threshold = decode_threshold
        self.dcp_local_seq_lens_buf = torch.empty(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.dcp_rank_arange = torch.arange(
            self.dcp_size,
            dtype=torch.int32,
            device=device,
        )

    def _local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        return get_dcp_local_seq_lens(seq_lens, self.layout)

    def _build_compact_kv_gather_metadata(
        self,
        dcp_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid_block_ids, compact_block_table = dcp_block_table.flatten().unique(return_inverse=True)
        compact_block_table = compact_block_table.view_as(dcp_block_table)
        num_blocks = valid_block_ids.shape[0]
        remapped_block_table = (
            compact_block_table.unsqueeze(-1) + (self.dcp_rank_arange * num_blocks).view(1, 1, -1).to(dcp_block_table)
        ).reshape(dcp_block_table.shape[0], -1)
        return valid_block_ids, remapped_block_table.to(torch.int32)

    def build(
        self,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        num_reqs: int,
        num_input_tokens: int,
        *,
        dcp_local_seq_lens: torch.Tensor | None = None,
        is_prefilling: torch.Tensor | None = None,
        query_lens_cpu: torch.Tensor | None = None,
        num_prefills: int | None = None,
    ) -> AscendSFADCPMetadata:
        dcp_block_table = block_table[:num_reqs]
        if dcp_local_seq_lens is None:
            dcp_local_seq_lens = self._local_seq_lens(seq_lens[:num_reqs])
        local_seq_lens_src = dcp_local_seq_lens[:num_reqs].to(
            device=self.device,
            dtype=torch.int32,
            non_blocking=True,
        )
        if num_reqs > self.dcp_local_seq_lens_buf.shape[0]:
            raise RuntimeError(
                f"dcp_local_seq_lens_buf is too small: "
                f"shape={tuple(self.dcp_local_seq_lens_buf.shape)}, num_reqs={num_reqs}"
            )
        self.dcp_local_seq_lens_buf[:num_reqs].copy_(local_seq_lens_src, non_blocking=True)
        local_seq_lens = self.dcp_local_seq_lens_buf[:num_reqs]

        if num_prefills is None:
            num_prefills = _count_prefills(
                query_start_loc_cpu,
                num_reqs,
                self.decode_threshold,
                is_prefilling,
                query_lens_cpu,
            )
        elif num_prefills < 0 or num_prefills > num_reqs:
            raise RuntimeError(f"num_prefills must be in [0, {num_reqs}], got {num_prefills}")
        kv_gather_block_ids = None
        kv_gather_block_table = None
        if num_prefills > 0:
            kv_gather_block_ids, kv_gather_block_table = self._build_compact_kv_gather_metadata(dcp_block_table)
        return AscendSFADCPMetadata(
            num_prefills=num_prefills,
            dcp_context=DCPContext(
                slot_mapping=slot_mapping[:num_input_tokens],
                block_table=dcp_block_table,
                seq_lens=local_seq_lens,
                kv_gather_block_ids=kv_gather_block_ids,
                kv_gather_block_table=kv_gather_block_table,
            ),
        )


class AscendSFADCPImpl:
    def __init__(
        self,
        dcp_group: GroupCoordinator,
        *,
        scale: float,
        index_topk: int,
        layout: KVShardLayout,
        device: torch.device,
    ) -> None:
        if index_topk <= 0:
            raise RuntimeError("index_topk must be a positive integer for DCP SFA.")
        self.dcp_group = dcp_group
        self.layout = layout
        self.dcp_size = layout.dcp_size
        self.dcp_rank = layout.dcp_rank
        self.scale = float(scale)
        self._dcp_index_topk = index_topk
        self._sfa_dcp_fusion_mode = sfa_dcp_fusion_mode()
        print(
            f"[sfa_dcp] {_SFA_DCP_FUSION_ENV}={self._sfa_dcp_fusion_mode}",
            flush=True,
        )

    @staticmethod
    def _remap_scratch_numel(num_tokens: int, index_topk: int) -> int:
        width = _REMAP_VEC_LEN if index_topk <= _REMAP_VEC_LEN else index_topk
        return num_tokens * width

    @staticmethod
    def _has_prefill(attn_metadata: AscendSFADCPMetadata) -> bool:
        return attn_metadata.num_prefills > 0

    def _record_dcp_kv_gather_context(
        self,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendSFADCPMetadata,
    ) -> None:
        """Start the compact KV all-gather used by prefill/mixed DCP batches."""
        if not self._has_prefill(attn_metadata):
            return
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."

        valid_block_ids = attn_metadata.dcp_context.kv_gather_block_ids
        block_table = attn_metadata.dcp_context.kv_gather_block_table
        assert valid_block_ids is not None and block_table is not None
        kv = torch.index_select(kv_cache[0], 0, valid_block_ids)
        if len(kv_cache) < 2:
            raise RuntimeError("DCP SFA KV all-gather requires nope and rope KV caches.")
        key_rope = torch.index_select(kv_cache[1], 0, valid_block_ids)
        if kv.shape[:-1] != key_rope.shape[:-1] or kv.dtype != key_rope.dtype:
            raise RuntimeError(
                "Cannot fuse DCP KV gather for KV/nope and KV/rope caches with "
                f"shapes {tuple(kv.shape)} / {tuple(key_rope.shape)} and dtypes {kv.dtype} / {key_rope.dtype}."
            )
        attn_metadata.dcp_context.gather_context = self._start_dcp_gather(
            torch.cat([kv, key_rope], dim=-1).contiguous(),
            dim=0,
            split_sizes=(kv.shape[-1], key_rope.shape[-1]),
        )

    def _start_dcp_gather(
        self,
        x: torch.Tensor,
        dim: int,
        split_sizes: tuple[int, ...],
    ) -> DCPGatherContext:
        gathered, handle, restore_perm = self._all_gather_dim_async(x, dim)
        return DCPGatherContext(
            gathered=gathered,
            handle=handle,
            restore_perm=restore_perm,
            split_sizes=split_sizes,
        )

    @staticmethod
    def _finish_dcp_gather(
        context: DCPGatherContext,
    ) -> tuple[torch.Tensor, ...]:
        if context.handle is not None:
            context.handle.wait()
        gathered = context.gathered
        if context.restore_perm is not None:
            gathered = gathered.permute(context.restore_perm).contiguous()
        return torch.split(gathered, context.split_sizes, dim=-1)

    def _all_gather_dim_async(
        self,
        x: torch.Tensor,
        dim: int,
    ) -> tuple[torch.Tensor, torch.distributed.Work | None, tuple[int, ...] | None]:
        assert self.dcp_group is not None
        if dim == 0:
            gathered, handle = all_gather_async(x.contiguous(), self.dcp_group)
            return gathered, handle, None

        perm = (dim, *[i for i in range(x.dim()) if i != dim])
        restore_perm = tuple(perm.index(i) for i in range(x.dim()))
        gathered, handle = all_gather_async(x.permute(perm).contiguous(), self.dcp_group)
        return gathered, handle, restore_perm

    def _use_fused_sfa_dcp(self) -> bool:
        return self._sfa_dcp_fusion_mode == _SFA_DCP_FUSION_FUSED

    def _remap_sparse_indices(self, topk_indices: torch.Tensor) -> torch.Tensor:
        if self.layout.dcp_size <= 1:
            return topk_indices
        if not self._use_fused_sfa_dcp():
            return remap_sparse_indices(topk_indices, self.layout, self._dcp_index_topk)

        topk_count = topk_indices.shape[-1]
        if topk_count > self._dcp_index_topk:
            raise RuntimeError(
                f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({self._dcp_index_topk})."
            )

        out = get_execution_buffer(
            ("SFA_DCP_REMAP_OUT", tuple(topk_indices.shape)),
            lambda: torch.empty_like(topk_indices),
        )
        num_tokens = int(topk_indices.numel() // topk_count)
        scratch_n = self._remap_scratch_numel(num_tokens, int(topk_count))
        idx_scratch = get_execution_buffer(
            ("SFA_DCP_REMAP_SCRATCH", scratch_n),
            lambda: torch.empty(scratch_n, dtype=torch.int32, device=topk_indices.device),
        )
        return torch.ops.xllm_ops.sfa_dcp_remap_out(
            topk_indices,
            int(self.layout.physical_block_size),
            int(self.layout.dcp_size),
            int(self.layout.dcp_rank),
            out,
            idx_scratch,
        )

    def _all_to_all_dcp_tensor(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
    ) -> torch.Tensor:
        assert self.dcp_group is not None, "DCP output All2All requires dcp_group when dcp_size > 1."
        scatter_size = tensor.shape[scatter_dim]
        if scatter_size % self.dcp_size != 0:
            raise RuntimeError(
                "DCP output All2All requires the scatter dimension to be divisible "
                f"by dcp_size, got shape={tuple(tensor.shape)}, scatter_dim={scatter_dim}, "
                f"and dcp_size={self.dcp_size}."
            )

        local_scatter_size = scatter_size // self.dcp_size
        send = tensor.movedim(scatter_dim, 0).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.dcp_group.device_group)
        recv = recv.view(self.dcp_size, local_scatter_size, *send.shape[1:])
        return recv

    def _merge_dcp_outputs_with_torch(
        self,
        output_recv: torch.Tensor,
        lse_recv: torch.Tensor,
    ) -> torch.Tensor:
        if not self._use_fused_sfa_dcp():
            return merge_dcp_outputs(output_recv, lse_recv)
        if output_recv.ndim != 4:
            raise RuntimeError(f"DCP output merge expects output_recv [dcp, H, T, D], got {tuple(output_recv.shape)}.")
        num_heads = int(output_recv.shape[1])
        num_tokens = int(output_recv.shape[2])
        head_dim = int(output_recv.shape[3])
        merged = get_execution_buffer(
            ("SFA_DCP_MERGE_OUT", num_tokens, num_heads, head_dim, str(output_recv.dtype)),
            lambda: torch.empty(
                (num_tokens, num_heads, head_dim),
                dtype=output_recv.dtype,
                device=output_recv.device,
            ),
        )
        return merge_dcp_outputs_with_attention_update(output_recv, lse_recv, merged)

    def _merge_dcp_outputs(
        self,
        sfa_output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> torch.Tensor:
        output_recv = self._all_to_all_dcp_tensor(sfa_output, 1)
        lse_recv = self._all_to_all_dcp_tensor(softmax_lse, 1).squeeze(-1)
        return self._merge_dcp_outputs_with_torch(output_recv, lse_recv)

    def _start_dcp_query_gather(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> DCPGatherContext:
        assert self.dcp_group is not None, "DCP query gather requires dcp_group when dcp_size > 1."
        if ql_nope.shape[:-1] != q_pe.shape[:-1] or ql_nope.dtype != q_pe.dtype:
            raise RuntimeError(
                "Cannot fuse DCP query gather for ql_nope/q_pe with "
                f"shapes {tuple(ql_nope.shape)} / {tuple(q_pe.shape)} "
                f"and dtypes {ql_nope.dtype} / {q_pe.dtype}."
            )

        # Avoid back-to-back DCP all_gather calls for the two SFA query
        # fragments. On Ascend the separate gathers can leave SFA with an
        # incomplete stream dependency on the first prefill. Native DCP
        # restores query shards on dim 1.
        fused_q = torch.cat([ql_nope, q_pe], dim=-1).contiguous()
        return self._start_dcp_gather(
            fused_q,
            dim=1,
            split_sizes=(ql_nope.shape[-1], q_pe.shape[-1]),
        )

    def _record_query_gather_context(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        attn_metadata: AscendSFADCPMetadata,
    ) -> None:
        # Prefill/mixed batches gather compact KV after its cache write instead.
        # Keeping Q local avoids a full query all-gather and the subsequent LSE
        # output merge in the all-KV attention path.
        if self._has_prefill(attn_metadata):
            return
        attn_metadata.dcp_context.gather_context = self._start_dcp_query_gather(ql_nope, q_pe)

    def _store_parallel_kv(
        self,
        k_pe: torch.Tensor | None,
        k_nope: torch.Tensor | None,
        k_li: torch.Tensor | None,
        kv_cache: tuple[torch.Tensor, ...] | None,
        attn_metadata: AscendSFADCPMetadata,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        # Prefill DCP gathers referenced blocks after the current layer writes
        # its SFA KV cache and before indexer/top-k work begins.
        if kv_cache is not None:
            self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
        return k_pe, k_nope, k_li

    def _npu_sparse_flash_attention(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
        block_table: torch.Tensor,
        *,
        sparse_mode: int,
        return_lse: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kv = kv_cache[0]
        attention_out, softmax_max, softmax_sum = torch.ops.xllm_ops.sparse_flash_attention_lse(
            query=ql_nope,
            key=kv,
            value=kv,
            sparse_indices=topk_indices,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_key,
            query_rope=q_pe,
            key_rope=kv_cache[1],
            scale_value=self.scale,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            attention_mode=2,
            return_softmax_lse=return_lse,
        )
        if return_lse:
            return attention_out, softmax_max, softmax_sum
        return attention_out

    def _execute_sparse_flash_attention_process(
        self,
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        topk_indices: torch.Tensor,
        attn_metadata: AscendSFADCPMetadata,
        actual_seq_lengths_query: torch.Tensor,
        actual_seq_lengths_key: torch.Tensor,
    ) -> torch.Tensor:
        assert self.dcp_group is not None, "DCP SFA requires dcp_group when dcp_size > 1."
        dcp_context = attn_metadata.dcp_context
        if self._has_prefill(attn_metadata):
            gather_context = dcp_context.gather_context
            dcp_context.gather_context = None
            if gather_context is None:
                # The normal forward path starts this after KV writes so it can
                # overlap indexer selection. Keep a synchronous fallback for
                # callers that invoke this method outside that path.
                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)
                gather_context = dcp_context.gather_context
                dcp_context.gather_context = None
            assert gather_context is not None
            gathered_kv_cache = self._finish_dcp_gather(gather_context)
            block_table = dcp_context.kv_gather_block_table
            assert block_table is not None
            # The gathered KV cache is complete, so each rank can attend with
            # its local Q heads/tokens directly.
            return self._npu_sparse_flash_attention(
                ql_nope,
                q_pe,
                gathered_kv_cache,
                topk_indices,
                actual_seq_lengths_query,
                actual_seq_lengths_key,
                block_table,
                sparse_mode=3,
                return_lse=False,
            )

        gather_context = dcp_context.gather_context
        dcp_context.gather_context = None
        if gather_context is None:
            gather_context = self._start_dcp_query_gather(ql_nope, q_pe)
        topk_indices = self._remap_sparse_indices(topk_indices)
        ql_nope, q_pe = self._finish_dcp_gather(gather_context)
        # The replicated-view indexer already applies the causal visibility rule.
        # After DCP remaps topk indices to local KV positions, local KV
        # length no longer shares the same coordinate system as global
        # query length, so SFA must not apply its right-down causal crop.
        sfa_output, softmax_max, softmax_sum = self._npu_sparse_flash_attention(
            ql_nope,
            q_pe,
            kv_cache,
            topk_indices,
            actual_seq_lengths_query,
            dcp_context.seq_lens,
            dcp_context.block_table,
            sparse_mode=0,
            return_lse=True,
        )
        softmax_lse = softmax_max + torch.log(softmax_sum)
        softmax_lse = softmax_lse.permute(1, 0, 2).reshape(softmax_lse.shape[1], -1, 1)
        output_dtype = sfa_output.dtype
        output = self._merge_dcp_outputs(sfa_output, softmax_lse)
        return output.to(output_dtype)
