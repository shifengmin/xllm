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

"""NPU MLA backend that shards decode KV across the DCP group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

from xllm.python.attention.backend import AttentionMetadata, LayerCache, MlaIndexContext
from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.layers.sfa_dcp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
)
from xllm.python.model_executor.forward_context import copy_into_execution_buffer, get_forward_context

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


@dataclass
class _ProcessGroupCoordinator:
    """Maps an xLLM ``ProcessGroup`` onto ``sfa_dcp.GroupCoordinator``."""

    world_size: int
    rank_in_group: int
    device_group: ProcessGroup


def _coordinator(group: ProcessGroup) -> _ProcessGroupCoordinator:
    return _ProcessGroupCoordinator(
        world_size=group.size(),
        rank_in_group=group.rank(),
        device_group=group,
    )


def dcp_layer_options(layer: Attention) -> tuple[int, int]:
    """Read SFA DCP knobs from a model attention layer, with safe defaults."""
    cfg = getattr(layer, "cfg", None)
    index_topk = int(getattr(cfg, "index_topk", 2048)) if cfg is not None else 2048
    num_nextn = int(getattr(cfg, "num_nextn_predict_layers", 0)) if cfg is not None else 0
    return index_topk, 1 + num_nextn


def _query_start_loc_cpu(metadata: AttentionMetadata, num_reqs: int) -> torch.Tensor:
    q_seq_lens_host = metadata.q_seq_lens_host
    if q_seq_lens_host is not None:
        q_lens = q_seq_lens_host.to(dtype=torch.int32, device="cpu")[:num_reqs]
        zeros = torch.zeros(1, dtype=torch.int32)
        return torch.cat([zeros, torch.cumsum(q_lens, dim=0)], dim=0)
    q_cu_seq_lens = metadata.q_cu_seq_lens
    if q_cu_seq_lens is not None:
        cu = q_cu_seq_lens.to(dtype=torch.int32, device="cpu")
        if cu.numel() == num_reqs + 1:
            return cu
        if cu.numel() == num_reqs:
            zeros = torch.zeros(1, dtype=torch.int32)
            return torch.cat([zeros, cu], dim=0)
    return torch.arange(num_reqs + 1, dtype=torch.int32)


class SfaDcpAttentionBackend(NpuPagedAttentionBackend):
    """Paged MLA backend whose ``execute_mla`` runs SFA DCP.

    Decode ACL graph capture records ``execute_mla``. Replay does not re-run
    Python, so DCP slot/seq/block metadata is built in ``prepare()`` and copied
    into graph-owned buffers.
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        dcp_group: ProcessGroup,
        *,
        index_topk: int,
        decode_threshold: int,
        max_num_reqs: int,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            sliding_window=sliding_window,
            device=device,
            dtype=dtype,
        )
        self._dcp_group = dcp_group
        self._index_topk = index_topk
        self._decode_threshold = decode_threshold
        self._max_num_reqs = max_num_reqs
        self._kv_layout: KVShardLayout | None = None
        self._impl: AscendSFADCPImpl | None = None
        self._builder: AscendSFADCPMetadataBuilder | None = None
        self._local_slot_mapping: torch.Tensor | None = None
        self._expanded_indexer_block_table: torch.Tensor | None = None
        self._sfa_metadata: AscendSFADCPMetadata | None = None

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        self._kv_layout = KVShardLayout(
            self.page_size,
            self._dcp_group.size(),
            self._dcp_group.rank(),
        )
        coordinator = _coordinator(self._dcp_group)
        self._impl = AscendSFADCPImpl(
            coordinator,
            scale=self.scale,
            index_topk=self._index_topk,
            layout=self._kv_layout,
            device=self.device,
        )
        self._builder = AscendSFADCPMetadataBuilder(
            coordinator,
            layout=self._kv_layout,
            device=self.device,
            max_num_reqs=max(self._max_num_reqs, 1),
            decode_threshold=self._decode_threshold,
        )

    def _ensure_builder_capacity(self, num_reqs: int, *, graph_mode: bool) -> None:
        if self._builder is None or self._kv_layout is None:
            raise RuntimeError("SFA DCP backend requires bind_kv_caches before execute")
        if num_reqs <= self._builder.dcp_local_seq_lens_buf.shape[0]:
            return
        if graph_mode:
            raise RuntimeError(
                "SFA DCP ACL graph builder cannot grow after capture; "
                f"max_num_reqs={self._builder.dcp_local_seq_lens_buf.shape[0]}, "
                f"num_reqs={num_reqs}"
            )
        self._builder = AscendSFADCPMetadataBuilder(
            _coordinator(self._dcp_group),
            layout=self._kv_layout,
            device=self.device,
            max_num_reqs=num_reqs,
            decode_threshold=self._decode_threshold,
        )

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        super().prepare(metadata, graph_mode=graph_mode)
        self._sfa_metadata = None
        if self._kv_layout is None or self._builder is None:
            return
        if metadata.block_table is None:
            raise RuntimeError("SFA DCP requires a block table.")
        if metadata.kv_seq_lens is None:
            raise RuntimeError("SFA DCP requires kv_seq_lens.")

        local_slots = self._kv_layout.localize_slots(metadata.slot_mapping)
        if graph_mode:
            local_slots = copy_into_execution_buffer(
                ("DCP_LOCAL_SLOTS", tuple(local_slots.shape)),
                local_slots,
            )
        self._local_slot_mapping = local_slots

        if self._block_table_i32 is not None:
            expanded = self._kv_layout.expand_indexer_block_table(self._block_table_i32)
            if graph_mode:
                expanded = copy_into_execution_buffer(
                    ("DCP_INDEXER_BT", tuple(expanded.shape)),
                    expanded,
                )
                padded_rows = metadata.slot_mapping < 0
                if padded_rows.numel() == expanded.shape[0]:
                    expanded[padded_rows] = -1
            self._expanded_indexer_block_table = expanded
        else:
            self._expanded_indexer_block_table = None

        num_reqs = int(metadata.block_table.shape[0])
        num_input_tokens = int(local_slots.numel())
        self._ensure_builder_capacity(num_reqs, graph_mode=graph_mode)
        seq_lens = metadata.kv_seq_lens.to(dtype=torch.int32)[:num_reqs]
        local_seq_lens = self._kv_layout.local_seq_lens(seq_lens)
        if graph_mode:
            local_seq_lens = copy_into_execution_buffer(
                ("DCP_LOCAL_SEQ", tuple(local_seq_lens.shape)),
                local_seq_lens,
            )

        is_prefilling = None
        num_prefills = 0 if graph_mode else None
        if not graph_mode and (metadata.is_prefill or metadata.is_chunked_prefill):
            is_prefilling = torch.ones(num_reqs, dtype=torch.bool)
        query_start_loc_cpu = (
            torch.arange(num_reqs + 1, dtype=torch.int32) if graph_mode else _query_start_loc_cpu(metadata, num_reqs)
        )

        attn_metadata = self._builder.build(
            slot_mapping=local_slots,
            block_table=self._block_table_i32
            if self._block_table_i32 is not None
            else metadata.block_table.to(torch.int32),
            seq_lens=seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            dcp_local_seq_lens=local_seq_lens,
            is_prefilling=is_prefilling,
            num_prefills=num_prefills,
        )
        attn_metadata.dcp_context.slot_mapping = local_slots[:num_input_tokens]
        attn_metadata.dcp_context.seq_lens = local_seq_lens[:num_reqs]
        if self._block_table_i32 is not None:
            attn_metadata.dcp_context.block_table = self._block_table_i32[:num_reqs]
        self._sfa_metadata = attn_metadata

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        context = super().mla_index_context(layer)
        if self._expanded_indexer_block_table is None:
            return context
        return MlaIndexContext(
            index_cache=context.index_cache,
            slot_mapping=context.slot_mapping,
            block_table=self._expanded_indexer_block_table,
            actual_seq_q=context.actual_seq_q,
            actual_seq_kv=context.actual_seq_kv,
            index_cache_scale=context.index_cache_scale,
            get_quant_indexer_metadata=context.get_quant_indexer_metadata,
            update_index_cache=context.update_index_cache,
        )

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        layer: Attention,
        topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if topk is None:
            raise NotImplementedError("dense MLA (topk=None) is not supported on SfaDcpAttentionBackend")
        if self._impl is None or self._kv_layout is None:
            raise RuntimeError("SFA DCP backend requires bind_kv_caches before execute_mla")
        attn_metadata = self._sfa_metadata
        if attn_metadata is None:
            raise RuntimeError("SFA DCP execute_mla requires prepare()")
        if self._mla_actual_seq_q is None or self._mla_actual_seq_kv is None:
            raise RuntimeError("SFA DCP execute_mla requires MLA sequence lengths from prepare()")
        ctx = get_forward_context()
        layer_cache = ctx.layer_caches[layer.layer_id]
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")

        attn_metadata.dcp_context.gather_context = None
        torch.ops.xllm_ops.reshape_paged_cache(
            attn_metadata.dcp_context.slot_mapping,
            k_latent_3d,
            k_pe_3d,
            nope_cache,
            rope_cache,
        )
        kv_cache = (nope_cache, rope_cache)
        self._impl._store_parallel_kv(k_pe_3d, k_latent_3d, None, kv_cache, attn_metadata)
        self._impl._record_query_gather_context(q_latent, q_pe, attn_metadata)
        return self._impl._execute_sparse_flash_attention_process(
            q_latent,
            q_pe,
            kv_cache,
            topk,
            attn_metadata,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
        )
