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

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.layers.sfa_dcp import AscendSFADCPImpl, AscendSFADCPMetadataBuilder
from xllm.python.model_executor.forward_context import get_forward_context

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


def dcp_layer_options(layer: Attention) -> tuple[int, int, int]:
    """Read SFA DCP knobs from a model attention layer, with safe defaults."""
    cfg = getattr(layer, "cfg", None)
    index_topk = int(getattr(cfg, "index_topk", 2048)) if cfg is not None else 2048
    interleave_size = int(getattr(cfg, "cp_kv_cache_interleave_size", 1)) if cfg is not None else 1
    num_nextn = int(getattr(cfg, "num_nextn_predict_layers", 0)) if cfg is not None else 0
    return index_topk, interleave_size, 1 + num_nextn


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


def _actual_seq_lengths(
    metadata: AttentionMetadata,
    num_reqs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_seq_lens = metadata.kv_seq_lens
    if kv_seq_lens is None:
        raise RuntimeError("SFA DCP requires kv_seq_lens in attention metadata.")
    actual_seq_kv = kv_seq_lens.to(dtype=torch.int32, device=device)[:num_reqs]
    q_cu_seq_lens = metadata.q_cu_seq_lens
    if q_cu_seq_lens is not None:
        cu = q_cu_seq_lens.to(dtype=torch.int32, device=device)
        if cu.numel() == num_reqs:
            actual_seq_q = cu
        elif cu.numel() == num_reqs + 1:
            actual_seq_q = cu[1:]
        else:
            actual_seq_q = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    else:
        actual_seq_q = torch.arange(1, num_reqs + 1, dtype=torch.int32, device=device)
    return actual_seq_q, actual_seq_kv


class SfaDcpAttentionBackend(NpuPagedAttentionBackend):
    """Paged MLA backend whose ``execute_mla`` runs SFA DCP.

    Indexer metadata, KV bind, and graph ``prepare`` stay on the NPU paged
    backend so decode ACL-graph capture does not special-case DCP.
    Frozen ``sfa_dcp`` still sees a vLLM-shaped coordinator and metadata.
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
        cp_kv_cache_interleave_size: int,
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
        coordinator = _coordinator(dcp_group)
        self._dcp_group = dcp_group
        self._interleave_size = cp_kv_cache_interleave_size
        self._decode_threshold = decode_threshold
        self._impl = AscendSFADCPImpl(
            coordinator,
            scale=scale,
            index_topk=index_topk,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            device=device,
        )
        self._builder = AscendSFADCPMetadataBuilder(
            coordinator,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            device=device,
            max_num_reqs=max_num_reqs,
            decode_threshold=decode_threshold,
        )

    def _ensure_builder_capacity(self, num_reqs: int) -> None:
        if num_reqs <= self._builder.dcp_local_seq_lens_buf.shape[0]:
            return
        self._builder = AscendSFADCPMetadataBuilder(
            _coordinator(self._dcp_group),
            cp_kv_cache_interleave_size=self._interleave_size,
            device=self.device,
            max_num_reqs=num_reqs,
            decode_threshold=self._decode_threshold,
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
        ctx = get_forward_context()
        metadata = ctx.metadata
        layer_cache = ctx.layer_caches[layer.layer_id]
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")
        if metadata.block_table is None:
            raise RuntimeError("SFA DCP requires a block table.")
        if metadata.kv_seq_lens is None:
            raise RuntimeError("SFA DCP requires kv_seq_lens.")

        num_reqs = int(metadata.block_table.shape[0])
        num_input_tokens = int(q_latent.shape[0])
        self._ensure_builder_capacity(num_reqs)
        seq_lens = metadata.kv_seq_lens.to(dtype=torch.int32)[:num_reqs]
        is_prefilling = None
        if metadata.is_prefill or metadata.is_chunked_prefill:
            is_prefilling = torch.ones(num_reqs, dtype=torch.bool)
        attn_metadata = self._builder.build(
            slot_mapping=metadata.slot_mapping,
            block_table=metadata.block_table.to(torch.int32),
            seq_lens=seq_lens,
            query_start_loc_cpu=_query_start_loc_cpu(metadata, num_reqs),
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            dcp_local_seq_lens=seq_lens,
            is_prefilling=is_prefilling,
        )

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
        actual_seq_q, actual_seq_kv = _actual_seq_lengths(metadata, num_reqs, q_latent.device)
        return self._impl._execute_sparse_flash_attention_process(
            q_latent,
            q_pe,
            kv_cache,
            topk,
            attn_metadata,
            actual_seq_q,
            actual_seq_kv,
        )
