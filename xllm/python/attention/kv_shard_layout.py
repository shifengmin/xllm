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

"""Logical-to-physical paged-KV mapping shared by DCP and KV-split."""

from __future__ import annotations

import torch


class KVShardLayout:
    """Maps a logical paged-KV coordinate onto one rank's physical cache.

    DCP and KV-split share this packing: a logical block is
    ``physical_block_size * shard_size`` tokens, and rank ``r`` stores the
    r-th physical slice. Callers pick the source of ``shard_size`` once at
    construction (``from_dcp`` or ``from_kv_split``); layout math only sees
    the stored shard width.
    """

    INVALID_SLOT = -1

    def __init__(
        self,
        physical_block_size: int,
        shard_size: int,
        shard_rank: int,
    ) -> None:
        if physical_block_size <= 0:
            raise ValueError(f"physical_block_size must be positive, got {physical_block_size}")
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        if shard_rank < 0 or shard_rank >= shard_size:
            raise ValueError(
                f"shard_rank must satisfy 0 <= shard_rank < shard_size, "
                f"got shard_rank={shard_rank}, shard_size={shard_size}"
            )
        self.physical_block_size = physical_block_size
        self.shard_size = shard_size
        self.shard_rank = shard_rank

    @classmethod
    def from_dcp(
        cls,
        physical_block_size: int,
        dcp_size: int,
        dcp_rank: int,
    ) -> KVShardLayout:
        return cls(physical_block_size, dcp_size, dcp_rank)

    @classmethod
    def from_kv_split(
        cls,
        physical_block_size: int,
        kv_split_size: int,
        kv_split_rank: int,
    ) -> KVShardLayout:
        return cls(physical_block_size, kv_split_size, kv_split_rank)

    @property
    def logical_block_size(self) -> int:
        return self.physical_block_size * self.shard_size

    def local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        logical = self.logical_block_size
        physical = self.physical_block_size
        full_blocks = torch.div(seq_lens, logical, rounding_mode="floor")
        remainder = torch.remainder(seq_lens, logical)
        rank_start = self.shard_rank * physical
        owned_in_remainder = torch.clamp(remainder - rank_start, 0, physical)
        return full_blocks * physical + owned_in_remainder

    def localize_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        valid_slots = logical_slots >= 0
        safe_slots = logical_slots.clamp_min(0)
        logical_offsets = torch.remainder(safe_slots, self.logical_block_size)
        owner_ranks = torch.div(
            logical_offsets,
            self.physical_block_size,
            rounding_mode="floor",
        )
        owned_slots = valid_slots & (owner_ranks == self.shard_rank)
        logical_block_ids = torch.div(
            safe_slots,
            self.logical_block_size,
            rounding_mode="floor",
        )
        local_offsets = torch.remainder(logical_offsets, self.physical_block_size)
        local_slots = logical_block_ids * self.physical_block_size + local_offsets
        return torch.where(
            owned_slots,
            local_slots,
            torch.full_like(local_slots, self.INVALID_SLOT),
        )

    def expand_indexer_block_table(
        self,
        logical_block_table: torch.Tensor,
    ) -> torch.Tensor:
        if logical_block_table.dim() != 2:
            raise ValueError("cache-shard indexer block table must be two-dimensional")
        shard_offsets = torch.arange(
            self.shard_size,
            dtype=logical_block_table.dtype,
            device=logical_block_table.device,
        )
        expanded = logical_block_table.unsqueeze(-1) * self.shard_size + shard_offsets
        expanded = torch.where(
            logical_block_table.unsqueeze(-1) >= 0,
            expanded,
            torch.full_like(expanded, -1),
        )
        return expanded.flatten(start_dim=1).contiguous()
