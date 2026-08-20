/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/kv_cache/kv_shard_layout.h"

#include <algorithm>

#include <glog/logging.h>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

KVShardLayout KVShardLayout::from_dcp(int32_t physical_block_size,
                                      int32_t dcp_size,
                                      int32_t dcp_rank) {
  return KVShardLayout(physical_block_size, dcp_size, dcp_rank);
}

KVShardLayout KVShardLayout::from_kv_split(int32_t physical_block_size,
                                           int32_t kv_split_size,
                                           int32_t kv_split_rank) {
  return KVShardLayout(physical_block_size, kv_split_size, kv_split_rank);
}

KVShardLayout KVShardLayout::from_parallel_args(
    int32_t physical_block_size,
    const ParallelArgs& parallel_args) {
  return KVShardLayout(physical_block_size,
                       parallel_args.kv_shard_size(),
                       parallel_args.kv_shard_rank());
}

KVShardLayout::KVShardLayout(int32_t physical_block_size,
                             int32_t shard_size,
                             int32_t shard_rank)
    : physical_block_size_(physical_block_size),
      shard_size_(shard_size),
      shard_rank_(shard_rank) {
  CHECK_GT(physical_block_size_, 0) << "physical_block_size must be positive";
  CHECK_GT(shard_size_, 0) << "shard_size must be positive";
  CHECK_GE(shard_rank_, 0) << "shard_rank must be non-negative";
  CHECK_LT(shard_rank_, shard_size_)
      << "shard_rank must be smaller than shard_size";
}

int32_t KVShardLayout::owner_of(int64_t global_slot) const {
  CHECK_GE(global_slot, 0) << "global_slot must be non-negative";
  const int64_t logical_offset = global_slot % logical_block_size();
  return static_cast<int32_t>(logical_offset / physical_block_size_);
}

bool KVShardLayout::owns(int64_t global_slot) const {
  return global_slot >= 0 && owner_of(global_slot) == shard_rank_;
}

int64_t KVShardLayout::localize(int64_t global_slot) const {
  if (!owns(global_slot)) {
    return kInvalidSlot;
  }
  const int64_t logical_block_id = global_slot / logical_block_size();
  const int64_t local_offset = global_slot % physical_block_size_;
  return logical_block_id * physical_block_size_ + local_offset;
}

int64_t KVShardLayout::globalize(int64_t local_slot) const {
  CHECK_GE(local_slot, 0) << "local_slot must be non-negative";
  const int64_t local_block_id = local_slot / physical_block_size_;
  const int64_t local_offset = local_slot % physical_block_size_;
  return (local_block_id * shard_size_ + shard_rank_) * physical_block_size_ +
         local_offset;
}

int64_t KVShardLayout::local_seq_len(int64_t global_seq_len) const {
  CHECK_GE(global_seq_len, 0) << "global_seq_len must be non-negative";
  const int64_t logical = logical_block_size();
  const int64_t full_blocks = global_seq_len / logical;
  const int64_t remainder = global_seq_len % logical;
  const int64_t rank_start =
      static_cast<int64_t>(shard_rank_) * physical_block_size_;
  const int64_t owned_in_remainder =
      std::clamp(remainder - rank_start,
                 int64_t{0},
                 static_cast<int64_t>(physical_block_size_));
  return full_blocks * physical_block_size_ + owned_in_remainder;
}

}  // namespace xllm
