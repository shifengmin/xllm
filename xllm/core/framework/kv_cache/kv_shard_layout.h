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

#pragma once

#include <cstdint>

namespace xllm {

struct ParallelArgs;

// Maps a logical paged-KV coordinate onto one rank's physical cache.
// DCP and KV-split share this packing: a logical block is
// `physical_block_size * shard_size` tokens, and rank `r` stores the r-th
// physical slice. Callers pick the source of `shard_size` once at
// construction (`from_dcp` or `from_kv_split`); layout math only sees the
// stored shard width.
class KVShardLayout final {
 public:
  static constexpr int64_t kInvalidSlot = -1;

  static KVShardLayout from_dcp(int32_t physical_block_size,
                                int32_t dcp_size,
                                int32_t dcp_rank);
  static KVShardLayout from_kv_split(int32_t physical_block_size,
                                     int32_t kv_split_size,
                                     int32_t kv_split_rank);
  static KVShardLayout from_parallel_args(int32_t physical_block_size,
                                          const ParallelArgs& parallel_args);

  int32_t physical_block_size() const { return physical_block_size_; }
  int32_t shard_size() const { return shard_size_; }
  int32_t shard_rank() const { return shard_rank_; }
  int64_t logical_block_size() const {
    return static_cast<int64_t>(physical_block_size_) * shard_size_;
  }

  int32_t owner_of(int64_t global_slot) const;
  bool owns(int64_t global_slot) const;
  int64_t localize(int64_t global_slot) const;
  int64_t globalize(int64_t local_slot) const;
  int64_t local_seq_len(int64_t global_seq_len) const;

 private:
  KVShardLayout(int32_t physical_block_size,
                int32_t shard_size,
                int32_t shard_rank);

  int32_t physical_block_size_ = 1;
  int32_t shard_size_ = 1;
  int32_t shard_rank_ = 0;
};

}  // namespace xllm
