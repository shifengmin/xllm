/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <glog/logging.h>

#include <cstdint>

namespace xllm {

class DecodeDcpLayerPlacement final {
 public:
  DecodeDcpLayerPlacement(bool enabled,
                          int32_t group_size,
                          int32_t local_rank)
      : enabled_(enabled),
        group_size_(group_size),
        local_rank_(local_rank) {
    CHECK_GT(group_size_, 0) << "Decode DCP group size must be positive.";
    CHECK_GE(local_rank_, 0) << "Decode DCP local rank must be non-negative.";
    CHECK_LT(local_rank_, group_size_)
        << "Decode DCP local rank must be smaller than group size.";
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  [[nodiscard]] int32_t group_size() const noexcept { return group_size_; }

  [[nodiscard]] int32_t local_rank() const noexcept { return local_rank_; }

  [[nodiscard]] int32_t owner_rank(int64_t layer_id) const {
    CHECK_GE(layer_id, 0) << "Layer id must be non-negative.";
    return static_cast<int32_t>(layer_id % group_size_);
  }

  [[nodiscard]] bool owns(int64_t layer_id) const {
    return !enabled_ || owner_rank(layer_id) == local_rank_;
  }

  [[nodiscard]] static int32_t local_rank_from_tp_rank(int32_t tp_rank,
                                                       int32_t group_size) {
    CHECK_GE(tp_rank, 0) << "TP rank must be non-negative.";
    CHECK_GT(group_size, 0) << "Decode DCP group size must be positive.";
    return tp_rank % group_size;
  }

 private:
  bool enabled_ = false;
  int32_t group_size_ = 1;
  int32_t local_rank_ = 0;
};

}  // namespace xllm
