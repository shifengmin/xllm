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
#include <string>

namespace xllm {

[[nodiscard]] inline bool is_layerwise_split_supported_model(
    const std::string& model_type) noexcept {
  return model_type == "deepseek_v32" || model_type == "glm_moe_dsa";
}

[[nodiscard]] inline int32_t effective_layerwise_split_size(
    int32_t configured,
    bool is_draft_model,
    const std::string& model_type) noexcept {
  if (configured <= 1 || is_draft_model ||
      !is_layerwise_split_supported_model(model_type)) {
    return 1;
  }
  return configured;
}

inline void validate_layerwise_split_size_config(int32_t layerwise_split_size) {
  CHECK_GE(layerwise_split_size, 1)
      << "layerwise_split_size must be >= 1, got " << layerwise_split_size;
}

class LayerwiseSplitPlacement final {
 public:
  LayerwiseSplitPlacement(bool enabled, int32_t group_size, int32_t local_rank)
      : enabled_(enabled), group_size_(group_size), local_rank_(local_rank) {
    CHECK_GT(group_size_, 0) << "Layerwise split group size must be positive.";
    CHECK_GE(local_rank_, 0) << "Layerwise split local rank must be non-negative.";
    CHECK_LT(local_rank_, group_size_)
        << "Layerwise split local rank must be smaller than group size.";
  }

  [[nodiscard]] int32_t owner_rank(int64_t layer_id) const {
    CHECK_GE(layer_id, 0) << "Layer id must be non-negative.";
    return static_cast<int32_t>(layer_id % group_size_);
  }

  [[nodiscard]] bool owns(int64_t layer_id) const {
    return !enabled_ || owner_rank(layer_id) == local_rank_;
  }

 private:
  bool enabled_ = false;
  int32_t group_size_ = 1;
  int32_t local_rank_ = 0;
};

}  // namespace xllm
