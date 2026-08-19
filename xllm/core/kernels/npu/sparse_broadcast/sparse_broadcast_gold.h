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
#include <cstring>

#include "sparse_broadcast_layout.h"

namespace xllm {
namespace kernel {
namespace npu {

inline void sparse_bcast_gold_gather(const uint8_t* src,
                                     const int32_t* index,
                                     uint8_t* selected,
                                     int32_t n_rows,
                                     int32_t k,
                                     int32_t d,
                                     int32_t elem_size) {
  uint32_t row_bytes = sparse_bcast_row_bytes(static_cast<uint32_t>(d), static_cast<uint32_t>(elem_size));
  for (int32_t i = 0; i < k; ++i) {
    int32_t src_idx = index[i];
    uint8_t* dst_row = selected + static_cast<uint32_t>(i) * row_bytes;
    if (src_idx < 0 || src_idx >= n_rows) {
      std::memset(dst_row, 0, row_bytes);
    } else {
      std::memcpy(dst_row, src + static_cast<uint32_t>(src_idx) * row_bytes, row_bytes);
    }
  }
}

inline void sparse_bcast_gold_scatter(const uint8_t* src,
                                      const int32_t* index,
                                      uint8_t* dst,
                                      int32_t n_rows,
                                      int32_t k,
                                      int32_t d,
                                      int32_t elem_size) {
  uint32_t row_bytes = sparse_bcast_row_bytes(static_cast<uint32_t>(d), static_cast<uint32_t>(elem_size));
  for (int32_t i = 0; i < k; ++i) {
    int32_t src_idx = index[i];
    if (src_idx < 0 || src_idx >= n_rows) {
      continue;
    }
    uint8_t* dst_row = dst + static_cast<uint32_t>(src_idx) * row_bytes;
    std::memcpy(dst_row, src + static_cast<uint32_t>(src_idx) * row_bytes, row_bytes);
  }
}

}  // namespace npu
}  // namespace kernel
}  // namespace xllm
