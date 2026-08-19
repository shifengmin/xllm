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
namespace kernel {
namespace npu {

constexpr uint32_t kSparseBcastAlign = 32;
constexpr uint32_t kSparseBcastHeaderBytes = 64;
constexpr uint32_t kSparseBcastBlockDim = 16;
constexpr uint32_t kSparseBcastDepth = 4;
constexpr uint32_t kSparseBcastChunkBytes = 40960;
constexpr uint32_t kSparseBcastMagic = 0x53424631u;
constexpr uint32_t kSparseBcastElemSize = 2;
constexpr uint32_t kSparseBcastUbBase = 1024;
constexpr uint32_t kSparseBcastPackDepth = 4;

inline uint32_t sparse_bcast_align32(uint32_t nbytes) {
  return (nbytes + (kSparseBcastAlign - 1u)) & ~(kSparseBcastAlign - 1u);
}

inline uint32_t sparse_bcast_index_offset() {
  return kSparseBcastHeaderBytes;
}

inline uint32_t sparse_bcast_index_bytes(uint32_t k) {
  return sparse_bcast_align32(k * static_cast<uint32_t>(sizeof(int32_t)));
}

inline uint32_t sparse_bcast_selected_offset(uint32_t k) {
  return kSparseBcastHeaderBytes + sparse_bcast_index_bytes(k);
}

inline uint32_t sparse_bcast_row_bytes(uint32_t d, uint32_t elem_size) {
  return d * elem_size;
}

inline uint32_t sparse_bcast_row_stride(uint32_t d, uint32_t elem_size) {
  return sparse_bcast_align32(sparse_bcast_row_bytes(d, elem_size));
}

inline uint32_t sparse_bcast_packed_bytes(uint32_t k, uint32_t d, uint32_t elem_size) {
  return sparse_bcast_selected_offset(k) + k * sparse_bcast_row_stride(d, elem_size);
}

inline uint32_t sparse_bcast_selected_bytes(uint32_t k, uint32_t d, uint32_t elem_size) {
  return k * sparse_bcast_row_bytes(d, elem_size);
}

inline uint32_t sparse_bcast_dst_bytes(uint32_t n_rows, uint32_t d, uint32_t elem_size) {
  return n_rows * sparse_bcast_row_bytes(d, elem_size);
}

inline uint32_t sparse_bcast_chunk_rows(uint32_t row_stride) {
  if (row_stride == 0 || row_stride > kSparseBcastChunkBytes) {
    return 1;
  }
  return kSparseBcastChunkBytes / row_stride;
}

inline void sparse_bcast_split_core_bytes(uint32_t total_bytes,
                                          uint32_t ncores,
                                          uint32_t core,
                                          uint32_t* off,
                                          uint32_t* len) {
  if (ncores == 0 || core >= ncores || total_bytes == 0) {
    *off = 0;
    *len = 0;
    return;
  }
  uint32_t units = (total_bytes + kSparseBcastAlign - 1u) / kSparseBcastAlign;
  uint32_t units_per_core = (units + ncores - 1u) / ncores;
  uint32_t core_off = core * units_per_core * kSparseBcastAlign;
  if (core_off >= total_bytes) {
    *off = 0;
    *len = 0;
    return;
  }
  uint32_t core_len = units_per_core * kSparseBcastAlign;
  if (core_off + core_len > total_bytes) {
    core_len = total_bytes - core_off;
  }
  *off = core_off;
  *len = core_len;
}

inline void sparse_bcast_split_core_rows(uint32_t k,
                                         uint32_t ncores,
                                         uint32_t core,
                                         uint32_t* row_begin,
                                         uint32_t* row_count) {
  if (ncores == 0 || core >= ncores || k == 0) {
    *row_begin = 0;
    *row_count = 0;
    return;
  }
  uint32_t rows_per_core = (k + ncores - 1u) / ncores;
  uint32_t begin = core * rows_per_core;
  if (begin >= k) {
    *row_begin = 0;
    *row_count = 0;
    return;
  }
  uint32_t count = rows_per_core;
  if (begin + count > k) {
    count = k - begin;
  }
  *row_begin = begin;
  *row_count = count;
}

}  // namespace npu
}  // namespace kernel
}  // namespace xllm
