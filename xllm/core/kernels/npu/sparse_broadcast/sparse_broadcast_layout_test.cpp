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

#include "sparse_broadcast_gold.h"
#include "sparse_broadcast_layout.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using xllm::kernel::npu::kSparseBcastAlign;
using xllm::kernel::npu::kSparseBcastBlockDim;
using xllm::kernel::npu::kSparseBcastChunkBytes;
using xllm::kernel::npu::kSparseBcastDepth;
using xllm::kernel::npu::kSparseBcastHeaderBytes;
using xllm::kernel::npu::sparse_bcast_chunk_rows;
using xllm::kernel::npu::sparse_bcast_gold_gather;
using xllm::kernel::npu::sparse_bcast_gold_scatter;
using xllm::kernel::npu::sparse_bcast_index_bytes;
using xllm::kernel::npu::sparse_bcast_index_offset;
using xllm::kernel::npu::sparse_bcast_packed_bytes;
using xllm::kernel::npu::sparse_bcast_row_bytes;
using xllm::kernel::npu::sparse_bcast_row_stride;
using xllm::kernel::npu::sparse_bcast_selected_bytes;
using xllm::kernel::npu::sparse_bcast_selected_offset;
using xllm::kernel::npu::sparse_bcast_split_core_bytes;
using xllm::kernel::npu::sparse_bcast_split_core_rows;

int32_t g_failed = 0;

void expect_true(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++g_failed;
  }
}

void expect_eq_u32(uint32_t got, uint32_t want, const char* msg) {
  if (got != want) {
    std::fprintf(stderr, "FAIL: %s got=%u want=%u\n", msg, got, want);
    ++g_failed;
  }
}

void test_packed_layout_p0() {
  constexpr uint32_t k = 2048;
  constexpr uint32_t d = 576;
  constexpr uint32_t elem = 2;
  expect_eq_u32(sparse_bcast_row_bytes(d, elem), 1152, "row bytes");
  expect_eq_u32(sparse_bcast_index_offset(), kSparseBcastHeaderBytes, "index offset");
  expect_eq_u32(sparse_bcast_index_bytes(k), 8192, "index bytes aligned");
  expect_eq_u32(sparse_bcast_selected_offset(k), 8256, "selected offset");
  expect_eq_u32(sparse_bcast_selected_bytes(k, d, elem), 2359296, "selected bytes");
  expect_eq_u32(sparse_bcast_packed_bytes(k, d, elem), 2367552, "packed bytes");
  expect_true(sparse_bcast_row_bytes(d, elem) % kSparseBcastAlign == 0, "row 32B aligned");
}

void test_tiling_covers_payload() {
  constexpr uint32_t k = 2048;
  constexpr uint32_t d = 576;
  constexpr uint32_t elem = 2;
  uint32_t row_stride = sparse_bcast_row_stride(d, elem);
  uint32_t chunk_rows = sparse_bcast_chunk_rows(row_stride);
  expect_eq_u32(chunk_rows, kSparseBcastChunkBytes / 1152, "chunk rows");
  uint32_t covered = 0;
  for (uint32_t core = 0; core < kSparseBcastBlockDim; ++core) {
    uint32_t begin = 0;
    uint32_t count = 0;
    sparse_bcast_split_core_rows(k, kSparseBcastBlockDim, core, &begin, &count);
    expect_eq_u32(begin, core * 128, "core row begin");
    expect_eq_u32(count, 128, "core row count");
    uint32_t nchunk = (count + chunk_rows - 1u) / chunk_rows;
    expect_eq_u32(nchunk, kSparseBcastDepth, "prologue covers core rows");
    covered += count;
  }
  expect_eq_u32(covered, k, "16 cores cover K rows");
}

void test_small_split_and_empty_core() {
  uint32_t off = 1;
  uint32_t len = 1;
  sparse_bcast_split_core_bytes(/*total_bytes=*/0, kSparseBcastBlockDim, 0, &off, &len);
  expect_eq_u32(off, 0, "empty total off");
  expect_eq_u32(len, 0, "empty total len");

  uint32_t begin = 1;
  uint32_t count = 1;
  sparse_bcast_split_core_rows(/*k=*/32, kSparseBcastBlockDim, 15, &begin, &count);
  expect_true(begin + count <= 32, "row split in range");
}

void test_gold_scatter_keeps_unselected() {
  constexpr int32_t n_rows = 4;
  constexpr int32_t k = 4;
  constexpr int32_t d = 16;
  constexpr int32_t elem = 2;
  uint32_t row_bytes = sparse_bcast_row_bytes(d, elem);
  std::vector<uint8_t> src(static_cast<size_t>(n_rows) * row_bytes);
  std::vector<int32_t> index = {1, -1, 3, 99};
  std::vector<uint8_t> dst(static_cast<size_t>(n_rows) * row_bytes, 0x3c);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i + 1);
  }
  sparse_bcast_gold_scatter(src.data(), index.data(), dst.data(), n_rows, k, d, elem);

  expect_true(std::memcmp(dst.data() + row_bytes, src.data() + row_bytes, row_bytes) == 0, "dst[1] from src[1]");
  expect_true(std::memcmp(dst.data() + 3 * row_bytes, src.data() + 3 * row_bytes, row_bytes) == 0,
              "dst[3] from src[3]");
  for (uint32_t i = 0; i < row_bytes; ++i) {
    expect_true(dst[i] == 0x3c, "unselected dst[0] unchanged");
    expect_true(dst[2 * row_bytes + i] == 0x3c, "unselected dst[2] unchanged");
  }
}

void test_gold_gather_padding() {
  constexpr int32_t n_rows = 4;
  constexpr int32_t k = 4;
  constexpr int32_t d = 16;
  constexpr int32_t elem = 2;
  uint32_t row_bytes = sparse_bcast_row_bytes(d, elem);
  std::vector<uint8_t> src(static_cast<size_t>(n_rows) * row_bytes);
  std::vector<int32_t> index = {1, -1, 3, 99};
  std::vector<uint8_t> selected(static_cast<size_t>(k) * row_bytes, 0x5a);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i + 1);
  }
  sparse_bcast_gold_gather(src.data(), index.data(), selected.data(), n_rows, k, d, elem);

  expect_true(std::memcmp(selected.data(), src.data() + row_bytes, row_bytes) == 0, "row0 from src[1]");
  for (uint32_t i = 0; i < row_bytes; ++i) {
    expect_true(selected[row_bytes + i] == 0, "negative index -> zero");
    expect_true(selected[3 * row_bytes + i] == 0, "oob index -> zero");
  }
  expect_true(std::memcmp(selected.data() + 2 * row_bytes, src.data() + 3 * row_bytes, row_bytes) == 0,
              "row2 from src[3]");
}

}  // namespace

int main() {
  test_packed_layout_p0();
  test_tiling_covers_payload();
  test_small_split_and_empty_core();
  test_gold_gather_padding();
  test_gold_scatter_keeps_unselected();
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return 1;
  }
  std::printf("sparse_broadcast_layout_test passed\n");
  return 0;
}
