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

#include "sparse_broadcast_layout.h"

namespace xllm {
namespace kernel {
namespace npu {

struct SparseBroadcastLaunchArgs {
  void* stream;
  uint64_t ffts_addr;
  uint8_t* src;
  int32_t* index;
  int32_t* index_out;
  uint8_t* dst;
  uint8_t* packed;
  int32_t* flag;
  int32_t n_rows;
  int32_t k;
  int32_t d;
  int32_t elem_size;
  int32_t root;
  int32_t seq;
  uint32_t block_dim;
};

void launch_sparse_broadcast(const SparseBroadcastLaunchArgs& args);

}  // namespace npu
}  // namespace kernel
}  // namespace xllm
