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

#include <acl/acl.h>

#include <cstdint>

namespace xllm::kernel::npu::tilelang {

void launch_topk_logical_to_physical_slots(const void* topk_positions,
                                           const void* block_tables,
                                           const void* packed_gather_indices,
                                           const void* packed_query_block_rows,
                                           void* physical_slots,
                                           int64_t topk_numel,
                                           int64_t block_table_rows,
                                           int64_t block_table_cols,
                                           int64_t packed_count,
                                           int32_t block_size,
                                           aclrtStream stream);

}  // namespace xllm::kernel::npu::tilelang
