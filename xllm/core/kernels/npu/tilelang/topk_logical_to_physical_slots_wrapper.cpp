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

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <limits>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_atb_ops_api.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_TOPK_LOGICAL_TO_PHYSICAL_SLOTS_REGISTRY_INC
#error "XLLM_TL_TOPK_LOGICAL_TO_PHYSICAL_SLOTS_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_TOPK_LOGICAL_TO_PHYSICAL_SLOTS_REGISTRY_INC

void check_int32_npu_contiguous(const torch::Tensor& tensor,
                                const char* tensor_name) {
  CHECK(tensor.defined()) << tensor_name << " must be defined";
  CHECK_EQ(tensor.device().type(), c10::DeviceType::PrivateUse1)
      << tensor_name << " must be on NPU";
  CHECK_EQ(tensor.dtype(), torch::kInt32)
      << tensor_name << " must have int32 dtype";
  CHECK(tensor.is_contiguous()) << tensor_name << " must be contiguous";
}

int32_t checked_int32(int64_t value, const char* value_name) {
  CHECK_GE(value, 0) << value_name << " must be non-negative";
  CHECK_LE(value, static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << value_name << " exceeds int32 range";
  return static_cast<int32_t>(value);
}

}  // namespace

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
                                           aclrtStream stream) {
  CHECK(topk_positions != nullptr);
  CHECK(block_tables != nullptr);
  CHECK(packed_gather_indices != nullptr);
  CHECK(packed_query_block_rows != nullptr);
  CHECK(physical_slots != nullptr);
  CHECK(stream != nullptr);
  CHECK_GT(topk_numel, 0);
  CHECK_GT(block_table_rows, 0);
  CHECK_GT(block_table_cols, 0);
  CHECK_GT(packed_count, 0);
  CHECK_GE(topk_numel, packed_count);
  CHECK_LE(topk_numel,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(block_table_rows,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(block_table_cols,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(packed_count,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  const TopkLogicalToPhysicalSlotsSpecialization specialization =
      make_topk_logical_to_physical_slots_specialization(
          TopkLogicalToPhysicalSlotsBlockSize{block_size});
  const auto* entry =
      find_topk_logical_to_physical_slots_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang topk logical-to-physical mapping has no specialization for "
      << "block_size=" << block_size << ". Available variants: "
      << available_topk_logical_to_physical_slots_variant_keys();

  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(topk_positions)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(block_tables)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(packed_gather_indices)),
      reinterpret_cast<uint8_t*>(const_cast<void*>(packed_query_block_rows)),
      reinterpret_cast<uint8_t*>(physical_slots),
      static_cast<int64_t>(topk_numel),
      static_cast<int64_t>(block_table_rows),
      static_cast<int64_t>(block_table_cols),
      static_cast<int64_t>(packed_count),
      stream);
}

torch::Tensor topk_logical_to_physical_slots(
    const torch::Tensor& topk_positions,
    const torch::Tensor& block_tables,
    const torch::Tensor& packed_gather_indices,
    const torch::Tensor& packed_query_block_rows,
    int64_t block_size) {
  check_int32_npu_contiguous(topk_positions, "topk_positions");
  check_int32_npu_contiguous(block_tables, "block_tables");
  check_int32_npu_contiguous(packed_gather_indices, "packed_gather_indices");
  check_int32_npu_contiguous(packed_query_block_rows,
                             "packed_query_block_rows");
  CHECK_EQ(block_tables.dim(), 2) << "block_tables must be 2D";
  CHECK_EQ(packed_gather_indices.dim(), 1)
      << "packed_gather_indices must be 1D";
  CHECK_EQ(packed_query_block_rows.dim(), 1)
      << "packed_query_block_rows must be 1D";
  CHECK_EQ(packed_gather_indices.numel(), packed_query_block_rows.numel())
      << "packed metadata tensors must have equal lengths";
  CHECK_EQ(topk_positions.device(), block_tables.device());
  CHECK_EQ(topk_positions.device(), packed_gather_indices.device());
  CHECK_EQ(topk_positions.device(), packed_query_block_rows.device());
  CHECK_GT(topk_positions.numel(), 0);
  CHECK_GT(block_tables.size(0), 0);
  CHECK_GT(block_tables.size(1), 0);
  CHECK_GT(packed_gather_indices.numel(), 0);
  CHECK_GE(topk_positions.numel(), packed_gather_indices.numel());

  const int32_t block_size_i32 = checked_int32(block_size, "block_size");
  CHECK_GT(block_size_i32, 0);
  torch::Tensor physical_slots = torch::empty_like(packed_gather_indices);
  const int32_t device_id = topk_positions.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  launch_topk_logical_to_physical_slots(topk_positions.data_ptr(),
                                        block_tables.data_ptr(),
                                        packed_gather_indices.data_ptr(),
                                        packed_query_block_rows.data_ptr(),
                                        physical_slots.data_ptr(),
                                        topk_positions.numel(),
                                        block_tables.size(0),
                                        block_tables.size(1),
                                        packed_gather_indices.numel(),
                                        block_size_i32,
                                        stream);
  return physical_slots;
}

}  // namespace xllm::kernel::npu::tilelang
