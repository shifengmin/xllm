/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <limits>

#include "acl/acl.h"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_SFA_DCP_REMAP_REGISTRY_INC
#error "XLLM_TL_SFA_DCP_REMAP_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kMaxTokens = 256;
constexpr int64_t kMaxTopk = 2048;
constexpr int64_t kRemapVecLen = 64;
constexpr int64_t kDefaultPhysicalBlockSize = 128;

#include XLLM_TL_SFA_DCP_REMAP_REGISTRY_INC

int64_t remap_scratch_numel(int64_t num_tokens, int64_t topk) {
  const int64_t width = topk <= kRemapVecLen ? kRemapVecLen : topk;
  return num_tokens * width;
}

SfaDcpRemapSpecialization build_runtime_specialization(
    int64_t topk,
    int64_t physical_block_size,
    int64_t shard_size) {
  CHECK_LE(topk, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(physical_block_size,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(shard_size,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  return make_sfa_dcp_remap_specialization(
      SfaDcpRemapTopk{static_cast<int32_t>(topk)},
      SfaDcpRemapPhysicalBlockSize{static_cast<int32_t>(physical_block_size)},
      SfaDcpRemapShardSize{static_cast<int32_t>(shard_size)});
}

}  // namespace

bool has_sfa_dcp_remap_specialization(int64_t topk,
                                      int64_t physical_block_size,
                                      int64_t shard_size) {
  if (topk <= 0 || physical_block_size <= 0 || shard_size <= 1) {
    return false;
  }
  const SfaDcpRemapSpecialization specialization =
      build_runtime_specialization(topk, physical_block_size, shard_size);
  return find_sfa_dcp_remap_kernel_entry(specialization) != nullptr;
}

void sfa_dcp_remap_out(const torch::Tensor& topk_indices,
                       int64_t physical_block_size,
                       int64_t shard_size,
                       int64_t shard_rank,
                       torch::Tensor& out,
                       torch::Tensor& idx_scratch) {
  CHECK(topk_indices.defined())
      << "TileLang sfa_dcp_remap: topk_indices must be defined";
  CHECK(out.defined()) << "TileLang sfa_dcp_remap: out must be defined";
  CHECK(idx_scratch.defined())
      << "TileLang sfa_dcp_remap: idx_scratch must be defined";

  CHECK(topk_indices.device().type() == c10::DeviceType::PrivateUse1 &&
        out.device() == topk_indices.device() &&
        idx_scratch.device() == topk_indices.device())
      << "TileLang sfa_dcp_remap: all tensors must be on NPU";

  CHECK_EQ(topk_indices.scalar_type(), torch::kInt32)
      << "TileLang sfa_dcp_remap: topk_indices must be int32";
  CHECK_EQ(out.scalar_type(), torch::kInt32)
      << "TileLang sfa_dcp_remap: out must be int32";
  CHECK_EQ(idx_scratch.scalar_type(), torch::kInt32)
      << "TileLang sfa_dcp_remap: idx_scratch must be int32";

  CHECK(topk_indices.is_contiguous())
      << "TileLang sfa_dcp_remap: topk_indices must be contiguous";
  CHECK(out.is_contiguous())
      << "TileLang sfa_dcp_remap: out must be contiguous";
  CHECK(idx_scratch.is_contiguous())
      << "TileLang sfa_dcp_remap: idx_scratch must be contiguous";

  CHECK_GE(topk_indices.dim(), 2) << "TileLang sfa_dcp_remap: topk_indices "
                                     "rank must be >= 2, last dim = topk";
  CHECK_EQ(out.sizes(), topk_indices.sizes())
      << "TileLang sfa_dcp_remap: out shape must match topk_indices";

  const int64_t topk = topk_indices.size(-1);
  CHECK_GT(topk, 0) << "TileLang sfa_dcp_remap: topk must be > 0";
  CHECK_EQ(topk_indices.numel() % topk, 0)
      << "TileLang sfa_dcp_remap: numel must be divisible by topk";
  const int64_t num_tokens = topk_indices.numel() / topk;
  CHECK_GT(num_tokens, 0) << "TileLang sfa_dcp_remap: T must be > 0";
  CHECK_LE(num_tokens, kMaxTokens)
      << "TileLang sfa_dcp_remap: T must be <= " << kMaxTokens;
  CHECK_LE(topk, kMaxTopk) << "TileLang sfa_dcp_remap: topk must be <= "
                           << kMaxTopk;
  CHECK_EQ(physical_block_size, kDefaultPhysicalBlockSize)
      << "TileLang sfa_dcp_remap: physical_block_size must be "
      << kDefaultPhysicalBlockSize;
  CHECK_GT(shard_size, 1) << "TileLang sfa_dcp_remap: shard_size must be > 1";
  CHECK_GE(shard_rank, 0) << "TileLang sfa_dcp_remap: shard_rank must be >= 0";
  CHECK_LT(shard_rank, shard_size)
      << "TileLang sfa_dcp_remap: shard_rank must be < shard_size";
  CHECK_LE(num_tokens,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(shard_rank,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  const int64_t scratch_n = remap_scratch_numel(num_tokens, topk);
  CHECK_GE(idx_scratch.numel(), scratch_n)
      << "TileLang sfa_dcp_remap: idx_scratch numel must be >= " << scratch_n;

  const SfaDcpRemapSpecialization specialization =
      build_runtime_specialization(topk, physical_block_size, shard_size);
  const auto* entry = find_sfa_dcp_remap_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang sfa_dcp_remap: no compiled variant. Available variants: "
      << available_sfa_dcp_remap_variant_keys();

  const int32_t device_id = topk_indices.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  uint8_t* scratch_ptr = reinterpret_cast<uint8_t*>(idx_scratch.data_ptr());
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(topk_indices.data_ptr())),
      reinterpret_cast<uint8_t*>(out.data_ptr()),
      scratch_ptr,
      scratch_ptr,
      static_cast<int32_t>(num_tokens),
      static_cast<int32_t>(shard_rank),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
