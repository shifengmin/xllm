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

#ifndef XLLM_TL_SFA_DCP_MERGE_REGISTRY_INC
#error "XLLM_TL_SFA_DCP_MERGE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

constexpr int64_t kMaxTokens = 256;
constexpr int64_t kMergeMaxHeads = 256;
constexpr int64_t kHeadDim = 512;

#include XLLM_TL_SFA_DCP_MERGE_REGISTRY_INC

SfaDcpMergeSpecialization build_runtime_specialization(int64_t dcp_size,
                                                       c10::ScalarType dtype) {
  CHECK_LE(dcp_size, static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  return make_sfa_dcp_merge_specialization(
      SfaDcpMergeDcpSize{static_cast<int32_t>(dcp_size)},
      SfaDcpMergeDType{to_tilelang_dtype(dtype)});
}

}  // namespace

bool has_sfa_dcp_merge_specialization(int64_t dcp_size, c10::ScalarType dtype) {
  if (dcp_size <= 1) {
    return false;
  }
  const SfaDcpMergeSpecialization specialization =
      build_runtime_specialization(dcp_size, dtype);
  return find_sfa_dcp_merge_kernel_entry(specialization) != nullptr;
}

void sfa_dcp_merge_out(const torch::Tensor& output_recv,
                       const torch::Tensor& lse_recv,
                       torch::Tensor& out) {
  CHECK(output_recv.defined())
      << "TileLang sfa_dcp_merge: output_recv must be defined";
  CHECK(lse_recv.defined())
      << "TileLang sfa_dcp_merge: lse_recv must be defined";
  CHECK(out.defined()) << "TileLang sfa_dcp_merge: out must be defined";

  CHECK(output_recv.device().type() == c10::DeviceType::PrivateUse1 &&
        lse_recv.device() == output_recv.device() &&
        out.device() == output_recv.device())
      << "TileLang sfa_dcp_merge: all tensors must be on NPU";

  CHECK_EQ(output_recv.dim(), 4)
      << "TileLang sfa_dcp_merge: output_recv must be 4D [dcp, H, T, D]";
  CHECK_EQ(lse_recv.dim(), 3)
      << "TileLang sfa_dcp_merge: lse_recv must be 3D [dcp, H, T]";
  CHECK_EQ(output_recv.size(0), lse_recv.size(0));
  CHECK_EQ(output_recv.size(1), lse_recv.size(1));
  CHECK_EQ(output_recv.size(2), lse_recv.size(2));
  CHECK_EQ(output_recv.size(3), kHeadDim)
      << "TileLang sfa_dcp_merge: head_dim must be " << kHeadDim;

  CHECK_EQ(output_recv.scalar_type(), torch::kBFloat16)
      << "TileLang sfa_dcp_merge: output_recv must be bfloat16";
  CHECK_EQ(lse_recv.scalar_type(), torch::kFloat32)
      << "TileLang sfa_dcp_merge: lse_recv must be float32";
  CHECK_EQ(out.scalar_type(), output_recv.scalar_type())
      << "TileLang sfa_dcp_merge: out dtype must match output_recv";

  CHECK(output_recv.is_contiguous())
      << "TileLang sfa_dcp_merge: output_recv must be contiguous";
  CHECK(lse_recv.is_contiguous())
      << "TileLang sfa_dcp_merge: lse_recv must be contiguous";
  CHECK(out.is_contiguous())
      << "TileLang sfa_dcp_merge: out must be contiguous";

  const int64_t dcp_size = output_recv.size(0);
  const int64_t num_heads = output_recv.size(1);
  const int64_t num_tokens = output_recv.size(2);
  CHECK_GT(dcp_size, 1) << "TileLang sfa_dcp_merge: dcp_size must be > 1";
  CHECK_GT(num_heads, 0) << "TileLang sfa_dcp_merge: H must be > 0";
  CHECK_LE(num_heads, kMergeMaxHeads)
      << "TileLang sfa_dcp_merge: H must be <= " << kMergeMaxHeads;
  CHECK_GT(num_tokens, 0) << "TileLang sfa_dcp_merge: T must be > 0";
  CHECK_LE(num_tokens, kMaxTokens)
      << "TileLang sfa_dcp_merge: T must be <= " << kMaxTokens;
  CHECK_EQ(out.dim(), 3);
  CHECK_EQ(out.size(0), num_tokens);
  CHECK_EQ(out.size(1), num_heads);
  CHECK_EQ(out.size(2), kHeadDim);
  CHECK_LE(num_tokens,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));
  CHECK_LE(num_heads,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()));

  const SfaDcpMergeSpecialization specialization =
      build_runtime_specialization(dcp_size, output_recv.scalar_type());
  const auto* entry = find_sfa_dcp_merge_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << "TileLang sfa_dcp_merge: no compiled variant. Available variants: "
      << available_sfa_dcp_merge_variant_keys();

  const int32_t device_id = output_recv.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  entry->fn(
      reinterpret_cast<uint8_t*>(const_cast<void*>(output_recv.data_ptr())),
      reinterpret_cast<uint8_t*>(const_cast<void*>(lse_recv.data_ptr())),
      reinterpret_cast<uint8_t*>(out.data_ptr()),
      static_cast<int32_t>(num_tokens),
      static_cast<int32_t>(num_heads),
      stream);
}

}  // namespace xllm::kernel::npu::tilelang
