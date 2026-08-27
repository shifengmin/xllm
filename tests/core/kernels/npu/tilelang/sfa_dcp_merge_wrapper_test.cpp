/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class SfaDcpMergeWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

torch::Tensor merge_reference(const torch::Tensor& output_recv,
                              const torch::Tensor& lse_recv) {
  auto lse = lse_recv.to(torch::kFloat32);
  lse = lse.masked_fill(~torch::isfinite(lse),
                        -std::numeric_limits<float>::infinity());
  auto weights = torch::softmax(lse, /*dim=*/0);
  weights = torch::nan_to_num(weights, /*nan=*/0.0);
  auto merged = (output_recv.to(lse.dtype()) * weights.unsqueeze(-1)).sum(0);
  return merged.movedim(1, 0).contiguous().to(output_recv.dtype());
}

TEST_F(SfaDcpMergeWrapperTest, MatchesNaiveSoftmaxMerge) {
  ASSERT_TRUE(has_sfa_dcp_merge_specialization(4, torch::kBFloat16));
  const auto npu = torch::Device("npu:0");
  torch::manual_seed(5);
  const int64_t dcp_size = 4;
  const int64_t num_heads = 8;
  const int64_t num_tokens = 8;
  auto output_recv =
      torch::randn({dcp_size, num_heads, num_tokens, 512},
                   torch::TensorOptions().dtype(torch::kBFloat16).device(npu));
  auto lse_recv =
      torch::randn({dcp_size, num_heads, num_tokens},
                   torch::TensorOptions().dtype(torch::kFloat32).device(npu));
  lse_recv[0][0][0] = -std::numeric_limits<float>::infinity();
  auto out = torch::empty({num_tokens, num_heads, 512}, output_recv.options());
  sfa_dcp_merge_out(output_recv, lse_recv, out);
  auto ref = merge_reference(output_recv, lse_recv);
  EXPECT_TRUE(torch::allclose(out, ref, /*rtol=*/1.56e-2, /*atol=*/9.77e-4));
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
