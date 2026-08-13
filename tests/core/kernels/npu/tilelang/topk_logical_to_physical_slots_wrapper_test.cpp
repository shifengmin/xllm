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

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>
#include <memory>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#include "operations/tilelang/topk_logical_to_physical_slots_operation.h"
#include "pytorch/adapter/utils/utils.h"

namespace xllm::kernel::npu::tilelang {
namespace {

class TopkLogicalToPhysicalSlotsWrapperTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { torch_npu::init_npu("npu:0"); }

  static void TearDownTestSuite() { torch_npu::finalize_npu(); }
};

TEST_F(TopkLogicalToPhysicalSlotsWrapperTest,
       MapsCompactedMultiQueryTopkInOriginalOrder) {
  constexpr int64_t kBlockSize = 128;
  torch::Tensor topk_cpu =
      torch::tensor({{{0, 127, 128, 255, 256, 383, 384, 511}},
                     {{511, 384, 257, 129, 128, 127, 1, 0}},
                     {{64, 192, 320, 448, 65, 193, 321, 449}}},
                    torch::kInt32);
  torch::Tensor block_tables_cpu = torch::tensor(
      {{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}}, torch::kInt32);
  torch::Tensor packed_indices_cpu =
      torch::tensor({0, 1, 2, 8, 9, 10, 11, 16, 17, 18, 19, 20}, torch::kInt32);
  torch::Tensor packed_rows_cpu =
      torch::tensor({0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2}, torch::kInt32);

  torch::Tensor result =
      topk_logical_to_physical_slots(topk_cpu.to("npu:0"),
                                     block_tables_cpu.to("npu:0"),
                                     packed_indices_cpu.to("npu:0"),
                                     packed_rows_cpu.to("npu:0"),
                                     kBlockSize);

  torch::Tensor compact_topk =
      topk_cpu.view({-1}).index_select(0, packed_indices_cpu.to(torch::kInt64));
  torch::Tensor logical_blocks = compact_topk / kBlockSize;
  torch::Tensor block_offsets = compact_topk % kBlockSize;
  torch::Tensor physical_blocks = block_tables_cpu.index(
      {packed_rows_cpu.to(torch::kInt64), logical_blocks.to(torch::kInt64)});
  torch::Tensor expected = physical_blocks * kBlockSize + block_offsets;

  EXPECT_TRUE(torch::equal(result.cpu(), expected))
      << "result=" << result.cpu() << ", expected=" << expected;
}

TEST_F(TopkLogicalToPhysicalSlotsWrapperTest, SupportsSingleDummyEntry) {
  constexpr int64_t kBlockSize = 16;
  torch::Tensor result = topk_logical_to_physical_slots(
      torch::tensor({15}, torch::kInt32).to("npu:0"),
      torch::tensor({{7}}, torch::kInt32).to("npu:0"),
      torch::tensor({0}, torch::kInt32).to("npu:0"),
      torch::tensor({0}, torch::kInt32).to("npu:0"),
      kBlockSize);

  EXPECT_EQ(result.cpu().item<int32_t>(), 127);
}

TEST_F(TopkLogicalToPhysicalSlotsWrapperTest,
       PreservesSlotsBeyondFloat32ExactIntegerRange) {
  constexpr int64_t kBlockSize = 128;
  constexpr int32_t kPhysicalBlock = 131072;
  constexpr int32_t kExpectedSlot = 16777217;
  torch::Tensor result = topk_logical_to_physical_slots(
      torch::tensor({1}, torch::kInt32).to("npu:0"),
      torch::tensor({{kPhysicalBlock}}, torch::kInt32).to("npu:0"),
      torch::tensor({0}, torch::kInt32).to("npu:0"),
      torch::tensor({0}, torch::kInt32).to("npu:0"),
      kBlockSize);

  EXPECT_EQ(result.cpu().item<int32_t>(), kExpectedSlot);
}

TEST_F(TopkLogicalToPhysicalSlotsWrapperTest,
       AtbOperationAcceptsGraphWorkspaceArguments) {
  constexpr int64_t kBlockSize = 128;
  torch::Tensor topk = torch::tensor({129}, torch::kInt32).to("npu:0");
  torch::Tensor block_tables =
      torch::tensor({{3, 5}}, torch::kInt32).to("npu:0");
  torch::Tensor packed_indices = torch::tensor({0}, torch::kInt32).to("npu:0");
  torch::Tensor packed_rows = torch::tensor({0}, torch::kInt32).to("npu:0");
  torch::Tensor result = torch::empty_like(packed_indices);

  atb::VariantPack variant_pack;
  variant_pack.inTensors = {
      atb_speed::Utils::AtTensor2Tensor(topk),
      atb_speed::Utils::AtTensor2Tensor(block_tables),
      atb_speed::Utils::AtTensor2Tensor(packed_indices),
      atb_speed::Utils::AtTensor2Tensor(packed_rows),
  };
  variant_pack.outTensors = {atb_speed::Utils::AtTensor2Tensor(result)};

  atb::Context* raw_context = nullptr;
  ASSERT_EQ(atb::CreateContext(&raw_context), atb::NO_ERROR);
  std::unique_ptr<atb::Context, decltype(&atb::DestroyContext)> context(
      raw_context, &atb::DestroyContext);
  aclrtStream stream = c10_npu::getCurrentNPUStream(0).stream();
  ASSERT_EQ(context->SetExecuteStream(stream), atb::NO_ERROR);

  atb_speed::TopkLogicalToPhysicalSlotsOperation operation(kBlockSize);
  uint64_t requested_workspace_size = 1;
  ASSERT_EQ(
      operation.Setup(variant_pack, requested_workspace_size, context.get()),
      atb::NO_ERROR);
  EXPECT_EQ(requested_workspace_size, 0);

  uint8_t graph_workspace = 0;
  EXPECT_EQ(operation.Execute(variant_pack,
                              &graph_workspace,
                              /*workspace_size=*/4096,
                              context.get()),
            atb::NO_ERROR);
  EXPECT_EQ(result.cpu().item<int32_t>(), 641);
}

TEST_F(TopkLogicalToPhysicalSlotsWrapperTest,
       AtbOperationRejectsInvalidTensorCount) {
  constexpr int64_t kBlockSize = 128;
  atb::VariantPack variant_pack;
  atb::Context* raw_context = nullptr;
  ASSERT_EQ(atb::CreateContext(&raw_context), atb::NO_ERROR);
  std::unique_ptr<atb::Context, decltype(&atb::DestroyContext)> context(
      raw_context, &atb::DestroyContext);
  aclrtStream stream = c10_npu::getCurrentNPUStream(0).stream();
  ASSERT_EQ(context->SetExecuteStream(stream), atb::NO_ERROR);

  atb_speed::TopkLogicalToPhysicalSlotsOperation operation(kBlockSize);
  EXPECT_EQ(operation.Execute(variant_pack,
                              /*workspace=*/nullptr,
                              /*workspace_size=*/0,
                              context.get()),
            atb::ERROR_INVALID_PARAM);
}

}  // namespace
}  // namespace xllm::kernel::npu::tilelang
