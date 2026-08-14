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

#include "operations/aclrt/ops/hccl_scatter_operation.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace atb_speed {
namespace common {
namespace {

atb::TensorDesc make_rank_major_desc(int64_t dcp_size,
                                     int64_t token_count,
                                     int64_t head_count,
                                     int64_t head_dim) {
  atb::TensorDesc desc;
  desc.format = ACL_FORMAT_ND;
  desc.dtype = ACL_BF16;
  desc.shape.dimNum = 4;
  desc.shape.dims[0] = dcp_size;
  desc.shape.dims[1] = token_count;
  desc.shape.dims[2] = head_count;
  desc.shape.dims[3] = head_dim;
  return desc;
}

void expect_local_attention_shape(int32_t dcp_size, int64_t token_count) {
  HcclScatterOperation operation("test_hccl_scatter", 0, dcp_size, 0, nullptr);
  atb::SVector<atb::TensorDesc> input_descs = {
      make_rank_major_desc(dcp_size, token_count, 8, 512)};
  atb::SVector<atb::TensorDesc> output_descs(1, atb::TensorDesc{});

  ASSERT_EQ(operation.InferShape(input_descs, output_descs), atb::NO_ERROR);
  EXPECT_EQ(output_descs[0].shape.dimNum, 3);
  EXPECT_EQ(output_descs[0].shape.dims[0], token_count);
  EXPECT_EQ(output_descs[0].shape.dims[1], 8);
  EXPECT_EQ(output_descs[0].shape.dims[2], 512);
  EXPECT_EQ(output_descs[0].dtype, ACL_BF16);
}

TEST(HcclScatterOperationTest, RemovesDcpRankDimensionForDcp2) {
  expect_local_attention_shape(2, 3);
}

TEST(HcclScatterOperationTest, RemovesDcpRankDimensionForDcp4) {
  expect_local_attention_shape(4, 7);
}

TEST(HcclScatterOperationTest, RejectsMismatchedRankDimension) {
  HcclScatterOperation operation("test_hccl_scatter", 0, 4, 0, nullptr);
  atb::SVector<atb::TensorDesc> input_descs = {
      make_rank_major_desc(2, 3, 8, 512)};
  atb::SVector<atb::TensorDesc> output_descs(1, atb::TensorDesc{});

  EXPECT_EQ(operation.InferShape(input_descs, output_descs),
            atb::ERROR_INVALID_PARAM);
}

}  // namespace
}  // namespace common
}  // namespace atb_speed
