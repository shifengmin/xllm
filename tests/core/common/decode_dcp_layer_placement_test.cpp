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

#include "core/common/decode_dcp_layer_placement.h"

#include <gtest/gtest.h>

namespace xllm {
namespace {

TEST(DecodeDcpLayerPlacementTest, SupportsOnlyNpuDsaModelFamilies) {
  EXPECT_TRUE(is_decode_dcp_layerwise_kv_cache_supported_model("deepseek_v32"));
  EXPECT_TRUE(is_decode_dcp_layerwise_kv_cache_supported_model("glm_moe_dsa"));
  EXPECT_FALSE(is_decode_dcp_layerwise_kv_cache_supported_model("deepseek_v3"));
  EXPECT_FALSE(is_decode_dcp_layerwise_kv_cache_supported_model("qwen3"));
}

TEST(DecodeDcpLayerPlacementTest, ResolvesTargetAndDraftFeatureFlags) {
  EXPECT_TRUE(resolve_decode_dcp_layerwise_kv_cache_enabled(
      /*configured=*/true, /*is_draft_model=*/false));
  EXPECT_FALSE(resolve_decode_dcp_layerwise_kv_cache_enabled(
      /*configured=*/true, /*is_draft_model=*/true));
  EXPECT_FALSE(resolve_decode_dcp_layerwise_kv_cache_enabled(
      /*configured=*/false, /*is_draft_model=*/false));
}

TEST(DecodeDcpLayerPlacementTest, DisabledConfigSkipsRoleValidation) {
  EXPECT_NO_FATAL_FAILURE(validate_decode_dcp_layerwise_kv_cache_config(
      /*enabled=*/false, InstanceRole::PREFILL, /*decode_dcp_size=*/1));
}

#if defined(USE_NPU)
TEST(DecodeDcpLayerPlacementTest, AcceptsEveryServingRoleAtStartup) {
  for (const InstanceRole role : {InstanceRole(InstanceRole::DECODE),
                                  InstanceRole(InstanceRole::PREFILL),
                                  InstanceRole(InstanceRole::DEFAULT),
                                  InstanceRole(InstanceRole::MIX)}) {
    EXPECT_NO_FATAL_FAILURE(validate_decode_dcp_layerwise_kv_cache_config(
        /*enabled=*/true, role, /*decode_dcp_size=*/2));
  }
}

TEST(DecodeDcpLayerPlacementTest, RejectsInvalidRoleAtStartup) {
  EXPECT_DEATH(validate_decode_dcp_layerwise_kv_cache_config(
                   /*enabled=*/true,
                   InstanceRole::INVALID,
                   /*decode_dcp_size=*/2),
               "unsupported instance role");
}

TEST(DecodeDcpLayerPlacementTest, RejectsSingleRankDcpAtStartup) {
  EXPECT_DEATH(validate_decode_dcp_layerwise_kv_cache_config(
                   /*enabled=*/true,
                   InstanceRole::DECODE,
                   /*decode_dcp_size=*/1),
               "requires decode_dcp_size > 1");
}
#else
TEST(DecodeDcpLayerPlacementTest, RejectsUnsupportedBackendAtStartup) {
  EXPECT_DEATH(validate_decode_dcp_layerwise_kv_cache_config(
                   /*enabled=*/true,
                   InstanceRole::DECODE,
                   /*decode_dcp_size=*/2),
               "only supported on the NPU backend");
}
#endif

}  // namespace
}  // namespace xllm
