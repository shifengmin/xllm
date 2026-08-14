/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/common/layerwise_split_placement.h"
#include "core/framework/parallel_state/parallel_args.h"

#include <gtest/gtest.h>

#include <vector>

namespace xllm {
namespace {

TEST(LayerwiseSplitPlacementTest, SupportsOnlyDsaModelFamilies) {
  EXPECT_TRUE(is_layerwise_split_supported_model("deepseek_v32"));
  EXPECT_TRUE(is_layerwise_split_supported_model("glm_moe_dsa"));
  EXPECT_FALSE(is_layerwise_split_supported_model("deepseek_v32_mtp"));
  EXPECT_FALSE(is_layerwise_split_supported_model("glm_moe_dsa_mtp"));
  EXPECT_FALSE(is_layerwise_split_supported_model("deepseek_v3"));
  EXPECT_FALSE(is_layerwise_split_supported_model("qwen3"));
}

TEST(LayerwiseSplitPlacementTest, ResolvesEffectiveLayerwiseSplitSize) {
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"deepseek_v32",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/8),
            2);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"glm_moe_dsa",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/8),
            2);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"deepseek_v32",
                /*is_draft_model=*/true,
                /*attn_tp_size=*/8),
            1);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/1,
                /*model_type=*/"deepseek_v32",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/8),
            1);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"qwen3",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/8),
            1);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"deepseek_v32_mtp",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/8),
            1);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"deepseek_v32",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/1),
            1);
  EXPECT_EQ(effective_layerwise_split_size(
                /*configured=*/2,
                /*model_type=*/"deepseek_v32",
                /*is_draft_model=*/false,
                /*attn_tp_size=*/3),
            1);
}

TEST(LayerwiseSplitPlacementTest, DisabledOwnsEveryLayer) {
  const LayerwiseSplitPlacement placement(
      /*enabled=*/false, /*group_size=*/2, /*local_rank=*/0);
  EXPECT_TRUE(placement.owns(/*layer_id=*/0));
  EXPECT_TRUE(placement.owns(/*layer_id=*/1));
  EXPECT_EQ(placement.owner_rank(/*layer_id=*/0), 0);
  EXPECT_EQ(placement.owner_rank(/*layer_id=*/1), 1);
}

TEST(LayerwiseSplitPlacementTest, OwnerRankRotatesByLayer) {
  const LayerwiseSplitPlacement placement(
      /*enabled=*/true, /*group_size=*/2, /*local_rank=*/1);
  EXPECT_EQ(placement.owner_rank(/*layer_id=*/0), 0);
  EXPECT_EQ(placement.owner_rank(/*layer_id=*/1), 1);
  EXPECT_EQ(placement.owner_rank(/*layer_id=*/2), 0);
  EXPECT_FALSE(placement.owns(/*layer_id=*/0));
  EXPECT_TRUE(placement.owns(/*layer_id=*/1));
  EXPECT_FALSE(placement.owns(/*layer_id=*/2));
}

TEST(LayerwiseSplitPlacementTest, DisabledConfigSkipsValidation) {
  EXPECT_NO_FATAL_FAILURE(
      validate_layerwise_split_size_config(/*layerwise_split_size=*/1));
}

TEST(LayerwiseSplitPlacementTest, RejectsNonPositiveSize) {
  EXPECT_DEATH(validate_layerwise_split_size_config(/*layerwise_split_size=*/0),
               "must be >= 1");
}

TEST(LayerwiseSplitPlacementTest, AcceptsEnabledMultiRankAtStartup) {
  EXPECT_NO_FATAL_FAILURE(
      validate_layerwise_split_size_config(/*layerwise_split_size=*/2));
}

TEST(LayerwiseSplitPlacementTest, DerivesLocalRankFromParallelArgs) {
  ParallelArgs rank_six(/*rank=*/6,
                        /*world_size=*/16,
                        /*dp_size=*/2,
                        /*cp_size=*/1,
                        /*process_group=*/nullptr,
                        /*ep_size=*/1);
  rank_six.layerwise_split_size(2);
  EXPECT_EQ(rank_six.attn_tp_size(), 8);
  EXPECT_EQ(rank_six.attn_tp_rank(), 6);
  EXPECT_EQ(rank_six.layerwise_split_rank(), 0);

  ParallelArgs rank_seven(/*rank=*/7,
                          /*world_size=*/16,
                          /*dp_size=*/2,
                          /*cp_size=*/1,
                          /*process_group=*/nullptr,
                          /*ep_size=*/1);
  rank_seven.layerwise_split_size(2);
  EXPECT_EQ(rank_seven.attn_tp_rank(), 7);
  EXPECT_EQ(rank_seven.layerwise_split_rank(), 1);
}

TEST(LayerwiseSplitPlacementTest, CollapsesLayerwiseSplitMappingData) {
  ParallelArgs args(/*rank=*/7,
                    /*world_size=*/16,
                    /*dp_size=*/2,
                    /*cp_size=*/1,
                    /*process_group=*/nullptr,
                    /*ep_size=*/1);
  nlohmann::json mapping_data;
  mapping_data["attnLayerwiseSplit"]["group_size"] = 2;
  mapping_data["attnLayerwiseSplit"]["rank"] = 1;
  mapping_data["attnLayerwiseSplit"]["rankIds"] = std::vector<uint32_t>{6, 7};
  args.mapping_data(mapping_data);

  args.collapse_layerwise_split_mapping();

  const nlohmann::json& split = args.mapping_data()["attnLayerwiseSplit"];
  EXPECT_EQ(split["group_size"].get<int32_t>(), 1);
  EXPECT_EQ(split["rank"].get<int32_t>(), 0);
  EXPECT_EQ(split["rankIds"].get<std::vector<uint32_t>>(),
            (std::vector<uint32_t>{7}));
}

}  // namespace
}  // namespace xllm
