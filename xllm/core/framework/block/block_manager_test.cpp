/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "block_manager_impl.h"

namespace xllm {

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);
  EXPECT_EQ(manager.block_size(), block_size);

  // Allocate a block
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.id(), 1);
    EXPECT_EQ(block.size(), block_size);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(manager.num_free_blocks(), n_blocks - 2);
  }
  // the block should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Allocate a list of blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 1; i < n_blocks; ++i) {
      auto block = manager.allocate();
      EXPECT_EQ(block.id(), i);
      EXPECT_EQ(block.size(), block_size);
      EXPECT_EQ(block.is_shared(), false);
      EXPECT_EQ(block.ref_count(), 1);
      blocks.push_back(std::move(block));
    }
    EXPECT_EQ(manager.num_free_blocks(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_DEATH(manager.allocate(), "No more blocks available");
  }

  // all blocks should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Test shared blocks
  {
    Block block = manager.allocate();
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
      EXPECT_EQ(block2, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test assignment operator
    {
      Block block4 = manager.allocate();
      block4 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block4.ref_count(), 2);
      EXPECT_EQ(block4.is_shared(), true);
      EXPECT_EQ(block4, block);

      Block invalid_block;
      invalid_block = block;
      EXPECT_EQ(block.ref_count(), 3);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(invalid_block.ref_count(), 3);
      EXPECT_EQ(invalid_block.is_shared(), true);
      EXPECT_EQ(invalid_block, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_FALSE(block.is_valid());

      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);
      EXPECT_FALSE(block3 == block);
    }
    EXPECT_FALSE(block.is_valid());
  }
}

// prefix_cache_match_length() reports the leading prefix-hit length in blocks
// and must leave the cache (size and utilization) untouched.
TEST(BlockManagerTest, PrefixCacheMatchLengthIsReadOnly) {
  BlockManager::Options options;
  options.num_blocks(16).block_size(2).enable_prefix_cache(true);
  // Leak intentionally: the prefix cache keeps the cached blocks referenced at
  // teardown, which would otherwise trip the free-list check in
  // ~BlockManagerImpl.
  auto* manager = new BlockManagerImpl(options);

  // Three full blocks: [1,2] [3,4] [5,6].
  const std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6};
  std::vector<Block> blocks = manager->allocate(/*num_blocks=*/3);
  ASSERT_EQ(blocks.size(), 3u);
  manager->cache(
      Slice<int32_t>(tokens), blocks, /*existed_shared_blocks_num=*/0);
  ASSERT_EQ(manager->num_blocks_in_prefix_cache(), 3u);

  const size_t cached_before = manager->num_blocks_in_prefix_cache();
  const double util_before = manager->kv_cache_utilization();

  // Full hit, and repeated probing is idempotent.
  const size_t matched1 =
      manager->prefix_cache_match_length(Slice<int32_t>(tokens));
  const size_t matched2 =
      manager->prefix_cache_match_length(Slice<int32_t>(tokens));
  EXPECT_EQ(matched1, 3u);
  EXPECT_EQ(matched2, matched1);

  // A shorter prefix matches only its leading full blocks.
  const std::vector<int32_t> short_tokens = {1, 2, 3, 4};
  EXPECT_EQ(manager->prefix_cache_match_length(Slice<int32_t>(short_tokens)),
            2u);

  // A diverging prefix matches only the shared leading block.
  const std::vector<int32_t> diverged = {1, 2, 9, 9};
  EXPECT_EQ(manager->prefix_cache_match_length(Slice<int32_t>(diverged)), 1u);

  // Probing mutated nothing.
  EXPECT_EQ(manager->num_blocks_in_prefix_cache(), cached_before);
  EXPECT_DOUBLE_EQ(manager->kv_cache_utilization(), util_before);
}

}  // namespace xllm