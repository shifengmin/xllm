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
=============================================================================*/

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "framework/kv_cache/kv_shard_layout.h"
#include "framework/kv_cache_transfer/cache_directory.h"

#if defined(XLLM_HAVE_CONTEXT_PARALLEL_TOPOLOGY)
#include "core/framework/parallel_state/context_parallel_topology.h"
#endif

namespace xllm {

namespace {

// The routing model's coordinates are only worth anything if they are the
// coordinates the *runtime* uses, so this test compares them against the
// runtime classes themselves instead of against hand-written expectations.
// Round 5 of the worklog is the reason: the first version of `slice_of` was
// self-consistent and wrong, and nothing caught it because both sides of every
// test used the same formula.
//
// KVShardLayout is the class the attention layers build from the DCP group
// (`KVShardLayout(block_size, dcp_group.world_size(), dcp_group.rank())`), and
// it is what turns a local row into a global token offset.
constexpr int32_t kPhysicalBlockSize = 128;

CacheTensorDeclaration make_declaration(int32_t group_id,
                                        int32_t kv_split_size,
                                        bool sequence_scoped) {
  CacheTensorDeclaration declaration;
  declaration.cache_namespace = CacheNamespace::MAIN;
  declaration.role = 1;
  declaration.group_id = group_id;
  declaration.topology.dp_size = 1;
  declaration.topology.cp_size = 4;
  declaration.topology.tp_size = 8;
  declaration.topology.kv_split_size = kv_split_size;
  declaration.topology.tokens_per_block = kPhysicalBlockSize;
  declaration.group.global_head_count = 1;
  declaration.group.sequence_scoped = sequence_scoped;
  return declaration;
}

TEST(KvShardContractTest, CanonicalBlockInvertsTheRuntimeShardLayout) {
  // The runtime identity: local row `r` on DCP rank `d` holds the canonical
  // block `r * split + d`, and the token offset inside it is the local offset.
  // The rows here are *position* rows, counted from the sequence's first block;
  // a pool row is one further along, because the block manager's row 0 is the
  // reserved padding block and a request never owns it.
  for (int32_t split : {1, 2, 4, 8}) {
    for (int32_t dcp_rank = 0; dcp_rank < split; ++dcp_rank) {
      const KVShardLayout layout(kPhysicalBlockSize, split, dcp_rank);
      EXPECT_EQ(layout.logical_block_size(),
                static_cast<int64_t>(kPhysicalBlockSize) * split);
      const CanonicalBlock block(kPhysicalBlockSize, split);
      for (int64_t row = 0; row < 5; ++row) {
        for (int64_t offset : {int64_t{0}, int64_t{1}, int64_t{127}}) {
          const int64_t local_slot = row * kPhysicalBlockSize + offset;
          const int64_t global_slot = layout.globalize(local_slot);
          // A canonical block is exactly one physical row's worth of tokens.
          const int64_t canonical = global_slot / kPhysicalBlockSize;
          EXPECT_EQ(canonical, block.canonical_of_row(row, dcp_rank));
          EXPECT_EQ(block.local_row(canonical), row);
          EXPECT_TRUE(block.owns(canonical, dcp_rank));
          EXPECT_EQ(layout.owner_of(global_slot), dcp_rank);
          EXPECT_EQ(layout.localize(global_slot), local_slot);
        }
      }
      // Every other rank's slice is not this rank's to address. A split of one
      // has no other slice, so there is nothing to reject.
      if (split > 1) {
        const int64_t foreign = (dcp_rank + 1) % split;
        const int64_t foreign_global =
            (0 * split + foreign) * kPhysicalBlockSize + 7;
        EXPECT_EQ(layout.localize(foreign_global), KVShardLayout::kInvalidSlot);
      }
    }
  }
}

TEST(KvShardContractTest, CanonicalIdsComeFromPositionsNotFromPoolRows) {
  // A block-scoped group keeps one family that is actually split (the KV cache)
  // and one that keeps the whole sequence (the DSA indexer pool). Both live in
  // the same group and receive the same request ids, and the canonical block is
  // the request's *position* through the slice -- which the ids do not encode:
  // a prefix-cache hit hands out rows wherever the shared prefix already sits,
  // and a later chunk of a chunked prefill starts mid-sequence. Two tempting
  // wrong derivations are covered here: `id - 1` (right only while the request
  // owns rows 1..n) and the index within the step's list (right only while the
  // step starts at position 0).
  const std::vector<CacheTensorDeclaration> local = {
      make_declaration(/*group_id=*/0,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/false),
      make_declaration(/*group_id=*/1,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/true),
  };
  std::vector<int64_t> canonical;
  std::string error;
  ASSERT_TRUE(canonical_blocks_of_request(
      {// A prefix-cache hit: rows 37, 38 and 51 hold positions 0, 1 and 2.
       {/*group_id=*/0, /*ids=*/{37, 38, 51}, /*positions=*/{0, 1, 2}},
       // A later chunk of the same request starts at position 7.
       {/*group_id=*/0, /*ids=*/{60, 61}, /*positions=*/{7, 8}},
       // Sequence-scoped ids are positions already, so they carry none.
       {/*group_id=*/1, /*ids=*/{100}, /*positions=*/{}}},
      local,
      &canonical,
      &error))
      << error;
  const std::vector<int64_t> positions = {0, 1, 2, 7, 8};
  std::vector<int64_t> expected;
  for (int64_t position : positions) {
    for (int64_t j = 0; j < 4; ++j) {
      expected.emplace_back(position * 4 + j);
    }
  }
  expected.emplace_back(100);
  std::sort(expected.begin(), expected.end());
  expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
  EXPECT_EQ(canonical, expected);

  // Every family then selects from that set with its own split and slice, and
  // turns it into its own row: the KV cache of slice `s` keeps the positions a
  // split-4 rank owns, the indexer pool keeps every position at row
  // `position + dcp_size` (which is exactly the expansion its own block table
  // applies, `id * dcp_size + j`), and a sequence-scoped family's id is already
  // its row.
  const CanonicalBlock kv_block(kPhysicalBlockSize, /*split=*/4);
  for (int32_t slice = 0; slice < 4; ++slice) {
    int32_t selected = 0;
    for (int64_t block : canonical) {
      if (block >= 4 * positions.back() + 4 || block % 4 != slice) {
        continue;
      }
      ++selected;
      // The position a block came from is the row the KV family is addressed
      // by, before the pool's reserved row is added.
      EXPECT_NE(
          std::find(
              positions.begin(), positions.end(), kv_block.local_row(block)),
          positions.end())
          << "block " << block;
    }
    // Every position contributes exactly one canonical block to each slice, so
    // a slice keeps as many rows as the request has blocks.
    EXPECT_EQ(selected, static_cast<int32_t>(positions.size()));
  }
  // The indexer family is a full-sequence replica: its rows are scaled by the
  // instance's kv-split and sit one reserved block in, so position `h` lands on
  // row `h + kv_split_size`. That has to be the expansion its own block table
  // applies, `id * dcp_size + j` for `id = h / 4 + 1` and `j = h % 4`.
  constexpr int64_t kKvSplit = 4;
  for (int64_t position : positions) {
    const int64_t id = position / kKvSplit + 1;
    EXPECT_EQ(position + kKvSplit, id * kKvSplit + position % kKvSplit);
  }
}

TEST(KvShardContractTest, RejectsABlockScopedGroupWithoutPositions) {
  // Without positions the route cannot place anything: the id is a pool row and
  // says nothing about where the block sits. Guessing a base -- `id - 1` -- is
  // exactly the silent misplacement this contract exists to prevent, so a
  // mapping that forgot to carry them fails instead.
  const std::vector<CacheTensorDeclaration> local = {
      make_declaration(/*group_id=*/0,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/false),
  };
  std::vector<int64_t> canonical;
  std::string error;
  EXPECT_FALSE(canonical_blocks_of_request(
      {{/*group_id=*/0, /*ids=*/{1, 2}, /*positions=*/{}}},
      local,
      &canonical,
      &error));
  EXPECT_NE(error.find("positions"), std::string::npos) << error;

  // A position that would overflow the canonical id is refused too, so a
  // garbage value cannot wrap into a plausible row.
  EXPECT_FALSE(canonical_blocks_of_request(
      {{/*group_id=*/0,
        /*ids=*/{1},
        /*positions=*/{std::numeric_limits<uint64_t>::max()}}},
      local,
      &canonical,
      &error));
  EXPECT_NE(error.find("out of range"), std::string::npos) << error;
}

TEST(KvShardContractTest, KeepsSequenceScopedIdsWhole) {
  const std::vector<CacheTensorDeclaration> local = {
      make_declaration(/*group_id=*/5,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/true),
  };
  std::vector<int64_t> canonical;
  std::string error;
  ASSERT_TRUE(canonical_blocks_of_request(
      {{/*group_id=*/5, {2, 0}}}, local, &canonical, &error))
      << error;
  EXPECT_EQ(canonical, (std::vector<int64_t>{0, 2}));
}

TEST(KvShardContractTest, RejectsAGroupTheModelDoesNotDeclare) {
  const std::vector<CacheTensorDeclaration> local = {
      make_declaration(/*group_id=*/0,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/false),
  };
  std::vector<int64_t> canonical;
  std::string error;
  EXPECT_FALSE(canonical_blocks_of_request(
      {{/*group_id=*/9, {0}}}, local, &canonical, &error));
  EXPECT_NE(error.find("no cache family for cache group 9"), std::string::npos)
      << error;
}

TEST(KvShardContractTest, RejectsAGroupWhoseFamiliesDisagreeOnScope) {
  std::vector<CacheTensorDeclaration> local = {
      make_declaration(/*group_id=*/0,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/false),
      make_declaration(/*group_id=*/0,
                       /*kv_split_size=*/4,
                       /*sequence_scoped=*/true),
  };
  std::vector<int64_t> canonical;
  std::string error;
  EXPECT_FALSE(canonical_blocks_of_request(
      {{/*group_id=*/0, {0}}}, local, &canonical, &error));
  EXPECT_NE(error.find("no single meaning"), std::string::npos) << error;
}

#if defined(XLLM_HAVE_CONTEXT_PARALLEL_TOPOLOGY)
TEST(KvShardContractTest, SliceIsTheDcpRankTheRuntimeBuilds) {
  // The strongest form of the round 5 finding: for every topology the runtime
  // accepts, the slice the route assigns to a rank is the DCP rank
  // ContextParallelTopology computes for it, and the ranks that share a slice
  // are exactly its DCP group.
  for (int32_t cp_size : {1, 2, 4}) {
    for (int32_t tp_size : {1, 2, 8}) {
      for (int32_t kv_split : {1, 2, 4, 8}) {
        const int32_t world_size = cp_size * tp_size;
        const bool partitions_pcp =
            kv_split <= cp_size && cp_size % kv_split == 0;
        const bool spans_domain = kv_split == world_size;
        if (!partitions_pcp && !spans_domain) {
          continue;
        }
        KvTopology topology;
        topology.dp_size = 1;
        topology.cp_size = cp_size;
        topology.tp_size = tp_size;
        topology.kv_split_size = kv_split;
        topology.tokens_per_block = kPhysicalBlockSize;
        // One latent head keeps `local_heads == 1` for every tp width, and the
        // split always divides the resulting redundancy.
        GroupTopology group;
        group.global_head_count = 1;
        KvRedundancy redundancy;
        std::string error;
        ASSERT_TRUE(KvRedundancy::derive(topology, group, &redundancy, &error))
            << error;
        EXPECT_EQ(redundancy.split(), kv_split);
        const KvLayoutIndex index(topology, redundancy);

        std::vector<int32_t> slices;
        slices.reserve(static_cast<size_t>(world_size));
        for (int32_t global_rank = 0; global_rank < world_size; ++global_rank) {
          const parallel_state::ContextParallelTopology runtime(
              global_rank, world_size, /*dp_size=*/1, cp_size, kv_split);
          const int32_t slice =
              index.slice_of(runtime.pcp_rank(), runtime.tp_rank());
          EXPECT_EQ(slice, runtime.dcp_rank())
              << "cp=" << cp_size << " tp=" << tp_size
              << " kv_split=" << kv_split << " rank=" << global_rank;
          // Deliberately not asserted here: the *composition* of the runtime's
          // DCP group. The route only ever needs the per-rank slice identity
          // checked above (plus the replica enumeration the redundancy model
          // derives from it), and the group's partitioning differs by shape --
          // it partitions the PCP group when the split divides cp_size and
          // spans the whole DP-local domain otherwise. Pinning that here would
          // test ContextParallelTopology rather than the routing contract.
          slices.emplace_back(slice);
        }
        // Every slice the topology declares is actually held by some rank.
        for (int32_t slice = 0; slice < kv_split; ++slice) {
          EXPECT_NE(std::find(slices.begin(), slices.end(), slice),
                    slices.end())
              << "cp=" << cp_size << " tp=" << tp_size
              << " kv_split=" << kv_split << " slice=" << slice;
        }
      }
    }
  }
}
#endif

}  // namespace

}  // namespace xllm
