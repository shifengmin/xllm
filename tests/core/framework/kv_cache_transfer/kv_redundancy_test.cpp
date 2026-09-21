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

#include "framework/kv_cache_transfer/kv_redundancy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace xllm {

namespace {

// GLM 5.3 flash (glm5_next) target scenario. The split is placed by the DCP
// process group, which ContextParallelTopology only allows to either partition
// the PCP group or cover the whole DP-local domain, so the prefill side is
// dp1 cp4 tp8 kv_split4 (world 32: world / kv_split == tp_size, i.e. the DCP
// group is the PCP group at a fixed tp rank) and the decode side is
// dp4 cp1 tp2 kv_split2 (kv_split == cp * tp, each rank its own slice).
// MLA latent and the shared-head indexer have exactly one global head, so they
// are the only groups whose redundancy can absorb a split; the KDA
// conv/recurrent states expose kda_num_heads (64) global heads and therefore
// have no redundancy to remove.
constexpr int32_t kMlaGlobalHeads = 1;
constexpr int32_t kKdaGlobalHeads = 64;
constexpr int32_t kTokensPerBlock = 128;

KvTopology make_topology(int32_t dp_size,
                         int32_t cp_size,
                         int32_t tp_size,
                         int32_t kv_split_size) {
  KvTopology topology;
  topology.dp_size = dp_size;
  topology.cp_size = cp_size;
  topology.tp_size = tp_size;
  topology.kv_split_size = kv_split_size;
  topology.tokens_per_block = kTokensPerBlock;
  return topology;
}

GroupTopology make_group(int32_t global_head_count, bool sequence_scoped) {
  GroupTopology group;
  group.global_head_count = global_head_count;
  group.head_bytes = 1024;
  group.sequence_scoped = sequence_scoped;
  return group;
}

// Returns the derived redundancy, asserting the configuration is accepted.
KvRedundancy derive_ok(const KvTopology& topology, const GroupTopology& group) {
  KvRedundancy redundancy;
  std::string error;
  EXPECT_TRUE(KvRedundancy::derive(topology, group, &redundancy, &error))
      << error;
  return redundancy;
}

void expect_rejected(const KvTopology& topology, const GroupTopology& group) {
  KvRedundancy redundancy;
  std::string error;
  EXPECT_FALSE(KvRedundancy::derive(topology, group, &redundancy, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace

TEST(KvRedundancyTest, DerivesTheGlm53FlashPilotScenario) {
  // Prefill MLA latent / indexer: G=1, CP=4, TP=8 => D_tp=8, D=32, split=4,
  // N_rep=8. The DCP group is the PCP group at a fixed tp rank, so the
  // sequence slice of a rank is its cp rank and the writers sit at tp 0.
  const KvTopology prefill_topology = make_topology(/*dp_size=*/1,
                                                    /*cp_size=*/4,
                                                    /*tp_size=*/8,
                                                    /*kv_split_size=*/4);
  const KvRedundancy prefill = derive_ok(
      prefill_topology, make_group(kMlaGlobalHeads, /*sequence_scoped=*/false));
  EXPECT_EQ(prefill.local_head_count(), 1);
  EXPECT_EQ(prefill.tp_redundancy(), 8);
  EXPECT_EQ(prefill.head_class_count(), 1);
  EXPECT_EQ(prefill.redundancy(), 32);
  EXPECT_EQ(prefill.split(), 4);
  EXPECT_EQ(prefill.replica_count(), 8);

  const KvLayoutIndex prefill_index(prefill_topology, prefill);
  for (int32_t cp = 0; cp < 4; ++cp) {
    for (int32_t tp = 0; tp < 8; ++tp) {
      // slice == dcp_rank == cp, replica == tp (all TP ranks are copies).
      EXPECT_EQ(prefill_index.slice_of(cp, tp), cp);
      EXPECT_EQ(prefill_index.replica_of(cp, tp), tp);
    }
    int32_t writer = -1;
    ASSERT_TRUE(prefill_index.writer_of(/*dp_rank=*/0, 0, cp, &writer));
    EXPECT_EQ(writer, cp * 8);
    std::vector<int32_t> replicas;
    ASSERT_TRUE(prefill_index.replicas_of(/*dp_rank=*/0, 0, cp, &replicas));
    ASSERT_EQ(replicas.size(), 8U);
    EXPECT_EQ(replicas.front(), writer);
  }

  // Decode MLA latent: G=1, CP=1, TP=2, split=2 == cp * tp, so the DCP group
  // covers the whole DP-local domain and every rank holds its own slice.
  const KvTopology decode_topology = make_topology(/*dp_size=*/4,
                                                   /*cp_size=*/1,
                                                   /*tp_size=*/2,
                                                   /*kv_split_size=*/2);
  const KvRedundancy decode = derive_ok(
      decode_topology, make_group(kMlaGlobalHeads, /*sequence_scoped=*/false));
  EXPECT_EQ(decode.tp_redundancy(), 2);
  EXPECT_EQ(decode.redundancy(), 2);
  EXPECT_EQ(decode.split(), 2);
  EXPECT_EQ(decode.replica_count(), 1);

  const KvLayoutIndex decode_index(decode_topology, decode);
  for (int32_t tp = 0; tp < 2; ++tp) {
    EXPECT_EQ(decode_index.slice_of(/*cp_rank=*/0, tp), tp);
    EXPECT_EQ(decode_index.replica_of(/*cp_rank=*/0, tp), 0);
  }
  int32_t writer = -1;
  ASSERT_TRUE(decode_index.writer_of(/*dp_rank=*/3, 0, /*slice=*/1, &writer));
  EXPECT_EQ(writer, 3 * 2 + 1);
}

TEST(KvRedundancyTest, SequenceScopedGroupsAreNeverSplit) {
  // The KDA conv / recurrent states are sequence scoped: there is no block
  // dimension to split, so the whole sequence stays on every rank even though
  // the instance is configured with kv_split > 1.
  const KvRedundancy prefill =
      derive_ok(make_topology(1, 2, 8, 2), make_group(kKdaGlobalHeads, true));
  EXPECT_EQ(prefill.local_head_count(), 8);
  EXPECT_EQ(prefill.head_class_count(), 8);
  EXPECT_EQ(prefill.tp_redundancy(), 1);
  EXPECT_EQ(prefill.redundancy(), 2);
  EXPECT_EQ(prefill.split(), 1);
  EXPECT_EQ(prefill.replica_count(), 2);
  EXPECT_TRUE(prefill.sequence_scoped());

  const KvRedundancy decode =
      derive_ok(make_topology(4, 1, 2, 2), make_group(kKdaGlobalHeads, true));
  EXPECT_EQ(decode.local_head_count(), 32);
  EXPECT_EQ(decode.head_class_count(), 2);
  EXPECT_EQ(decode.split(), 1);
}

TEST(KvRedundancyTest, FullSequenceReplicasSplitOnAShardedInstance) {
  // The DSA indexer pool has a single global head, so its redundancy could
  // absorb the configured split, and its top-k reads the whole sequence: the
  // group declares itself a full-sequence replica. That holds on an instance
  // without CP, where every rank sees every token -- but with CP each rank
  // computes only its own sequence shard and writes just that shard into the
  // pool, so the family has to split exactly like the K/V cache. Routing it as
  // a replica there shipped one writer's shard to the peer as if it were the
  // whole sequence.
  GroupTopology indexer =
      make_group(kMlaGlobalHeads, /*sequence_scoped=*/false);
  indexer.full_sequence_replica = true;

  const KvRedundancy sharded = derive_ok(make_topology(/*dp_size=*/1,
                                                       /*cp_size=*/4,
                                                       /*tp_size=*/8,
                                                       /*kv_split_size=*/4),
                                         indexer);
  EXPECT_EQ(sharded.split(), 4);
  EXPECT_EQ(sharded.replica_count(), 8);
  EXPECT_FALSE(sharded.sequence_scoped());
  EXPECT_FALSE(sharded.full_sequence_replica());

  const KvRedundancy replica = derive_ok(make_topology(4, 1, 2, 2), indexer);
  EXPECT_EQ(replica.split(), 1);
  EXPECT_EQ(replica.replica_count(), 2);
  EXPECT_TRUE(replica.full_sequence_replica());
}

TEST(KvRedundancyTest, GroupsWithoutRedundancyKeepSplitOne) {
  // G >= TP shards the heads; there is nothing to remove, so the split stays 1.
  // Once C3 restricts the instance split to a DCP shape, D == 1 also forces the
  // configured split to 1 (a case-(b) split would need G == 1 to divide D), so
  // this is the only reachable shape for such a group.
  const KvRedundancy sharded =
      derive_ok(make_topology(1, 1, 8, 1), make_group(/*G=*/8, false));
  EXPECT_EQ(sharded.tp_redundancy(), 1);
  EXPECT_EQ(sharded.redundancy(), 1);
  EXPECT_EQ(sharded.split(), 1);
}

TEST(KvRedundancyTest, CpMultipliesTheRedundancy) {
  // CP replicates KV, so D = CP * D_tp.
  const KvRedundancy redundancy =
      derive_ok(make_topology(1, /*cp_size=*/2, 8, 2), make_group(8, false));
  EXPECT_EQ(redundancy.tp_redundancy(), 1);
  EXPECT_EQ(redundancy.redundancy(), 2);
  EXPECT_EQ(redundancy.split(), 2);
  EXPECT_EQ(redundancy.replica_count(), 1);
}

TEST(KvRedundancyTest, RejectsHeadCountThatIsNeitherDivisibleNorDividing) {
  expect_rejected(make_topology(1, 1, 8, 1), make_group(/*G=*/3, false));
  expect_rejected(make_topology(1, 1, /*tp_size=*/3, 1), make_group(8, false));
  expect_rejected(make_topology(1, 1, 8, 1), make_group(/*G=*/0, false));
}

TEST(KvRedundancyTest, RejectsSplitThatIsNotADcpShape) {
  // The runtime builds its DCP process group from the instance split, and
  // ContextParallelTopology accepts only the three shapes the derivation
  // documents: divides cp_size, covers the whole DP-local domain, or -- without
  // PCP -- divides tp_size. cp2/tp3/S3 passes the redundancy check (3 divides
  // D=6) but is none of them, so it aborts when the group is built and has to
  // be rejected here first.
  const KvTopology invalid = make_topology(/*dp_size=*/1,
                                           /*cp_size=*/2,
                                           /*tp_size=*/3,
                                           /*kv_split_size=*/3);
  KvRedundancy redundancy;
  std::string error;
  EXPECT_FALSE(KvRedundancy::derive(
      invalid, make_group(/*G=*/1, false), &redundancy, &error));
  EXPECT_NE(error.find("DCP shape"), std::string::npos) << error;

  // Without PCP a split that divides TP is legal: the DCP groups are the
  // consecutive blocks of that width, which is what makes the 8-card
  // tp8/S2 -> tp8/S4 pair expressible at all.
  EXPECT_TRUE(KvRedundancy::derive(make_topology(1, 1, 8, 4),
                                   make_group(/*G=*/1, false),
                                   &redundancy,
                                   &error))
      << error;
  // 4 divides CP=4 as well, so the same split is fine one PCP width up.
  EXPECT_TRUE(KvRedundancy::derive(make_topology(1, 4, 8, 4),
                                   make_group(/*G=*/1, false),
                                   &redundancy,
                                   &error))
      << error;
}

TEST(KvRedundancyTest, RejectsSplitThatExceedsOrDoesNotDivideRedundancy) {
  // CP=1, TP=8, S=8 covers the whole DP-local domain: a DCP shape, but D=2 for
  // G=4, so the split exceeds the group redundancy.
  expect_rejected(make_topology(1, 1, 8, 8), make_group(/*G=*/4, false));
  // Same shape, G=2 => D=4, and 8 does not divide 4.
  expect_rejected(make_topology(1, 1, 8, 8), make_group(/*G=*/2, false));
  // CP=2, TP=8, S=2 is a DCP shape, but D=2 for G=8 means it cannot absorb the
  // split... it can here, so use CP=2 with S=2 and G=32 (D_tp=1, D=2).
  expect_rejected(make_topology(1, 2, 8, 4), make_group(8, false));
}

TEST(KvRedundancyTest, HeadClassesTileTheGlobalHeadRange) {
  const std::vector<int32_t> head_counts = {1, 2, 4, 8, 16, 32, 64};
  const std::vector<int32_t> tp_sizes = {1, 2, 4, 8};
  const std::vector<int32_t> cp_sizes = {1, 2, 4};
  int32_t accepted = 0;
  for (int32_t global_heads : head_counts) {
    for (int32_t tp_size : tp_sizes) {
      for (int32_t cp_size : cp_sizes) {
        for (int32_t split = 1;
             split <= cp_size * std::max(tp_size / global_heads, 1) + 1;
             ++split) {
          const KvTopology topology = make_topology(1, cp_size, tp_size, split);
          const GroupTopology group = make_group(global_heads, false);
          KvRedundancy redundancy;
          std::string error;
          if (!KvRedundancy::derive(topology, group, &redundancy, &error)) {
            continue;
          }
          ++accepted;
          EXPECT_EQ(
              redundancy.head_class_count() * redundancy.local_head_count(),
              global_heads);
          EXPECT_GE(redundancy.split(), 1);
          EXPECT_LE(redundancy.split(), redundancy.redundancy());
          EXPECT_EQ(redundancy.redundancy() % redundancy.split(), 0);
          EXPECT_EQ(redundancy.replica_count(),
                    redundancy.redundancy() / redundancy.split());
        }
      }
    }
  }
  EXPECT_GT(accepted, 100);
}

TEST(KvLayoutIndexTest, EverySliceHasOneWriterAndNrepReplicas) {
  const std::vector<int32_t> head_counts = {1, 2, 4, 8, 16, 32, 64};
  const std::vector<int32_t> tp_sizes = {1, 2, 4, 8};
  const std::vector<int32_t> cp_sizes = {1, 2, 4};
  int32_t accepted = 0;
  for (int32_t global_heads : head_counts) {
    for (int32_t tp_size : tp_sizes) {
      for (int32_t cp_size : cp_sizes) {
        for (int32_t split = 1;
             split <= cp_size * std::max(tp_size / global_heads, 1);
             ++split) {
          const KvTopology topology =
              make_topology(/*dp_size=*/2, cp_size, tp_size, split);
          const GroupTopology group = make_group(global_heads, false);
          KvRedundancy redundancy;
          std::string error;
          if (!KvRedundancy::derive(topology, group, &redundancy, &error)) {
            continue;
          }
          ++accepted;
          const KvLayoutIndex index(topology, redundancy);
          const std::string tag = "G=" + std::to_string(global_heads) +
                                  " TP=" + std::to_string(tp_size) +
                                  " CP=" + std::to_string(cp_size) +
                                  " S=" + std::to_string(split);

          for (int32_t dp = 0; dp < topology.dp_size; ++dp) {
            for (int32_t h = 0; h < redundancy.head_class_count(); ++h) {
              EXPECT_EQ(index.head_begin(h), h * redundancy.local_head_count())
                  << tag;
              EXPECT_EQ(index.head_end(h),
                        (h + 1) * redundancy.local_head_count())
                  << tag;

              std::map<int32_t, int32_t> slice_histogram;
              for (int32_t cp = 0; cp < cp_size; ++cp) {
                for (int32_t tp = 0; tp < tp_size; ++tp) {
                  if (index.head_class_of(tp) != h) {
                    continue;
                  }
                  const int32_t slice = index.slice_of(cp, tp);
                  const int32_t replica = index.replica_of(cp, tp);
                  EXPECT_GE(slice, 0) << tag;
                  EXPECT_LT(slice, redundancy.split()) << tag;
                  EXPECT_GE(replica, 0) << tag;
                  EXPECT_LT(replica, redundancy.replica_count()) << tag;
                  ++slice_histogram[slice];
                }
              }
              // No gaps and no extra slices; each slice is held exactly
              // N_rep times, once per redundant copy.
              EXPECT_EQ(static_cast<int32_t>(slice_histogram.size()),
                        redundancy.split())
                  << tag;
              for (const auto& [slice, count] : slice_histogram) {
                EXPECT_EQ(count, redundancy.replica_count()) << tag;
              }

              std::set<int32_t> writers;
              for (int32_t slice = 0; slice < redundancy.split(); ++slice) {
                int32_t writer = -1;
                ASSERT_TRUE(index.writer_of(dp, h, slice, &writer)) << tag;
                EXPECT_TRUE(writers.insert(writer).second) << tag;
                const int32_t cp = (writer % (cp_size * tp_size)) / tp_size;
                const int32_t tp = writer % tp_size;
                EXPECT_EQ(index.head_class_of(tp), h) << tag;
                EXPECT_EQ(index.slice_of(cp, tp), slice) << tag;
                EXPECT_EQ(index.replica_of(cp, tp), 0) << tag;

                std::vector<int32_t> replicas;
                ASSERT_TRUE(index.replicas_of(dp, h, slice, &replicas)) << tag;
                EXPECT_EQ(static_cast<int32_t>(replicas.size()),
                          redundancy.replica_count())
                    << tag;
                EXPECT_EQ(
                    std::set<int32_t>(replicas.begin(), replicas.end()).size(),
                    replicas.size())
                    << tag;
                EXPECT_EQ(replicas.front(), writer) << tag;
                for (int32_t rank : replicas) {
                  const int32_t replica_cp =
                      (rank % (cp_size * tp_size)) / tp_size;
                  const int32_t replica_tp = rank % tp_size;
                  EXPECT_EQ(index.head_class_of(replica_tp), h) << tag;
                  EXPECT_EQ(index.slice_of(replica_cp, replica_tp), slice)
                      << tag;
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(KvLayoutIndexTest, TilesTheTpAxisWhenThereIsNoPcp) {
  // cp1/tp8 with S=2: the DCP groups are consecutive pairs of TP ranks, so the
  // even ranks hold slice 0 and the odd ranks slice 1, each of them four times.
  const KvTopology topology = make_topology(/*dp_size=*/1,
                                            /*cp_size=*/1,
                                            /*tp_size=*/8,
                                            /*kv_split_size=*/2);
  const KvRedundancy redundancy =
      derive_ok(topology, make_group(/*G=*/1, false));
  EXPECT_EQ(redundancy.split(), 2);
  EXPECT_EQ(redundancy.replica_count(), 4);
  const KvLayoutIndex index(topology, redundancy);
  for (int32_t tp = 0; tp < 8; ++tp) {
    EXPECT_EQ(index.slice_of(/*cp_rank=*/0, tp), tp % 2);
    EXPECT_EQ(index.replica_of(/*cp_rank=*/0, tp), tp / 2);
  }
  int32_t writer = -1;
  ASSERT_TRUE(index.writer_of(/*dp_rank=*/0, /*head_class=*/0, 0, &writer));
  EXPECT_EQ(writer, 0);
  ASSERT_TRUE(index.writer_of(/*dp_rank=*/0, /*head_class=*/0, 1, &writer));
  EXPECT_EQ(writer, 1);

  std::vector<int32_t> replicas;
  ASSERT_TRUE(index.replicas_of(/*dp_rank=*/0, /*head_class=*/0, 0, &replicas));
  EXPECT_EQ(replicas, (std::vector<int32_t>{0, 2, 4, 6}));
  ASSERT_TRUE(index.replicas_of(/*dp_rank=*/0, /*head_class=*/0, 1, &replicas));
  EXPECT_EQ(replicas, (std::vector<int32_t>{1, 3, 5, 7}));

  // S=4 on the same instance: quarters of the TP axis, two replicas per slice.
  const KvTopology quarters = make_topology(1, 1, 8, 4);
  const KvRedundancy quarter_redundancy =
      derive_ok(quarters, make_group(/*G=*/1, false));
  EXPECT_EQ(quarter_redundancy.split(), 4);
  EXPECT_EQ(quarter_redundancy.replica_count(), 2);
  const KvLayoutIndex quarter_index(quarters, quarter_redundancy);
  ASSERT_TRUE(
      quarter_index.replicas_of(/*dp_rank=*/0, /*head_class=*/0, 0, &replicas));
  EXPECT_EQ(replicas, (std::vector<int32_t>{0, 4}));
  ASSERT_TRUE(
      quarter_index.replicas_of(/*dp_rank=*/0, /*head_class=*/0, 3, &replicas));
  EXPECT_EQ(replicas, (std::vector<int32_t>{3, 7}));

  // With more than one head class the ranks of one TP block belong to different
  // classes, so a class-slice pair can be held by a rank that is not the first
  // of its block -- and a replica index counts from the first holder of that
  // pair rather than from the block.
  const KvTopology two_classes = make_topology(1, 1, 4, 2);
  const KvRedundancy class_redundancy =
      derive_ok(two_classes, make_group(/*G=*/2, false));
  EXPECT_EQ(class_redundancy.split(), 2);
  EXPECT_EQ(class_redundancy.replica_count(), 1);
  const KvLayoutIndex class_index(two_classes, class_redundancy);
  for (int32_t tp = 0; tp < 4; ++tp) {
    EXPECT_EQ(class_index.slice_of(/*cp_rank=*/0, tp), tp % 2);
    EXPECT_EQ(class_index.replica_of(/*cp_rank=*/0, tp), 0);
  }
  // Head class 1 owns TP ranks 2 and 3, so the holder of (class 1, slice 0) is
  // rank 2 and it is that pair's writer.
  ASSERT_TRUE(
      class_index.writer_of(/*dp_rank=*/0, /*head_class=*/1, 0, &writer));
  EXPECT_EQ(writer, 2);
  ASSERT_TRUE(
      class_index.writer_of(/*dp_rank=*/0, /*head_class=*/1, 1, &writer));
  EXPECT_EQ(writer, 3);
  ASSERT_TRUE(
      class_index.replicas_of(/*dp_rank=*/0, /*head_class=*/1, 0, &replicas));
  EXPECT_EQ(replicas, (std::vector<int32_t>{2}));
}

TEST(KvLayoutIndexTest, RejectsOutOfRangeQueries) {
  const KvTopology topology = make_topology(1, 4, 8, 4);
  const KvRedundancy redundancy =
      derive_ok(topology, make_group(kMlaGlobalHeads, false));
  const KvLayoutIndex index(topology, redundancy);

  int32_t rank = -1;
  EXPECT_FALSE(index.writer_of(/*dp_rank=*/-1, 0, 0, &rank));
  EXPECT_FALSE(index.writer_of(/*dp_rank=*/topology.dp_size, 0, 0, &rank));
  EXPECT_FALSE(index.writer_of(0, /*head_class=*/-1, 0, &rank));
  EXPECT_FALSE(index.writer_of(
      0, /*head_class=*/redundancy.head_class_count(), 0, &rank));
  EXPECT_FALSE(index.writer_of(0, 0, /*slice=*/-1, &rank));
  EXPECT_FALSE(index.writer_of(0, 0, /*slice=*/redundancy.split(), &rank));
  EXPECT_FALSE(index.writer_of(0, 0, 0, nullptr));

  std::vector<int32_t> ranks;
  EXPECT_FALSE(index.replicas_of(/*dp_rank=*/-1, 0, 0, &ranks));
  EXPECT_FALSE(index.replicas_of(/*dp_rank=*/topology.dp_size, 0, 0, &ranks));
  EXPECT_FALSE(index.replicas_of(0, 0, /*slice=*/redundancy.split(), &ranks));
  EXPECT_FALSE(index.replicas_of(0, 0, 0, nullptr));
}

TEST(CanonicalBlockTest, RowMappingRoundTripsForOwnedBlocks) {
  // Prefill split 4 and decode split 2 of the pilot scenario share the same
  // canonical block unit (the physical row = tokens_per_block tokens).
  const CanonicalBlock prefill(kTokensPerBlock, /*split=*/4);
  const CanonicalBlock decode(kTokensPerBlock, /*split=*/2);

  EXPECT_EQ(prefill.block_of_token(0), 0);
  EXPECT_EQ(prefill.block_of_token(kTokensPerBlock - 1), 0);
  EXPECT_EQ(prefill.block_of_token(kTokensPerBlock), 1);
  EXPECT_EQ(prefill.token_begin(3), 3 * kTokensPerBlock);
  EXPECT_EQ(prefill.token_end(3), 4 * kTokensPerBlock);

  for (int64_t block = 0; block < 64; ++block) {
    for (int32_t slice = 0; slice < 4; ++slice) {
      EXPECT_EQ(prefill.owns(block, slice), block % 4 == slice);
      if (!prefill.owns(block, slice)) {
        continue;
      }
      EXPECT_EQ(prefill.canonical_of_row(prefill.local_row(block), slice),
                block);
    }
  }
  // Decode collapses two canonical blocks into one physical row.
  EXPECT_EQ(decode.local_row(0), 0);
  EXPECT_EQ(decode.local_row(2), 1);
  EXPECT_EQ(decode.local_row(4), 2);
  EXPECT_EQ(decode.canonical_of_row(2, 1), 5);
  EXPECT_TRUE(decode.owns(5, 1));
  EXPECT_FALSE(decode.owns(5, 0));
}

}  // namespace xllm
