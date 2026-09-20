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

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/pd_route_table.h"
#include "framework/kv_cache_transfer/route_binder.h"

namespace xllm {

namespace {

// GLM 5.3 flash geometry: MLA latent is one 1024 byte head, blocks hold 128
// tokens.
constexpr int32_t kHeadBytes = 1024;
constexpr int32_t kTokensPerBlock = 128;
constexpr int32_t kSsmHeadBytes = 1024;

KvTopology make_topology(int32_t dp_size,
                         int32_t cp_size,
                         int32_t tp_size,
                         int32_t kv_split_size,
                         int32_t tokens_per_block) {
  KvTopology topology;
  topology.dp_size = dp_size;
  topology.cp_size = cp_size;
  topology.tp_size = tp_size;
  topology.kv_split_size = kv_split_size;
  topology.tokens_per_block = tokens_per_block;
  return topology;
}

GroupTopology make_group(int32_t global_head_count,
                         uint64_t head_bytes,
                         bool sequence_scoped,
                         bool full_sequence_replica) {
  GroupTopology group;
  group.global_head_count = global_head_count;
  group.head_bytes = head_bytes;
  group.sequence_scoped = sequence_scoped;
  group.full_sequence_replica = full_sequence_replica;
  return group;
}

// prefill: CP4 + TP8 + DCP4 (world 32, the DCP group is the PCP group at a
// fixed tp rank, so slice == cp_rank); decode: DP4 + CP1 + TP2 + DCP2 (the DCP
// group covers the whole DP-local domain, so slice == tp).
KvTopology prefill_topology(int32_t tokens_per_block) {
  return make_topology(/*dp_size=*/1,
                       /*cp_size=*/4,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       tokens_per_block);
}

KvTopology decode_topology(int32_t tokens_per_block) {
  return make_topology(/*dp_size=*/4,
                       /*cp_size=*/1,
                       /*tp_size=*/2,
                       /*kv_split_size=*/2,
                       tokens_per_block);
}

std::vector<RouteEdge> build_edges(const KvTopology& src_topology,
                                   const GroupTopology& src_group,
                                   const KvTopology& dst_topology,
                                   const GroupTopology& dst_group) {
  std::vector<RouteEdge> edges;
  std::string error;
  EXPECT_TRUE(PdRouteTable::build(
      src_topology, src_group, dst_topology, dst_group, &edges, &error))
      << error;
  return edges;
}

PeerCacheView make_view(const KvTopology& topology,
                        const GroupTopology& group,
                        const BufferDirectoryEntry& entry) {
  PeerCacheView view;
  view.topology = topology;
  view.group = group;
  view.entry = entry;
  // Production reads the head layout out of the descriptor; a fixture whose
  // family is one contiguous head range states it the way the descriptor does:
  // a single run over every local head, whose bytes are one sub-unit.
  KvRedundancy redundancy;
  std::string reason;
  EXPECT_TRUE(KvRedundancy::derive(topology, group, &redundancy, &reason))
      << reason;
  HeadRun run;
  run.head_bytes = group.head_bytes;
  view.head_runs.emplace_back(run);
  view.unit_stride_bytes =
      static_cast<uint64_t>(redundancy.local_head_count()) * group.head_bytes;
  return view;
}

int32_t greatest_common_divisor(int32_t lhs, int32_t rhs) {
  while (rhs != 0) {
    const int32_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

BufferDirectoryEntry make_entry(uint64_t buffer_id,
                                uint64_t resource_count,
                                uint64_t resource_stride_bytes,
                                uint64_t units_per_resource) {
  BufferDirectoryEntry entry;
  entry.buffer_id = buffer_id;
  entry.resource_count = resource_count;
  entry.resource_stride_bytes = resource_stride_bytes;
  entry.buffer_bytes = resource_count * resource_stride_bytes;
  entry.units_per_resource = units_per_resource;
  return entry;
}

}  // namespace

TEST(PdRouteTest, MlaLatentEdgeTableMatchesThePilotScenario) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);

  const std::vector<RouteEdge> edges = build_edges(prefill, mla, decode, mla);
  ASSERT_EQ(edges.size(), 4u);

  // Only the replica-0 ranks of the source write: slice tP lives on cp rank tP
  // at tp 0 (local rank 8 * tP), and slice tP lands on destination slice
  // tP % 2, which on the decode side is tp rank tP % 2.
  const std::array<std::array<int32_t, 4>, 4> golden = {
      {{0, 0, 0, 0}, {8, 1, 1, 1}, {16, 0, 2, 0}, {24, 1, 3, 1}}};
  for (size_t index = 0; index < edges.size(); ++index) {
    const RouteEdge& edge = edges[index];
    EXPECT_EQ(edge.src_local_rank, golden[index][0]);
    EXPECT_EQ(edge.dst_local_rank, golden[index][1]);
    EXPECT_EQ(edge.src_slice, golden[index][2]);
    EXPECT_EQ(edge.dst_slice, golden[index][3]);
    EXPECT_EQ(edge.head_begin, 0);
    EXPECT_EQ(edge.head_end, 1);
  }

  std::string error;
  EXPECT_TRUE(PdRouteTable::validate(edges, prefill, mla, decode, mla, &error))
      << error;

  // The DP widening is applied when binding, so one table serves all four DP
  // pairs: 4 table edges x 4 destination DP groups = 16 effective edges.
  const int32_t dst_local_count = decode.cp_size * decode.tp_size;
  EXPECT_EQ(edges.size() * static_cast<size_t>(decode.dp_size), 16u);
  for (const RouteEdge& edge : edges) {
    const int32_t dst_global = 3 * dst_local_count + edge.dst_local_rank;
    EXPECT_EQ(dst_global % dst_local_count, edge.dst_local_rank);
  }
}

TEST(PdRouteTest, KdaStateCoversEveryPrefillRankAcrossHeadClasses) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology kda = make_group(/*global_head_count=*/64,
                                       kSsmHeadBytes,
                                       /*sequence_scoped=*/true,
                                       /*full_sequence_replica=*/false);

  const std::vector<RouteEdge> edges = build_edges(prefill, kda, decode, kda);
  ASSERT_EQ(edges.size(), 8u);

  // G = 64 shards over TP8 -> eight head classes of eight heads on the source,
  // and over TP2 -> two head classes of 32 heads on the destination. Every
  // prefill rank takes part, unlike the MLA case above.
  const std::array<std::array<int32_t, 3>, 8> golden = {{{0, 0, 0},
                                                         {1, 0, 8},
                                                         {2, 0, 16},
                                                         {3, 0, 24},
                                                         {4, 1, 32},
                                                         {5, 1, 40},
                                                         {6, 1, 48},
                                                         {7, 1, 56}}};
  for (size_t index = 0; index < edges.size(); ++index) {
    const RouteEdge& edge = edges[index];
    EXPECT_EQ(edge.src_local_rank, golden[index][0]);
    EXPECT_EQ(edge.dst_local_rank, golden[index][1]);
    EXPECT_EQ(edge.head_begin, golden[index][2]);
    EXPECT_EQ(edge.head_end, golden[index][2] + 8);
    EXPECT_EQ(edge.src_slice, 0);
    EXPECT_EQ(edge.dst_slice, 0);
  }

  std::string error;
  EXPECT_TRUE(PdRouteTable::validate(edges, prefill, kda, decode, kda, &error))
      << error;
  EXPECT_EQ(edges.size() * static_cast<size_t>(decode.dp_size), 32u);
}

TEST(PdRouteTest, IndexerPoolIsAShardedSourceAndAReplicaDestination) {
  // The pair the PD deployment runs: a prefill instance that shards the
  // sequence (CP 4 + DCP 4) and a decode instance that does not (DCP 2, no CP).
  // The indexer pool is declared a full-sequence replica by the model, and the
  // decode instance honours that -- but on the prefill instance every rank
  // holds only its own DCP slice of it, so each slice has to be carried by the
  // rank that computed it and fanned out to every destination replica rank.
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology indexer = make_group(
      /*global_head_count=*/1,
      /*head_bytes=*/256,
      /*sequence_scoped=*/false,
      /*full_sequence_replica=*/true);

  const std::vector<RouteEdge> edges =
      build_edges(prefill, indexer, decode, indexer);
  ASSERT_EQ(edges.size(), 8u);

  // Four source slices, each written by the replica-0 rank of its DCP group
  // (cp rank == slice, tp 0 of 8), and each fanned out to both destination
  // replicas at destination slice 0.
  std::vector<std::vector<bool>> covered(4, std::vector<bool>(2, false));
  for (const RouteEdge& edge : edges) {
    ASSERT_GE(edge.src_slice, 0);
    ASSERT_LT(edge.src_slice, 4);
    EXPECT_EQ(edge.src_local_rank, edge.src_slice * 8);
    EXPECT_EQ(edge.dst_slice, 0);
    ASSERT_GE(edge.dst_local_rank, 0);
    ASSERT_LT(edge.dst_local_rank, 2);
    covered[static_cast<size_t>(edge.src_slice)]
           [static_cast<size_t>(edge.dst_local_rank)] = true;
  }
  for (const std::vector<bool>& row : covered) {
    EXPECT_TRUE(row[0]);
    EXPECT_TRUE(row[1]);
  }

  std::string error;
  EXPECT_TRUE(
      PdRouteTable::validate(edges, prefill, indexer, decode, indexer, &error))
      << error;
}

TEST(PdRouteTest, RouteInvariantsHoldOverTheTopologyMatrix) {
  const std::array<int32_t, 7> head_counts = {1, 2, 4, 8, 16, 32, 64};
  const std::array<int32_t, 4> tp_widths = {1, 2, 4, 8};
  const std::array<int32_t, 3> pcp_widths = {1, 2, 4};
  const std::array<int32_t, 4> splits = {1, 2, 4, 8};
  int32_t buildable = 0;

  for (int32_t global_heads : head_counts) {
    for (int32_t tp_src : tp_widths) {
      for (int32_t tp_dst : tp_widths) {
        for (int32_t cp_src : pcp_widths) {
          for (int32_t cp_dst : pcp_widths) {
            for (bool sequence_scoped : {false, true}) {
              const GroupTopology group =
                  make_group(global_heads, kHeadBytes, sequence_scoped, false);
              for (int32_t split_src : splits) {
                for (int32_t split_dst : splits) {
                  const KvTopology src = make_topology(
                      1, cp_src, tp_src, split_src, kTokensPerBlock);
                  const KvTopology dst = make_topology(
                      1, cp_dst, tp_dst, split_dst, kTokensPerBlock);

                  KvRedundancy src_redundancy;
                  KvRedundancy dst_redundancy;
                  std::string error;
                  if (!KvRedundancy::derive(
                          src, group, &src_redundancy, &error) ||
                      !KvRedundancy::derive(
                          dst, group, &dst_redundancy, &error)) {
                    continue;
                  }
                  const int32_t src_split = src_redundancy.split();
                  const int32_t dst_split = dst_redundancy.split();
                  const bool nested =
                      src_split % dst_split == 0 || dst_split % src_split == 0;

                  std::vector<RouteEdge> edges;
                  const bool built = PdRouteTable::build(
                      src, group, dst, group, &edges, &error);
                  if (!nested) {
                    EXPECT_FALSE(built) << "splits " << src_split << "/"
                                        << dst_split << " are not nested";
                    continue;
                  }
                  ASSERT_TRUE(built) << error;
                  ++buildable;
                  EXPECT_TRUE(PdRouteTable::validate(
                      edges, src, group, dst, group, &error))
                      << error;

                  // Every head of every source replica-0 rank is sent exactly
                  // once per destination replica.
                  const int32_t src_ranks = src.cp_size * src.tp_size;
                  std::vector<int32_t> per_source(
                      static_cast<size_t>(src_ranks) *
                          static_cast<size_t>(src_split) *
                          static_cast<size_t>(global_heads),
                      0);
                  for (const RouteEdge& edge : edges) {
                    for (int32_t head = edge.head_begin; head < edge.head_end;
                         ++head) {
                      const size_t index =
                          (static_cast<size_t>(edge.src_local_rank) *
                               static_cast<size_t>(src_split) +
                           static_cast<size_t>(edge.src_slice)) *
                              static_cast<size_t>(global_heads) +
                          static_cast<size_t>(head);
                      ++per_source[index];
                    }
                  }
                  const int32_t period =
                      dst_split / greatest_common_divisor(src_split, dst_split);
                  size_t non_zero = 0;
                  for (int32_t count : per_source) {
                    if (count == 0) {
                      continue;
                    }
                    ++non_zero;
                    EXPECT_EQ(count, period * dst_redundancy.replica_count());
                  }
                  // Only the replica-0 rank of each (head class, slice) writes.
                  EXPECT_EQ(non_zero,
                            static_cast<size_t>(global_heads) *
                                static_cast<size_t>(src_split));
                }
              }
            }
          }
        }
      }
    }
  }
  EXPECT_GT(buildable, 0);
}

TEST(PdRouteTest, BinderProducesTheGoldenByteRanges) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, mla, decode, mla);

  const uint64_t row_bytes =
      static_cast<uint64_t>(kTokensPerBlock) * kHeadBytes;
  const BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/7,
                 /*resource_count=*/8,
                 row_bytes,
                 /*units_per_resource=*/kTokensPerBlock);
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/9,
                 /*resource_count=*/8,
                 row_bytes,
                 /*units_per_resource=*/kTokensPerBlock);
  const PeerCacheView local = make_view(prefill, mla, local_entry);
  const PeerCacheView remote = make_view(decode, mla, remote_entry);

  // local_row = block / 4 + 1, remote_row = block / 2 + 1 (row 0 of a pool is
  // the reserved padding block), and the whole row moves because the head range
  // covers the single MLA head and the token dimension compresses into one
  // contiguous run.
  for (int64_t block = 0; block < 8; ++block) {
    const int32_t dst_rank = static_cast<int32_t>(block % 2);
    std::vector<RouteRegion> regions;
    std::string error;
    ASSERT_TRUE(RouteBinder::bind(
        edges, dst_rank, {block}, local, remote, &regions, &error))
        << error;
    ASSERT_EQ(regions.size(), 1u) << "block " << block;
    EXPECT_EQ(regions[0].local_buffer_id, 7u);
    EXPECT_EQ(regions[0].remote_buffer_id, 9u);
    EXPECT_EQ(regions[0].local_offset,
              (block / 4 + 1) * static_cast<int64_t>(row_bytes));
    EXPECT_EQ(regions[0].remote_offset,
              (block / 2 + 1) * static_cast<int64_t>(row_bytes));
    EXPECT_EQ(regions[0].length, row_bytes);
  }
}

TEST(PdRouteTest, BinderFansOutWhenTheSourceIsNarrower) {
  const KvTopology narrow = decode_topology(kTokensPerBlock);
  const KvTopology wide = prefill_topology(kTokensPerBlock);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(narrow, mla, wide, mla);
  // Two source slices, two destination slices each, and eight destination
  // replicas (every TP rank of the wide side holds the single MLA head).
  ASSERT_EQ(edges.size(), 32u);

  const uint64_t row_bytes =
      static_cast<uint64_t>(kTokensPerBlock) * kHeadBytes;
  const BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/7,
                 /*resource_count=*/8,
                 row_bytes,
                 /*units_per_resource=*/kTokensPerBlock);
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/9,
                 /*resource_count=*/8,
                 row_bytes,
                 /*units_per_resource=*/kTokensPerBlock);
  const PeerCacheView local = make_view(narrow, mla, local_entry);
  const PeerCacheView remote = make_view(wide, mla, remote_entry);

  // Destination rank 0 owns slice 0, which is fed by source slice 0 (rank 0);
  // blocks 0 and 4 are the ones it needs. Pool rows are one past the position
  // row: local (split 2) rows 1 and 3, remote (split 4) rows 1 and 2.
  std::vector<RouteRegion> regions;
  std::string error;
  ASSERT_TRUE(RouteBinder::bind(
      edges, /*dst_local_rank=*/0, {0, 4}, local, remote, &regions, &error))
      << error;
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_EQ(regions[0].local_offset, static_cast<int64_t>(row_bytes));
  EXPECT_EQ(regions[0].remote_offset, static_cast<int64_t>(row_bytes));
  EXPECT_EQ(regions[0].length, row_bytes);
  EXPECT_EQ(regions[1].local_offset, 3 * static_cast<int64_t>(row_bytes));
  EXPECT_EQ(regions[1].remote_offset, 2 * static_cast<int64_t>(row_bytes));
  EXPECT_EQ(regions[1].length, row_bytes);
}

TEST(PdRouteTest, BinderHonoursExplicitRowBases) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, mla, decode, mla);

  const uint64_t row_bytes =
      static_cast<uint64_t>(kTokensPerBlock) * kHeadBytes;
  BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/7,
                 /*resource_count=*/8,
                 row_bytes,
                 /*units_per_resource=*/kTokensPerBlock);
  BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/9,
                 /*resource_count=*/8,
                 row_bytes,
                 /*units_per_resource=*/kTokensPerBlock);
  remote_entry.explicit_offsets = true;

  PeerCacheView local = make_view(prefill, mla, local_entry);
  PeerCacheView remote = make_view(decode, mla, remote_entry);
  remote.row_offsets.reserve(8);
  for (int64_t row = 0; row < 8; ++row) {
    remote.row_offsets.emplace_back(static_cast<uint64_t>(row) * 4096 + 1024);
  }

  for (int64_t block = 0; block < 8; ++block) {
    const int32_t dst_rank = static_cast<int32_t>(block % 2);
    std::vector<RouteRegion> regions;
    std::string error;
    ASSERT_TRUE(RouteBinder::bind(
        edges, dst_rank, {block}, local, remote, &regions, &error))
        << error;
    ASSERT_EQ(regions.size(), 1u);
    EXPECT_EQ(regions[0].remote_offset,
              remote.row_offsets[static_cast<size_t>(block / 2 + 1)]);
    EXPECT_EQ(regions[0].local_offset,
              (block / 4 + 1) * static_cast<int64_t>(row_bytes));
  }
}

TEST(PdRouteTest, BinderExpandsCheckpointSubUnitsOfSequenceState) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology kda = make_group(/*global_head_count=*/64,
                                       kSsmHeadBytes,
                                       /*sequence_scoped=*/true,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, kda, decode, kda);
  ASSERT_EQ(edges.size(), 8u);

  // Checkpointed recurrent state: three rows per sequence slot, eight local
  // heads on the source and 32 on the destination.
  const uint64_t local_row_bytes = 3 * 8 * static_cast<uint64_t>(kSsmHeadBytes);
  const uint64_t remote_row_bytes =
      3 * 32 * static_cast<uint64_t>(kSsmHeadBytes);
  const BufferDirectoryEntry local_entry = make_entry(/*buffer_id=*/11,
                                                      /*resource_count=*/8,
                                                      local_row_bytes,
                                                      /*units_per_resource=*/3);
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/12,
                 /*resource_count=*/8,
                 remote_row_bytes,
                 /*units_per_resource=*/3);
  const PeerCacheView local = make_view(prefill, kda, local_entry);
  const PeerCacheView remote = make_view(decode, kda, remote_entry);

  // Slot 0 arrives from prefill rank 0 with the head range [0, 8).
  std::vector<RouteRegion> regions;
  std::string error;
  ASSERT_TRUE(RouteBinder::bind(
      {edges[0]}, /*dst_local_rank=*/0, {0}, local, remote, &regions, &error))
      << error;
  ASSERT_EQ(regions.size(), 3u);
  for (size_t unit = 0; unit < regions.size(); ++unit) {
    EXPECT_EQ(regions[unit].length, 8 * static_cast<uint64_t>(kSsmHeadBytes));
    EXPECT_EQ(regions[unit].local_offset,
              unit * 8 * static_cast<uint64_t>(kSsmHeadBytes));
    EXPECT_EQ(regions[unit].remote_offset,
              unit * 32 * static_cast<uint64_t>(kSsmHeadBytes));
  }

  // The four source ranks that own parts of destination rank 0's head class
  // contribute one head range each, and each pair is bound on its own because
  // one region list describes one source buffer.
  int32_t pairs_into_rank_zero = 0;
  for (const RouteEdge& edge : edges) {
    if (edge.dst_local_rank != 0) {
      continue;
    }
    ++pairs_into_rank_zero;
    ASSERT_TRUE(RouteBinder::bind(
        {edge}, /*dst_local_rank=*/0, {0}, local, remote, &regions, &error))
        << error;
    EXPECT_EQ(regions.size(), 3u);
    EXPECT_EQ(regions.front().length, 8 * static_cast<uint64_t>(kSsmHeadBytes));
  }
  EXPECT_EQ(pairs_into_rank_zero, 4);
}

// A composite family packs several logical tensors into one physical row -- the
// Qwen3.5 conv state is `[key_a | key_b | value]` -- so a head range is one
// byte range per component rather than one contiguous range. Binding therefore
// has to walk every component's own offsets, and those offsets differ between
// peers whenever their TP widths do: this test reshard one slot from one TP1
// rank onto the two ranks of a TP2 instance.
TEST(PdRouteTest, BinderWalksEveryRunOfACompositeRow) {
  const KvTopology source = make_topology(/*dp_size=*/1,
                                          /*cp_size=*/1,
                                          /*tp_size=*/1,
                                          /*kv_split_size=*/1,
                                          kTokensPerBlock);
  const KvTopology dest = make_topology(/*dp_size=*/1,
                                        /*cp_size=*/1,
                                        /*tp_size=*/2,
                                        /*kv_split_size=*/1,
                                        kTokensPerBlock);
  const GroupTopology conv = make_group(/*global_head_count=*/4,
                                        /*head_bytes=*/16,
                                        /*sequence_scoped=*/true,
                                        /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(source, conv, dest, conv);
  ASSERT_EQ(edges.size(), 2u);

  // TP1 holds all four key-a, four key-b and four value heads: 96 features per
  // state row and three rows per slot, so one slot is 576 bytes and every
  // component starts 64 bytes into each row.
  const uint64_t source_slot_bytes = 576;
  const uint64_t source_state_stride = 192;
  const std::array<uint64_t, 3> source_component_offsets = {0, 64, 128};
  const BufferDirectoryEntry source_entry =
      make_entry(/*buffer_id=*/41,
                 /*resource_count=*/4,
                 source_slot_bytes,
                 /*units_per_resource=*/1);
  PeerCacheView local = make_view(source, conv, source_entry);
  local.head_runs = {{0, 16, 3, source_state_stride},
                     {source_component_offsets[1], 16, 3, source_state_stride},
                     {source_component_offsets[2], 16, 3, source_state_stride}};
  local.unit_stride_bytes = source_slot_bytes;

  // TP2 holds half the heads, which halves the feature dimension with them: 48
  // features per state row, one slot 288 bytes, components 32 bytes apart.
  const uint64_t remote_slot_bytes = 288;
  const uint64_t remote_state_stride = 96;
  const std::array<uint64_t, 3> remote_component_offsets = {0, 32, 64};
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/42,
                 /*resource_count=*/4,
                 remote_slot_bytes,
                 /*units_per_resource=*/1);
  PeerCacheView remote = make_view(dest, conv, remote_entry);
  remote.head_runs = {
      {0, 16, 3, remote_state_stride},
      {remote_component_offsets[1], 16, 3, remote_state_stride},
      {remote_component_offsets[2], 16, 3, remote_state_stride}};
  remote.unit_stride_bytes = remote_slot_bytes;

  // Slot 2 through the second head class: destination rank 1 holds global heads
  // [2, 4), the second half of every component. Inside that rank the half *is*
  // the component's first two heads, so its run offset needs no head shift --
  // while the source rank holds all four heads and therefore reads those two
  // heads 32 bytes into every component. The two peers' offsets have to differ
  // here: that is the head reshard.
  const uint64_t source_head_offset = 32;
  const uint64_t remote_head_offset = 0;
  std::vector<RouteRegion> regions;
  std::string error;
  ASSERT_TRUE(RouteBinder::bind(
      {edges[1]}, /*dst_local_rank=*/1, {2}, local, remote, &regions, &error))
      << error;
  // Three components times three state rows, each a 32 byte run.
  ASSERT_EQ(regions.size(), 9u);
  size_t region = 0;
  for (uint64_t repeat = 0; repeat < 3; ++repeat) {
    for (size_t component = 0; component < source_component_offsets.size();
         ++component) {
      EXPECT_EQ(regions[region].length, 32u);
      EXPECT_EQ(regions[region].local_offset,
                2 * source_slot_bytes + source_component_offsets[component] +
                    source_head_offset + repeat * source_state_stride);
      EXPECT_EQ(regions[region].remote_offset,
                2 * remote_slot_bytes + remote_component_offsets[component] +
                    remote_head_offset + repeat * remote_state_stride);
      ++region;
    }
  }

  // The first head class of the same slot lands on rank 0, and there neither
  // peer shifts inside a component: the source's rank 0 starts at global head 0
  // and the destination's rank 0 starts at its own class start. The bytes are a
  // different set from rank 1's, which is what keeps the two calls apart.
  ASSERT_TRUE(RouteBinder::bind(
      {edges[0]}, /*dst_local_rank=*/0, {2}, local, remote, &regions, &error))
      << error;
  ASSERT_EQ(regions.size(), 9u);
  EXPECT_EQ(regions[0].local_offset, 2 * source_slot_bytes);
  EXPECT_EQ(regions[0].remote_offset, 2 * remote_slot_bytes);
  EXPECT_EQ(regions[2].local_offset,
            2 * source_slot_bytes + source_component_offsets[2]);
  EXPECT_EQ(regions[2].remote_offset,
            2 * remote_slot_bytes + remote_component_offsets[2]);
}

// Two peers whose rows pack different runs disagree about what the bytes after
// a head mean, so binding refuses instead of addressing another component's
// bytes.
TEST(PdRouteTest, BinderRejectsPeersThatDisagreeOnTheHeadRuns) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology conv = make_group(/*global_head_count=*/64,
                                        /*head_bytes=*/16,
                                        /*sequence_scoped=*/true,
                                        /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, conv, decode, conv);
  ASSERT_FALSE(edges.empty());

  const BufferDirectoryEntry entry = make_entry(/*buffer_id=*/11,
                                                /*resource_count=*/8,
                                                /*resource_stride_bytes=*/288,
                                                /*units_per_resource=*/1);
  PeerCacheView local = make_view(prefill, conv, entry);
  local.head_runs = {{0, 16, 1, 0}, {64, 16, 1, 0}, {128, 16, 1, 0}};
  local.unit_stride_bytes = 288;
  PeerCacheView remote = make_view(decode, conv, entry);
  remote.head_runs = local.head_runs;
  remote.unit_stride_bytes = 288;
  std::vector<RouteRegion> regions;
  std::string error;

  PeerCacheView short_row = remote;
  short_row.head_runs.pop_back();
  EXPECT_FALSE(RouteBinder::bind({edges[0]},
                                 edges[0].dst_local_rank,
                                 {0},
                                 local,
                                 short_row,
                                 &regions,
                                 &error));
  EXPECT_NE(error.find("number of head runs"), std::string::npos) << error;

  PeerCacheView other_repeat = remote;
  other_repeat.head_runs[1].repeat_count = 2;
  EXPECT_FALSE(RouteBinder::bind({edges[0]},
                                 edges[0].dst_local_rank,
                                 {0},
                                 local,
                                 other_repeat,
                                 &regions,
                                 &error));
  EXPECT_NE(error.find("how often run"), std::string::npos) << error;

  PeerCacheView no_stride = remote;
  no_stride.unit_stride_bytes = 0;
  EXPECT_FALSE(RouteBinder::bind({edges[0]},
                                 edges[0].dst_local_rank,
                                 {0},
                                 local,
                                 no_stride,
                                 &regions,
                                 &error));
  EXPECT_NE(error.find("sub-unit stride"), std::string::npos) << error;
}

namespace {

// Tiny stand-in for the real tensors: the same routing and layout, but small
// enough to compare every byte in the test process.
constexpr int32_t kMockHeadBytes = 4;
constexpr int32_t kMockTokens = 2;
constexpr uint64_t kMockRows = 8;
constexpr uint64_t kMockRowBytes =
    static_cast<uint64_t>(kMockTokens) * kMockHeadBytes;
constexpr uint64_t kMockBufferBytes = kMockRows * kMockRowBytes;

uint8_t mock_pattern(int32_t rank, uint64_t row, uint64_t offset_in_row) {
  return static_cast<uint8_t>((rank * 37 + static_cast<int32_t>(row) * 11 +
                               static_cast<int32_t>(offset_in_row) * 7) %
                                  251 +
                              1);
}

// Copies the route into the destination buffers exactly like the transport
// would, then returns the number of bytes that do not match an independently
// derived expectation.
int32_t run_mock_transfer(bool shift_remote_rows) {
  const KvTopology prefill = prefill_topology(kMockTokens);
  const KvTopology decode = decode_topology(kMockTokens);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kMockHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, mla, decode, mla);
  EXPECT_EQ(edges.size(), 4u);

  const BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/100,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/200,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  if (shift_remote_rows) {
    remote_entry.explicit_offsets = true;
  }
  PeerCacheView local = make_view(prefill, mla, local_entry);
  PeerCacheView remote = make_view(decode, mla, remote_entry);
  if (shift_remote_rows) {
    remote.row_offsets.reserve(kMockRows);
    for (uint64_t row = 0; row < kMockRows; ++row) {
      remote.row_offsets.emplace_back((row + 1) * kMockRowBytes);
    }
  }

  std::vector<std::vector<uint8_t>> source(
      32, std::vector<uint8_t>(kMockBufferBytes, 0));
  for (int32_t rank = 0; rank < 32; ++rank) {
    for (uint64_t row = 0; row < kMockRows; ++row) {
      for (uint64_t offset = 0; offset < kMockRowBytes; ++offset) {
        source[rank][row * kMockRowBytes + offset] =
            mock_pattern(rank, row, offset);
      }
    }
  }
  std::vector<std::vector<uint8_t>> destination(
      8, std::vector<uint8_t>(kMockBufferBytes, 0));

  // One source slice at a time, carrying the blocks that its writer holds and
  // the destination rank that owns the matching destination slice needs. The
  // writer of source slice t is cp rank t at tp 0, i.e. local rank 8 * t, and
  // the destination slice is t % 2 = tp rank t % 2.
  for (int32_t src_slice = 0; src_slice < 4; ++src_slice) {
    const int32_t src_local = src_slice * 8;
    const int32_t dst_local = src_slice % 2;
    std::vector<int64_t> blocks;
    for (int64_t block = 0; block < 8; ++block) {
      if (block % 4 == src_slice && block % 2 == dst_local) {
        blocks.emplace_back(block);
      }
    }
    EXPECT_FALSE(blocks.empty());
    for (int32_t dst_dp = 0; dst_dp < 4; ++dst_dp) {
      std::vector<RouteRegion> regions;
      std::string error;
      EXPECT_TRUE(RouteBinder::bind(
          edges, dst_local, blocks, local, remote, &regions, &error))
          << error;
      const size_t dst_global = static_cast<size_t>(dst_dp * 2 + dst_local);
      for (const RouteRegion& region : regions) {
        EXPECT_LE(region.local_offset + region.length, kMockBufferBytes);
        EXPECT_LE(region.remote_offset + region.length, kMockBufferBytes);
        std::memcpy(
            destination[dst_global].data() + region.remote_offset,
            source[static_cast<size_t>(src_local)].data() + region.local_offset,
            region.length);
      }
    }
  }

  int32_t mismatches = 0;
  for (int32_t dst_dp = 0; dst_dp < 4; ++dst_dp) {
    for (int32_t dst_local = 0; dst_local < 2; ++dst_local) {
      const size_t dst_global = static_cast<size_t>(dst_dp * 2 + dst_local);
      for (int64_t row = 1; row < static_cast<int64_t>(kMockRows); ++row) {
        // Pool row `row` of this rank holds the canonical block that its slice
        // owns at position row `row - 1`; row 0 is the reserved padding row and
        // never carries a block.
        const int64_t block = (row - 1) * 2 + dst_local % 2;
        const bool carries_data = block < 8;
        for (int64_t unit = 0; unit < kMockTokens; ++unit) {
          for (int32_t byte = 0; byte < kMockHeadBytes; ++byte) {
            const uint64_t offset_in_row =
                static_cast<uint64_t>(unit) * kMockHeadBytes + byte;
            const uint8_t actual =
                destination[dst_global]
                           [static_cast<size_t>(row) * kMockRowBytes +
                            offset_in_row];
            // The expectation is derived independently of the route table: the
            // block's writer is cp rank (block % src_split) at tp 0, i.e. local
            // rank 8 * (block % src_split), and it keeps the block at the pool
            // row of that position, which is one past the position row.
            const uint8_t expected =
                carries_data ? mock_pattern(static_cast<int32_t>(block % 4) * 8,
                                            block / 4 + 1,
                                            offset_in_row)
                             : 0;
            if (actual != expected) {
              ++mismatches;
            }
          }
        }
      }
    }
  }
  return mismatches;
}

// The byte-level contract of the pair above: every canonical block has to
// arrive from the rank that holds its shard and land in the destination's
// replica row. The expectation is derived independently of the route table, so
// a wrong row on either side moves bytes without breaking any structural
// invariant.
//
// ``wrong_source_rank`` seeds every source rank with its neighbour's pattern:
// the route still binds legal regions (nothing structural changes, and the
// binder's own bounds guard stays quiet), but the bytes that land in the
// destination are no longer the ones the contract names -- which is what the
// "is discriminating" test needs.  Shifting the *remote* row offsets instead
// would be caught by that guard and turn the test into a check of the guard.
int32_t run_mock_index_transfer(bool wrong_source_rank) {
  const KvTopology prefill = prefill_topology(kMockTokens);
  const KvTopology decode = decode_topology(kMockTokens);
  const GroupTopology indexer = make_group(/*global_head_count=*/1,
                                           kMockHeadBytes,
                                           /*sequence_scoped=*/false,
                                           /*full_sequence_replica=*/true);
  const std::vector<RouteEdge> edges =
      build_edges(prefill, indexer, decode, indexer);
  EXPECT_EQ(edges.size(), 8u);

  const BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/100,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/200,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  const PeerCacheView local = make_view(prefill, indexer, local_entry);
  const PeerCacheView remote = make_view(decode, indexer, remote_entry);

  std::vector<std::vector<uint8_t>> source(
      32, std::vector<uint8_t>(kMockBufferBytes, 0));
  for (int32_t rank = 0; rank < 32; ++rank) {
    for (uint64_t row = 0; row < kMockRows; ++row) {
      for (uint64_t offset = 0; offset < kMockRowBytes; ++offset) {
        source[rank][row * kMockRowBytes + offset] =
            mock_pattern(wrong_source_rank ? rank + 1 : rank, row, offset);
      }
    }
  }
  std::vector<std::vector<uint8_t>> destination(
      8, std::vector<uint8_t>(kMockBufferBytes, 0));

  // Both destination ranks keep the whole pool, so each source slice ships
  // every canonical block it holds to both of them. Six canonical blocks fit
  // both row spaces: source row block / 4 + 1, destination row block + 2.
  constexpr int64_t kCanonical = 6;
  for (int32_t src_slice = 0; src_slice < 4; ++src_slice) {
    const int32_t src_local = src_slice * 8;
    std::vector<int64_t> blocks;
    for (int64_t block = 0; block < kCanonical; ++block) {
      if (block % 4 == src_slice) {
        blocks.emplace_back(block);
      }
    }
    EXPECT_FALSE(blocks.empty());
    for (int32_t dst_local = 0; dst_local < 2; ++dst_local) {
      std::vector<RouteRegion> regions;
      std::string error;
      EXPECT_TRUE(RouteBinder::bind(
          edges, dst_local, blocks, local, remote, &regions, &error))
          << error;
      for (int32_t dst_dp = 0; dst_dp < 4; ++dst_dp) {
        const size_t dst_global = static_cast<size_t>(dst_dp * 2 + dst_local);
        for (const RouteRegion& region : regions) {
          EXPECT_LE(region.local_offset + region.length, kMockBufferBytes);
          EXPECT_LE(region.remote_offset + region.length, kMockBufferBytes);
          std::memcpy(destination[dst_global].data() + region.remote_offset,
                      source[static_cast<size_t>(src_local)].data() +
                          region.local_offset,
                      region.length);
        }
      }
    }
  }

  int32_t mismatches = 0;
  for (int32_t dst_dp = 0; dst_dp < 4; ++dst_dp) {
    for (int32_t dst_local = 0; dst_local < 2; ++dst_local) {
      const size_t dst_global = static_cast<size_t>(dst_dp * 2 + dst_local);
      for (int64_t block = 0; block < kCanonical; ++block) {
        const uint64_t row = static_cast<uint64_t>(block) + 2;
        for (int64_t unit = 0; unit < kMockTokens; ++unit) {
          for (int32_t byte = 0; byte < kMockHeadBytes; ++byte) {
            const uint64_t offset_in_row =
                static_cast<uint64_t>(unit) * kMockHeadBytes + byte;
            const uint8_t actual =
                destination[dst_global]
                           [static_cast<size_t>(row) * kMockRowBytes +
                            offset_in_row];
            // A canonical block is written by the rank whose DCP slice owns it,
            // which keeps it at the compact pool row of its position.
            const uint8_t expected =
                mock_pattern(static_cast<int32_t>(block % 4) * 8,
                             block / 4 + 1,
                             offset_in_row);
            if (actual != expected) {
              ++mismatches;
            }
          }
        }
      }
    }
  }
  return mismatches;
}

TEST(PdRouteTest, MockIndexTransferCopiesCanonicalBlocksByteForByte) {
  EXPECT_EQ(run_mock_index_transfer(/*wrong_source_rank=*/false), 0);
}

TEST(PdRouteTest, MockIndexTransferIsDiscriminating) {
  // Seeding the wrong source pattern must move bytes without breaking any
  // structural invariant, so only the byte check can see it.
  EXPECT_GT(run_mock_index_transfer(/*wrong_source_rank=*/true), 0);
}

TEST(PdRouteTest, MockTransferCopiesCanonicalBlocksByteForByte) {
  EXPECT_EQ(run_mock_transfer(/*shift_remote_rows=*/false), 0);
}

TEST(PdRouteTest, MockTransferIsDiscriminating) {
  // A wrong remote row base moves every byte into the wrong place without
  // breaking any structural invariant, so only the byte check can see it.
  EXPECT_GT(run_mock_transfer(/*shift_remote_rows=*/true), 0);
}

// The replica pair (`kv4kv2`): context parallelism is off on both instances, so
// the indexer family is a whole-sequence replica on every rank of both sides
// and the only heterogeneity left is the kv split (4 -> 2).  The route
// therefore moves whole rows and only re-bases them: canonical block h sits at
// source row `h + S_P` and belongs at destination row `h + S_D`, with no
// ownership split at all.  Every source replica holds identical bytes (that
// invariant is what the end-to-end PD owner check measures), so one seed models
// every rank's buffer and the expectation stays independent of which replica
// the route picked.
int32_t run_mock_replica_transfer(bool wrong_source_row) {
  constexpr int32_t kSrcSplit = 4;
  constexpr int32_t kDstSplit = 2;
  // Rows are `kMockRows` = 8 wide and the source row base is 4, so four
  // canonical blocks fit inside one buffer.
  constexpr int64_t kCanonical = 4;

  const KvTopology prefill = make_topology(/*dp_size=*/1,
                                           /*cp_size=*/1,
                                           /*tp_size=*/4,
                                           /*kv_split_size=*/kSrcSplit,
                                           kMockTokens);
  const KvTopology decode = make_topology(/*dp_size=*/1,
                                          /*cp_size=*/1,
                                          /*tp_size=*/2,
                                          /*kv_split_size=*/kDstSplit,
                                          kMockTokens);
  const GroupTopology indexer = make_group(/*global_head_count=*/1,
                                           kMockHeadBytes,
                                           /*sequence_scoped=*/false,
                                           /*full_sequence_replica=*/true);
  const std::vector<RouteEdge> edges =
      build_edges(prefill, indexer, decode, indexer);
  EXPECT_GT(edges.size(), 0u);

  std::string error;
  EXPECT_TRUE(
      PdRouteTable::validate(edges, prefill, indexer, decode, indexer, &error))
      << error;

  const BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/100,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/200,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  const PeerCacheView local = make_view(prefill, indexer, local_entry);
  const PeerCacheView remote = make_view(decode, indexer, remote_entry);

  std::vector<std::vector<uint8_t>> source(
      static_cast<size_t>(prefill.tp_size * prefill.cp_size),
      std::vector<uint8_t>(kMockBufferBytes, 0));
  for (std::vector<uint8_t>& rank_buffer : source) {
    for (uint64_t row = 0; row < kMockRows; ++row) {
      for (uint64_t offset = 0; offset < kMockRowBytes; ++offset) {
        // Rank-independent on purpose: with the sequence replicated the bytes
        // of a pool row are the same on every source rank.
        rank_buffer[row * kMockRowBytes + offset] =
            mock_pattern(0, row, offset);
      }
    }
  }
  std::vector<std::vector<uint8_t>> destination(
      static_cast<size_t>(decode.tp_size * decode.cp_size),
      std::vector<uint8_t>(kMockBufferBytes, 0));

  std::vector<int64_t> blocks;
  for (int64_t block = 0; block < kCanonical; ++block) {
    blocks.emplace_back(block);
  }
  for (int32_t dst_local = 0; dst_local < decode.tp_size * decode.cp_size;
       ++dst_local) {
    std::vector<RouteRegion> regions;
    EXPECT_TRUE(RouteBinder::bind(
        edges, dst_local, blocks, local, remote, &regions, &error))
        << error;
    for (const RouteRegion& region : regions) {
      EXPECT_LE(region.local_offset + region.length, kMockBufferBytes);
      EXPECT_LE(region.remote_offset + region.length, kMockBufferBytes);
      std::memcpy(destination[static_cast<size_t>(dst_local)].data() +
                      region.remote_offset,
                  source[0].data() + region.local_offset,
                  region.length);
    }
  }

  int32_t mismatches = 0;
  for (int32_t dst_local = 0; dst_local < decode.tp_size * decode.cp_size;
       ++dst_local) {
    for (int64_t block = 0; block < kCanonical; ++block) {
      const uint64_t row = static_cast<uint64_t>(block) + kDstSplit;
      for (int64_t unit = 0; unit < kMockTokens; ++unit) {
        for (int32_t byte = 0; byte < kMockHeadBytes; ++byte) {
          const uint64_t offset_in_row =
              static_cast<uint64_t>(unit) * kMockHeadBytes + byte;
          const uint8_t actual =
              destination[static_cast<size_t>(dst_local)]
                         [static_cast<size_t>(row) * kMockRowBytes +
                          offset_in_row];
          // Derived independently of the route table: a replica keeps canonical
          // block h at `h + S`, so the destination must hold the source's
          // `h + S_P` row and nothing else.
          const uint64_t source_row =
              static_cast<uint64_t>(block) + (wrong_source_row ? 0 : kSrcSplit);
          const uint8_t expected = mock_pattern(0, source_row, offset_in_row);
          if (actual != expected) {
            ++mismatches;
          }
        }
      }
    }
  }
  return mismatches;
}

}  // namespace

TEST(PdRouteTest, MockReplicaTransferCopiesWholeSequenceRows) {
  EXPECT_EQ(run_mock_replica_transfer(/*wrong_source_row=*/false), 0)
      << "a whole-sequence source must ship its `block + S_P` rows";
}

TEST(PdRouteTest, MockReplicaTransferIsDiscriminating) {
  EXPECT_GT(run_mock_replica_transfer(/*wrong_source_row=*/true), 0)
      << "reading the compact `block` row instead of `block + S_P` must differ";
}

TEST(PdRouteTest, BinderRejectsPeersThatDisagreeOnTheCanonicalUnit) {
  const KvTopology prefill = prefill_topology(kMockTokens);
  const KvTopology decode = decode_topology(kMockTokens);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kMockHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, mla, decode, mla);

  const BufferDirectoryEntry local_entry =
      make_entry(/*buffer_id=*/100,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  const BufferDirectoryEntry remote_entry =
      make_entry(/*buffer_id=*/200,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens + 1);
  const PeerCacheView local = make_view(prefill, mla, local_entry);
  const PeerCacheView remote = make_view(decode, mla, remote_entry);

  std::vector<RouteRegion> regions;
  std::string error;
  EXPECT_FALSE(RouteBinder::bind(
      edges, /*dst_local_rank=*/0, {0}, local, remote, &regions, &error));
  EXPECT_FALSE(error.empty());
}

TEST(PdRouteTest, BinderRejectsABlockNobodyCarries) {
  const KvTopology prefill = prefill_topology(kMockTokens);
  const KvTopology decode = decode_topology(kMockTokens);
  const GroupTopology mla = make_group(/*global_head_count=*/1,
                                       kMockHeadBytes,
                                       /*sequence_scoped=*/false,
                                       /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> edges = build_edges(prefill, mla, decode, mla);

  const BufferDirectoryEntry entry =
      make_entry(/*buffer_id=*/100,
                 kMockRows,
                 kMockRowBytes,
                 /*units_per_resource=*/kMockTokens);
  const PeerCacheView local = make_view(prefill, mla, entry);
  const PeerCacheView remote = make_view(decode, mla, entry);

  std::vector<RouteRegion> regions;
  std::string error;
  EXPECT_FALSE(RouteBinder::bind(
      edges, /*dst_local_rank=*/0, {0, 1}, local, remote, &regions, &error));
  EXPECT_FALSE(error.empty());
}

}  // namespace xllm
