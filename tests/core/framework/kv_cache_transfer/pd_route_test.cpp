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

TEST(PdRouteTest, IndexerPoolStaysAFullSequenceReplica) {
  const KvTopology prefill = prefill_topology(kTokensPerBlock);
  const KvTopology decode = decode_topology(kTokensPerBlock);
  const GroupTopology indexer = make_group(
      /*global_head_count=*/1,
      /*head_bytes=*/256,
      /*sequence_scoped=*/false,
      /*full_sequence_replica=*/true);

  const std::vector<RouteEdge> edges =
      build_edges(prefill, indexer, decode, indexer);
  // Every destination rank keeps the whole pool, so one writer fans out to both
  // replicas of the destination group instead of one slice each.
  ASSERT_EQ(edges.size(), 2u);
  EXPECT_EQ(edges[0].src_local_rank, 0);
  EXPECT_EQ(edges[0].dst_local_rank, 0);
  EXPECT_EQ(edges[1].src_local_rank, 0);
  EXPECT_EQ(edges[1].dst_local_rank, 1);
  for (const RouteEdge& edge : edges) {
    EXPECT_EQ(edge.src_slice, 0);
    EXPECT_EQ(edge.dst_slice, 0);
  }

  std::string error;
  EXPECT_TRUE(
      PdRouteTable::validate(edges, prefill, indexer, decode, indexer, &error))
      << error;

  // Without the declaration the same group would split the pool over the DCP
  // group, which is the shape the declaration exists to override.
  const GroupTopology split_indexer =
      make_group(/*global_head_count=*/1,
                 /*head_bytes=*/256,
                 /*sequence_scoped=*/false,
                 /*full_sequence_replica=*/false);
  const std::vector<RouteEdge> split_edges =
      build_edges(prefill, split_indexer, decode, split_indexer);
  EXPECT_EQ(split_edges.size(), 4u);
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

  // local_row = block / 4, remote_row = block / 2, and the whole row moves
  // because the head range covers the single MLA head and the token dimension
  // compresses into one contiguous run.
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
              (block / 4) * static_cast<int64_t>(row_bytes));
    EXPECT_EQ(regions[0].remote_offset,
              (block / 2) * static_cast<int64_t>(row_bytes));
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
  // blocks 0 and 4 are the ones it needs.
  std::vector<RouteRegion> regions;
  std::string error;
  ASSERT_TRUE(RouteBinder::bind(
      edges, /*dst_local_rank=*/0, {0, 4}, local, remote, &regions, &error))
      << error;
  ASSERT_EQ(regions.size(), 2u);
  EXPECT_EQ(regions[0].local_offset, 0u);
  EXPECT_EQ(regions[0].remote_offset, 0u);
  EXPECT_EQ(regions[0].length, row_bytes);
  EXPECT_EQ(regions[1].local_offset, 2 * static_cast<int64_t>(row_bytes));
  EXPECT_EQ(regions[1].remote_offset, static_cast<int64_t>(row_bytes));
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
              remote.row_offsets[static_cast<size_t>(block / 2)]);
    EXPECT_EQ(regions[0].local_offset,
              (block / 4) * static_cast<int64_t>(row_bytes));
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
      for (int64_t row = 0; row < static_cast<int64_t>(kMockRows); ++row) {
        // Row `row` of this rank holds the canonical block that its slice owns
        // at that position: blocks arrive in ascending order, one per row.
        const int64_t block = row * 2 + dst_local % 2;
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
            // rank 8 * (block % src_split), and its row is block / src_split.
            const uint8_t expected =
                carries_data ? mock_pattern(static_cast<int32_t>(block % 4) * 8,
                                            block / 4,
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

}  // namespace

TEST(PdRouteTest, MockTransferCopiesCanonicalBlocksByteForByte) {
  EXPECT_EQ(run_mock_transfer(/*shift_remote_rows=*/false), 0);
}

TEST(PdRouteTest, MockTransferIsDiscriminating) {
  // A wrong remote row base moves every byte into the wrong place without
  // breaking any structural invariant, so only the byte check can see it.
  EXPECT_GT(run_mock_transfer(/*shift_remote_rows=*/true), 0);
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
