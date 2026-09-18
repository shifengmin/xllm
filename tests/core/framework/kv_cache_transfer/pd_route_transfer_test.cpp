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

#include "framework/kv_cache_transfer/pd_route_transfer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xllm {

namespace {

// Host verification of the unified data-plane entry.
//
// The peers are built by hand -- one physical view per (rank, family) with the
// geometry PeerDirectory would derive -- and the transport is memcpy into host
// vectors. Every scenario is run twice, once from the writer with PUSH and once
// from the reader with PULL, and both results are compared against an
// expectation built from the model (KvLayoutIndex + CanonicalBlock) and the two
// sides' declared geometry. The expectation never looks at a route edge or a
// bound region, so a leg that picks the wrong rank, the wrong slice, the wrong
// head or the wrong sub-unit cannot match.

constexpr int64_t kResources = 8;  // canonical blocks, or sequence slots
constexpr uint64_t kHeadBytes = 16;
constexpr uint64_t kUnits = 4;         // tokens per canonical block
constexpr uint64_t kSequenceRows = 3;  // checkpoint rows per sequence slot
constexpr int32_t kKvGroup = 0;
constexpr int32_t kLinearGroup = 1;
constexpr int32_t kKeyRole = 1;
constexpr int32_t kSsmRole = 4;

struct SideSpec {
  int32_t cp_size = 1;
  int32_t tp_size = 1;
  int32_t kv_split_size = 1;
};

struct FamilySpec {
  int32_t role = kKeyRole;
  int32_t group_id = kKvGroup;
  int32_t global_heads = 1;
  bool sequence_scoped = false;
  int64_t resources = kResources;
};

struct Side {
  SideSpec spec;
  std::vector<FamilySpec> families;
  std::vector<PeerCacheView> views;
  std::vector<std::string> addrs;
};

int32_t local_rank_count(const SideSpec& spec) {
  return spec.cp_size * spec.tp_size;
}

uint64_t units_of(const FamilySpec& family) {
  return family.sequence_scoped ? kSequenceRows : kUnits;
}

// Content depends on the physical identity of the byte only, so a byte that
// lands at the wrong offset of the wrong buffer is a mismatch.
uint8_t content_byte(uint64_t buffer_id, uint64_t offset) {
  return static_cast<uint8_t>((buffer_id * 131 + offset * 17 + 7) % 251);
}

void build_side(const SideSpec& spec,
                const std::vector<FamilySpec>& families,
                uint64_t* next_buffer_id,
                Side* side) {
  side->spec = spec;
  side->families = families;
  const int32_t ranks = local_rank_count(spec);
  side->addrs.clear();
  side->addrs.reserve(static_cast<size_t>(ranks));
  for (int32_t rank = 0; rank < ranks; ++rank) {
    side->addrs.emplace_back("10.0.0." + std::to_string(rank) + ":26000");
  }
  side->views.clear();
  side->views.reserve(static_cast<size_t>(ranks) * families.size());
  for (int32_t rank = 0; rank < ranks; ++rank) {
    for (const FamilySpec& family : families) {
      PeerCacheView view;
      view.topology.dp_size = 1;
      view.topology.cp_size = spec.cp_size;
      view.topology.tp_size = spec.tp_size;
      view.topology.kv_split_size = spec.kv_split_size;
      view.topology.tokens_per_block =
          family.sequence_scoped ? 0 : static_cast<int32_t>(units_of(family));
      view.group.global_head_count = family.global_heads;
      view.group.head_bytes = kHeadBytes;
      view.group.sequence_scoped = family.sequence_scoped;
      view.group.full_sequence_replica = false;

      KvRedundancy redundancy;
      std::string error;
      const bool derived =
          KvRedundancy::derive(view.topology, view.group, &redundancy, &error);
      EXPECT_TRUE(derived) << error;
      if (!derived) {
        return;
      }
      const int32_t split = redundancy.split();
      EXPECT_EQ(family.resources % split, 0);

      view.entry.cache_namespace = CacheNamespace::MAIN;
      view.entry.layer_id = 0;
      view.entry.role = family.role;
      view.entry.group_id = family.group_id;
      view.entry.buffer_id = *next_buffer_id;
      ++*next_buffer_id;
      // One row per canonical block, plus the pool row the block manager
      // reserves for its padding block: the request's `p`-th block is pool row
      // `p + 1`, so a fixture whose canonical positions start at 0 needs one
      // row of room in front of them.
      view.entry.resource_count =
          static_cast<uint64_t>(family.resources / split) + 1;
      view.entry.units_per_resource = units_of(family);
      view.entry.resource_stride_bytes =
          units_of(family) *
          static_cast<uint64_t>(redundancy.local_head_count()) * kHeadBytes;
      view.entry.buffer_bytes =
          view.entry.resource_count * view.entry.resource_stride_bytes;
      view.entry.explicit_offsets = false;
      view.local_rank = rank;
      side->views.emplace_back(std::move(view));
    }
  }
}

const PeerCacheView* find_view(const Side& side,
                               int32_t rank,
                               int32_t role,
                               int32_t group_id) {
  for (const PeerCacheView& view : side.views) {
    if (view.local_rank == rank && view.entry.role == role &&
        view.entry.group_id == group_id) {
      return &view;
    }
  }
  return nullptr;
}

std::vector<PeerCacheView> views_of_rank(const Side& side, int32_t rank) {
  std::vector<PeerCacheView> views;
  for (const PeerCacheView& view : side.views) {
    if (view.local_rank == rank) {
      views.emplace_back(view);
    }
  }
  return views;
}

RoutePeer peer_of(const Side& side) {
  RoutePeer peer;
  peer.addrs = side.addrs;
  peer.views = side.views;
  return peer;
}

std::map<uint64_t, std::vector<uint8_t>> make_initial(const Side& source,
                                                      const Side& destination) {
  std::map<uint64_t, std::vector<uint8_t>> memory;
  for (const Side* side : {&source, &destination}) {
    for (const PeerCacheView& view : side->views) {
      std::vector<uint8_t>& buffer = memory[view.entry.buffer_id];
      buffer.resize(static_cast<size_t>(view.entry.buffer_bytes));
      for (uint64_t offset = 0; offset < view.entry.buffer_bytes; ++offset) {
        buffer[static_cast<size_t>(offset)] =
            content_byte(view.entry.buffer_id, offset);
      }
    }
  }
  return memory;
}

// The transport: PUSH writes the peer's buffer from this rank's, PULL reads the
// peer's buffer into this rank's. Both are the same memcpy seen from opposite
// ends, which is exactly what the two orientations have to encode.
PdRouteTransfer::MoveFn make_move(
    std::map<uint64_t, std::vector<uint8_t>>* memory) {
  return [memory](const std::string& peer_addr,
                  const std::vector<RouteRegion>& regions,
                  RouteOpcode opcode) {
    (void)peer_addr;
    for (const RouteRegion& region : regions) {
      std::vector<uint8_t>& local = memory->at(region.local_buffer_id);
      std::vector<uint8_t>& remote = memory->at(region.remote_buffer_id);
      if (opcode == RouteOpcode::PULL) {
        std::memcpy(local.data() + region.local_offset,
                    remote.data() + region.remote_offset,
                    static_cast<size_t>(region.length));
      } else {
        std::memcpy(remote.data() + region.remote_offset,
                    local.data() + region.local_offset,
                    static_cast<size_t>(region.length));
      }
    }
    return true;
  };
}

void expect_buffers_equal(
    const std::map<uint64_t, std::vector<uint8_t>>& expected,
    const std::map<uint64_t, std::vector<uint8_t>>& actual) {
  EXPECT_EQ(expected.size(), actual.size());
  for (const auto& pair : expected) {
    const auto it = actual.find(pair.first);
    EXPECT_TRUE(it != actual.end()) << "buffer " << pair.first << " is missing";
    if (it == actual.end()) {
      continue;
    }
    EXPECT_EQ(pair.second.size(), it->second.size())
        << "buffer " << pair.first << " changed size";
    if (pair.second.size() != it->second.size()) {
      continue;
    }
    for (size_t offset = 0; offset < pair.second.size(); ++offset) {
      if (pair.second[offset] != it->second[offset]) {
        ADD_FAILURE() << "buffer " << pair.first << " byte " << offset
                      << " holds " << static_cast<int32_t>(it->second[offset])
                      << " instead of "
                      << static_cast<int32_t>(pair.second[offset]);
        break;
      }
    }
  }
}

// The canonical content of every destination byte: which rank holds the block
// (the model's writer), where that rank keeps the head and the sub-unit inside
// its resource, and where the destination keeps them inside its own. Derived
// from KvLayoutIndex and the descriptors alone.
void fill_expected(const Side& source,
                   const Side& destination,
                   const std::vector<int64_t>& canonical_blocks,
                   std::map<uint64_t, std::vector<uint8_t>>* expected) {
  for (const FamilySpec& family : destination.families) {
    const PeerCacheView* source_any =
        find_view(source, 0, family.role, family.group_id);
    const PeerCacheView* destination_any =
        find_view(destination, 0, family.role, family.group_id);
    EXPECT_TRUE(source_any != nullptr && destination_any != nullptr);
    if (source_any == nullptr || destination_any == nullptr) {
      continue;
    }
    KvRedundancy source_redundancy;
    KvRedundancy destination_redundancy;
    std::string error;
    const bool derived = KvRedundancy::derive(source_any->topology,
                                              source_any->group,
                                              &source_redundancy,
                                              &error) &&
                         KvRedundancy::derive(destination_any->topology,
                                              destination_any->group,
                                              &destination_redundancy,
                                              &error);
    EXPECT_TRUE(derived) << error;
    if (!derived) {
      continue;
    }
    const KvLayoutIndex source_index(source_any->topology, source_redundancy);
    const KvLayoutIndex destination_index(destination_any->topology,
                                          destination_redundancy);
    const int32_t source_split = source_redundancy.split();
    const int32_t destination_split = destination_redundancy.split();
    const uint64_t units = destination_any->entry.units_per_resource;
    EXPECT_EQ(units, source_any->entry.units_per_resource);

    for (int32_t rank = 0; rank < local_rank_count(destination.spec); ++rank) {
      const PeerCacheView* destination_view =
          find_view(destination, rank, family.role, family.group_id);
      EXPECT_TRUE(destination_view != nullptr);
      if (destination_view == nullptr) {
        continue;
      }
      const int32_t local_head_count =
          destination_redundancy.local_head_count();
      const int32_t head_begin = destination_index.head_begin(
          destination_index.head_class_of(rank % destination.spec.tp_size));
      const int32_t destination_slice = destination_index.slice_of(
          rank / destination.spec.tp_size, rank % destination.spec.tp_size);

      for (int64_t block : canonical_blocks) {
        if (block % destination_split != destination_slice) {
          continue;
        }
        // The runtime's addressing, restated here so the expectation never
        // reuses a route-side helper: a block-scoped canonical block is a
        // position, and the pool row holding it is that position's logical
        // block one row further along, because row 0 is the reserved padding
        // block. A sequence-scoped canonical id is a slot, and a slot *is* the
        // row.
        const uint64_t destination_row =
            family.sequence_scoped
                ? static_cast<uint64_t>(block)
                : static_cast<uint64_t>(block) /
                          static_cast<uint64_t>(destination_split) +
                      1;
        for (int32_t head_offset = 0; head_offset < local_head_count;
             ++head_offset) {
          const int32_t head = head_begin + head_offset;
          const int32_t source_slice =
              static_cast<int32_t>(block % source_split);
          int32_t writer_rank = -1;
          const bool has_writer = source_index.writer_of(
              /*dp_rank=*/0,
              head / source_redundancy.local_head_count(),
              source_slice,
              &writer_rank);
          EXPECT_TRUE(has_writer);
          if (!has_writer) {
            continue;
          }
          const PeerCacheView* writer_view =
              find_view(source, writer_rank, family.role, family.group_id);
          EXPECT_TRUE(writer_view != nullptr);
          if (writer_view == nullptr) {
            continue;
          }
          const uint64_t source_row =
              family.sequence_scoped
                  ? static_cast<uint64_t>(block)
                  : static_cast<uint64_t>(block) /
                            static_cast<uint64_t>(source_split) +
                        1;
          const uint64_t writer_head_offset =
              static_cast<uint64_t>(
                  head - source_index.head_begin(
                             head / source_redundancy.local_head_count())) *
              kHeadBytes;
          for (uint64_t unit = 0; unit < units; ++unit) {
            const uint64_t source_offset =
                source_row * writer_view->entry.resource_stride_bytes +
                writer_head_offset +
                unit *
                    static_cast<uint64_t>(
                        source_redundancy.local_head_count()) *
                    kHeadBytes;
            const uint64_t destination_offset =
                destination_row *
                    destination_view->entry.resource_stride_bytes +
                static_cast<uint64_t>(head_offset) * kHeadBytes +
                unit * static_cast<uint64_t>(local_head_count) * kHeadBytes;
            std::vector<uint8_t>& buffer =
                expected->at(destination_view->entry.buffer_id);
            for (uint64_t byte = 0; byte < kHeadBytes; ++byte) {
              buffer[static_cast<size_t>(destination_offset + byte)] =
                  content_byte(writer_view->entry.buffer_id,
                               source_offset + byte);
            }
          }
        }
      }
    }
  }
}

// Runs the same request twice -- once as PUSH from every writer-side rank, once
// as PULL from every reader-side rank -- and checks both against the model.
void expect_route_migrates(const Side& source,
                           const Side& destination,
                           const std::vector<int64_t>& canonical_blocks) {
  const std::map<uint64_t, std::vector<uint8_t>> initial =
      make_initial(source, destination);
  std::map<uint64_t, std::vector<uint8_t>> pushed = initial;
  std::map<uint64_t, std::vector<uint8_t>> pulled = initial;

  const RoutePeer destination_peer = peer_of(destination);
  const RoutePeer source_peer = peer_of(source);
  PdRouteCache cache;
  for (int32_t rank = 0; rank < local_rank_count(source.spec); ++rank) {
    std::string error;
    const bool pushed_ok =
        PdRouteTransfer::transfer(&cache,
                                  RouteOpcode::PUSH,
                                  rank,
                                  canonical_blocks,
                                  views_of_rank(source, rank),
                                  destination_peer,
                                  make_move(&pushed),
                                  /*legs=*/nullptr,
                                  &error);
    EXPECT_TRUE(pushed_ok) << "push from rank " << rank << ": " << error;
  }
  // One table covers the whole scenario: the source and destination shapes do
  // not change from rank to rank.
  EXPECT_EQ(cache.size(), static_cast<size_t>(source.families.size()));

  for (int32_t rank = 0; rank < local_rank_count(destination.spec); ++rank) {
    std::string error;
    const bool pulled_ok =
        PdRouteTransfer::transfer(&cache,
                                  RouteOpcode::PULL,
                                  rank,
                                  canonical_blocks,
                                  views_of_rank(destination, rank),
                                  source_peer,
                                  make_move(&pulled),
                                  /*legs=*/nullptr,
                                  &error);
    EXPECT_TRUE(pulled_ok) << "pull into rank " << rank << ": " << error;
  }

  std::map<uint64_t, std::vector<uint8_t>> expected = initial;
  fill_expected(source, destination, canonical_blocks, &expected);
  expect_buffers_equal(expected, pushed);
  expect_buffers_equal(expected, pulled);
  // PUSH and PULL are the same move seen from its two ends; neither may depend
  // on which side initiated it.
  expect_buffers_equal(pushed, pulled);
}

std::vector<int64_t> all_blocks() {
  std::vector<int64_t> blocks;
  blocks.reserve(static_cast<size_t>(kResources));
  for (int64_t block = 0; block < kResources; ++block) {
    blocks.emplace_back(block);
  }
  return blocks;
}

const std::vector<FamilySpec>& mla_families() {
  static const std::vector<FamilySpec> kFamilies = {
      FamilySpec{/*role=*/kKeyRole,
                 /*group_id=*/kKvGroup,
                 /*global_heads=*/1,
                 /*sequence_scoped=*/false,
                 /*resources=*/kResources},
      FamilySpec{/*role=*/kSsmRole,
                 /*group_id=*/kLinearGroup,
                 /*global_heads=*/8,
                 /*sequence_scoped=*/true,
                 /*resources=*/kResources},
  };
  return kFamilies;
}

const std::vector<FamilySpec>& sharded_families() {
  static const std::vector<FamilySpec> kFamilies = {
      FamilySpec{/*role=*/kKeyRole,
                 /*group_id=*/kKvGroup,
                 /*global_heads=*/8,
                 /*sequence_scoped=*/false,
                 /*resources=*/kResources},
  };
  return kFamilies;
}

void build_pair(const SideSpec& source_spec,
                const SideSpec& destination_spec,
                const std::vector<FamilySpec>& families,
                Side* source,
                Side* destination) {
  uint64_t next_buffer_id = 1;
  build_side(source_spec, families, &next_buffer_id, source);
  build_side(destination_spec, families, &next_buffer_id, destination);
}

TEST(PdRouteTransferTest, MigratesAnAnchoredEqualSplitBothWays) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 8, 4},
             mla_families(),
             &source,
             &destination);
  expect_route_migrates(source, destination, all_blocks());
}

TEST(PdRouteTransferTest, FoldsFourSlicesIntoTwoBothWays) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 8, 2},
             mla_families(),
             &source,
             &destination);
  expect_route_migrates(source, destination, all_blocks());
}

TEST(PdRouteTransferTest, ExpandsTwoSlicesIntoFourBothWays) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 2},
             /*destination_spec=*/SideSpec{4, 8, 4},
             mla_families(),
             &source,
             &destination);
  expect_route_migrates(source, destination, all_blocks());
}

TEST(PdRouteTransferTest, IntersectsShardedHeadsAcrossHeadClassesBothWays) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 4, 2},
             sharded_families(),
             &source,
             &destination);
  expect_route_migrates(source, destination, all_blocks());
}

TEST(PdRouteTransferTest, SendsOneLegPerReaderOfAWriterSlice) {
  // The pilot shape: one MLA latent head, cp 4 x tp 8, kv split 4. The writer
  // of (head class 0, slice t) is the rank whose tp is 0 and whose cp is t, and
  // it fills the eight tp replicas of the destination.
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 8, 4},
             {FamilySpec{/*role=*/kKeyRole,
                         /*group_id=*/kKvGroup,
                         /*global_heads=*/1,
                         /*sequence_scoped=*/false,
                         /*resources=*/kResources}},
             &source,
             &destination);

  PdRouteCache cache;
  std::vector<RouteLeg> legs;
  std::string error;
  const RoutePeer peer = peer_of(destination);
  ASSERT_TRUE(PdRouteTransfer::plan(&cache,
                                    RouteOpcode::PUSH,
                                    /*local_rank=*/8,
                                    all_blocks(),
                                    views_of_rank(source, 8),
                                    peer,
                                    &legs,
                                    &error))
      << error;
  ASSERT_EQ(legs.size(), 8u);
  for (size_t index = 0; index < legs.size(); ++index) {
    EXPECT_EQ(legs[index].local_rank, 8);
    EXPECT_EQ(legs[index].peer_local_rank, static_cast<int32_t>(8 + index));
    EXPECT_EQ(legs[index].peer_addr, destination.addrs[8 + index]);
    EXPECT_EQ(legs[index].opcode, RouteOpcode::PUSH);
    EXPECT_FALSE(legs[index].regions.empty());
  }

  // A rank that is a redundant copy of the writer moves nothing, and that is
  // not an error: the copy holds the same bytes already.
  legs.clear();
  ASSERT_TRUE(PdRouteTransfer::plan(&cache,
                                    RouteOpcode::PUSH,
                                    /*local_rank=*/1,
                                    all_blocks(),
                                    views_of_rank(source, 1),
                                    peer,
                                    &legs,
                                    &error))
      << error;
  EXPECT_TRUE(legs.empty());
}

TEST(PdRouteTransferTest, PullsFromTheWriterOfEverySliceItNeeds) {
  // kv 2 -> kv 4: each destination slice takes blocks from the source slice
  // that holds them, and one source slice feeds two destination slices.
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 2},
             /*destination_spec=*/SideSpec{4, 8, 4},
             {FamilySpec{/*role=*/kKeyRole,
                         /*group_id=*/kKvGroup,
                         /*global_heads=*/1,
                         /*sequence_scoped=*/false,
                         /*resources=*/kResources}},
             &source,
             &destination);

  PdRouteCache cache;
  const RoutePeer peer = peer_of(source);
  std::vector<RouteLeg> legs;
  std::string error;
  ASSERT_TRUE(PdRouteTransfer::plan(&cache,
                                    RouteOpcode::PULL,
                                    /*local_rank=*/0,
                                    all_blocks(),
                                    views_of_rank(destination, 0),
                                    peer,
                                    &legs,
                                    &error))
      << error;
  ASSERT_EQ(legs.size(), 1u);
  EXPECT_EQ(legs[0].peer_local_rank, 0);
  EXPECT_EQ(legs[0].opcode, RouteOpcode::PULL);

  // Destination rank 8 holds slice 1, which covers canonical blocks 1 and 5.
  // Their source slice is 1 too, and the writer of (head class 0, slice 1) is
  // the source rank whose cp is 1 * (cp_size / split) = 2, i.e. local rank 16.
  legs.clear();
  ASSERT_TRUE(PdRouteTransfer::plan(&cache,
                                    RouteOpcode::PULL,
                                    /*local_rank=*/8,
                                    all_blocks(),
                                    views_of_rank(destination, 8),
                                    peer,
                                    &legs,
                                    &error))
      << error;
  ASSERT_EQ(legs.size(), 1u);
  EXPECT_EQ(legs[0].peer_local_rank, 16);
}

TEST(PdRouteTransferTest, ReusesOneTableForEveryPeerOfAShape) {
  PdRouteCache cache;
  const KvTopology source_topology{/*dp_size=*/1,
                                   /*cp_size=*/4,
                                   /*tp_size=*/8,
                                   /*kv_split_size=*/4,
                                   /*tokens_per_block=*/128};
  const KvTopology destination_topology{/*dp_size=*/1,
                                        /*cp_size=*/4,
                                        /*tp_size=*/8,
                                        /*kv_split_size=*/2,
                                        /*tokens_per_block=*/128};
  const GroupTopology group{/*global_head_count=*/1,
                            /*head_bytes=*/16,
                            /*sequence_scoped=*/false,
                            /*full_sequence_replica=*/false};
  const std::vector<RouteEdge>* first = nullptr;
  const std::vector<RouteEdge>* second = nullptr;
  std::string error;
  ASSERT_TRUE(cache.find_or_build(
      source_topology, group, destination_topology, group, &first, &error))
      << error;
  ASSERT_TRUE(cache.find_or_build(
      source_topology, group, destination_topology, group, &second, &error))
      << error;
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_EQ(first, second);
  EXPECT_FALSE(first->empty());

  // A different shape is a different table.
  KvTopology other_split = source_topology;
  other_split.kv_split_size = 1;
  const std::vector<RouteEdge>* third = nullptr;
  ASSERT_TRUE(cache.find_or_build(
      other_split, group, destination_topology, group, &third, &error))
      << error;
  EXPECT_EQ(cache.size(), 2u);
  EXPECT_NE(first, third);
}

TEST(PdRouteTransferTest, RejectsARequestThatIsNotAscending) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 8, 4},
             mla_families(),
             &source,
             &destination);
  PdRouteCache cache;
  std::vector<RouteLeg> legs;
  std::string error;
  EXPECT_FALSE(PdRouteTransfer::plan(&cache,
                                     RouteOpcode::PULL,
                                     /*local_rank=*/0,
                                     /*canonical_blocks=*/{3, 1},
                                     views_of_rank(destination, 0),
                                     peer_of(source),
                                     &legs,
                                     &error));
  EXPECT_NE(error.find("ascending"), std::string::npos) << error;

  EXPECT_FALSE(PdRouteTransfer::plan(&cache,
                                     RouteOpcode::PULL,
                                     /*local_rank=*/0,
                                     /*canonical_blocks=*/{-1},
                                     views_of_rank(destination, 0),
                                     peer_of(source),
                                     &legs,
                                     &error));
  EXPECT_NE(error.find("negative"), std::string::npos) << error;
}

TEST(PdRouteTransferTest, RequiresTheConcreteViewOfEveryPeerRank) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 8, 4},
             {FamilySpec{/*role=*/kKeyRole,
                         /*group_id=*/kKvGroup,
                         /*global_heads=*/1,
                         /*sequence_scoped=*/false,
                         /*resources=*/kResources}},
             &source,
             &destination);
  // Drop the view of one writer rank: that writer cannot be enumerated from the
  // table any more, so the pull that needs it has to fail rather than quietly
  // read from whichever rank is left.
  RoutePeer peer = peer_of(source);
  peer.views.erase(std::remove_if(peer.views.begin(),
                                  peer.views.end(),
                                  [](const PeerCacheView& view) {
                                    return view.local_rank == 8;
                                  }),
                   peer.views.end());
  PdRouteCache cache;
  std::vector<RouteLeg> legs;
  std::string error;
  EXPECT_FALSE(PdRouteTransfer::plan(&cache,
                                     RouteOpcode::PULL,
                                     /*local_rank=*/8,
                                     all_blocks(),
                                     views_of_rank(destination, 8),
                                     peer,
                                     &legs,
                                     &error));
  EXPECT_NE(error.find("published no view"), std::string::npos) << error;
}

TEST(PdRouteTransferTest, FailsTheRouteWhenTheTransportFails) {
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 4},
             /*destination_spec=*/SideSpec{4, 8, 4},
             {FamilySpec{/*role=*/kKeyRole,
                         /*group_id=*/kKvGroup,
                         /*global_heads=*/1,
                         /*sequence_scoped=*/false,
                         /*resources=*/kResources}},
             &source,
             &destination);
  std::map<uint64_t, std::vector<uint8_t>> memory =
      make_initial(source, destination);
  int32_t calls = 0;
  PdRouteTransfer::MoveFn failing = [&calls](
                                        const std::string& peer_addr,
                                        const std::vector<RouteRegion>& regions,
                                        RouteOpcode opcode) {
    (void)peer_addr;
    (void)regions;
    (void)opcode;
    ++calls;
    return false;
  };
  PdRouteCache cache;
  std::vector<RouteLeg> legs;
  std::string error;
  EXPECT_FALSE(PdRouteTransfer::transfer(&cache,
                                         RouteOpcode::PUSH,
                                         /*local_rank=*/0,
                                         all_blocks(),
                                         views_of_rank(source, 0),
                                         peer_of(destination),
                                         failing,
                                         &legs,
                                         &error));
  EXPECT_EQ(calls, 1);
  EXPECT_NE(error.find("transport failed"), std::string::npos) << error;
  EXPECT_FALSE(legs.empty());
}

TEST(PdRouteTransferTest, RejectsABlockOfAnotherDestinationSlice) {
  // The caller groups the request's canonical blocks by destination rank; a
  // block of another slice would be written into this rank's buffer and never
  // reach its owner, so the binder has to reject it instead of binding it.
  //
  // A source split narrower than the destination's is what makes such a request
  // expressible: source slice 0 holds canonical blocks 0, 2, 4 and 6, and the
  // destination needs them in two different slices.
  Side source;
  Side destination;
  build_pair(/*source_spec=*/SideSpec{4, 8, 2},
             /*destination_spec=*/SideSpec{4, 8, 4},
             {FamilySpec{/*role=*/kKeyRole,
                         /*group_id=*/kKvGroup,
                         /*global_heads=*/1,
                         /*sequence_scoped=*/false,
                         /*resources=*/kResources}},
             &source,
             &destination);
  const PeerCacheView* local = find_view(source, 0, kKeyRole, kKvGroup);
  const PeerCacheView* remote = find_view(destination, 0, kKeyRole, kKvGroup);
  ASSERT_TRUE(local != nullptr && remote != nullptr);

  PdRouteCache cache;
  const std::vector<RouteEdge>* edges = nullptr;
  std::string error;
  ASSERT_TRUE(cache.find_or_build(local->topology,
                                  local->group,
                                  remote->topology,
                                  remote->group,
                                  &edges,
                                  &error))
      << error;
  std::vector<RouteRegion> regions;
  // Destination rank 0 holds slice 0; block 2 belongs to slice 2.
  EXPECT_FALSE(RouteBinder::bind(*edges,
                                 /*dst_local_rank=*/0,
                                 /*canonical_blocks=*/{2},
                                 *local,
                                 *remote,
                                 &regions,
                                 &error));
  EXPECT_NE(error.find("belongs to destination slice"), std::string::npos)
      << error;
}

TEST(PdRouteTransferTest, FlattensALegIntoAscendingLayerBatches) {
  // Two layers, two peers, and a region per (leg, layer). A buffer belongs to
  // one layer, so the same plan becomes one transport call per (layer, peer),
  // in layer order, because the push loop synchronizes a layer before sending
  // it.
  std::vector<RouteLeg> legs;
  RouteLeg first;
  first.peer_local_rank = 3;
  first.peer_addr = "peer-3";
  first.regions = {RouteRegion{/*local_buffer_id=*/10, 0, 100, 0, 8},
                   RouteRegion{/*local_buffer_id=*/11, 0, 200, 0, 8}};
  RouteLeg second;
  second.peer_local_rank = 5;
  second.peer_addr = "peer-5";
  second.regions = {RouteRegion{/*local_buffer_id=*/10, 8, 300, 0, 4},
                    RouteRegion{/*local_buffer_id=*/20, 0, 400, 0, 4}};
  legs.emplace_back(std::move(first));
  legs.emplace_back(std::move(second));

  const std::unordered_map<uint64_t, int64_t> layer_of_buffer = {
      {10, 0}, {11, 1}, {20, 1}};
  std::vector<RouteLayerBatch> batches;
  std::string error;
  ASSERT_TRUE(flatten_route_for_layers(legs, layer_of_buffer, &batches, &error))
      << error;
  ASSERT_EQ(batches.size(), 4u);
  // Layer 0 first, and inside it the legs in plan order.
  EXPECT_EQ(batches[0].layer_id, 0);
  EXPECT_EQ(batches[0].peer_addr, "peer-3");
  ASSERT_EQ(batches[0].regions.size(), 1u);
  EXPECT_EQ(batches[0].regions[0].local_buffer_id, 10U);
  EXPECT_EQ(batches[0].regions[0].local_offset, 0U);
  EXPECT_EQ(batches[1].layer_id, 0);
  EXPECT_EQ(batches[1].peer_addr, "peer-5");
  EXPECT_EQ(batches[1].regions[0].local_offset, 8U);
  EXPECT_EQ(batches[2].layer_id, 1);
  EXPECT_EQ(batches[2].peer_addr, "peer-3");
  EXPECT_EQ(batches[2].regions[0].local_buffer_id, 11U);
  EXPECT_EQ(batches[3].layer_id, 1);
  EXPECT_EQ(batches[3].peer_addr, "peer-5");
  EXPECT_EQ(batches[3].regions[0].local_buffer_id, 20U);

  // Every region of the plan is emitted exactly once.
  size_t emitted = 0;
  for (const RouteLayerBatch& batch : batches) {
    emitted += batch.regions.size();
  }
  EXPECT_EQ(emitted, 4u);
}

TEST(PdRouteTransferTest, RejectsARegionWhoseBufferHasNoLayer) {
  std::vector<RouteLeg> legs;
  RouteLeg leg;
  leg.peer_addr = "peer-0";
  leg.regions = {RouteRegion{/*local_buffer_id=*/7, 0, 0, 0, 4}};
  legs.emplace_back(std::move(leg));
  std::vector<RouteLayerBatch> batches;
  std::string error;
  EXPECT_FALSE(
      flatten_route_for_layers(legs, /*layer_of_buffer=*/{}, &batches, &error));
  EXPECT_NE(error.find("belongs to no layer"), std::string::npos) << error;
  EXPECT_TRUE(batches.empty());
}

TEST(PdRouteTransferTest, ParsesTheRouteModesAndRejectsATypo) {
  PdRouteMode mode = PdRouteMode::CANONICAL;
  EXPECT_TRUE(parse_pd_route_mode("legacy", &mode));
  EXPECT_EQ(mode, PdRouteMode::LEGACY);
  EXPECT_TRUE(parse_pd_route_mode("canonical", &mode));
  EXPECT_EQ(mode, PdRouteMode::CANONICAL);
  EXPECT_FALSE(parse_pd_route_mode("Canonical", &mode));
  EXPECT_FALSE(parse_pd_route_mode("", &mode));
  EXPECT_FALSE(parse_pd_route_mode("legacy", nullptr));
  EXPECT_STREQ(pd_route_mode_name(PdRouteMode::LEGACY), "legacy");
  EXPECT_STREQ(pd_route_mode_name(PdRouteMode::CANONICAL), "canonical");
}

}  // namespace

}  // namespace xllm
