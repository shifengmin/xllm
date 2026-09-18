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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache/cache_layout_builder.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache_transfer/cache_directory.h"

namespace xllm {

namespace {

// End to end host integration test: real cache tensors -> real
// describe_cache_tensor -> manifest -> PeerDirectory -> PdRouteTable ->
// RouteBinder -> memcpy, with every destination byte checked against the
// canonical content of the resource it must hold.
//
// The transport is replaced by memcpy and the device by host vectors, which is
// the strongest verification available while GLM 5.3 flash does not support PD
// disaggregation (handover §7.4). The expected content of a destination byte is
// built from the destination's own descriptor (the layout authority) and the
// canonical identity of the row (the model), never from the bind arithmetic, so
// a byte that lands at the wrong (resource, head, sub-unit, offset) cannot
// match.

constexpr uint8_t kPoison = 0xEE;
constexpr int64_t kTokensPerBlock = 128;
constexpr int64_t kCanonicalBlocks = 8;
constexpr int64_t kSlots = 8;
constexpr int64_t kMlaValues = 576;    // kv_lora_rank + rope
constexpr int64_t kIndexValues = 257;  // packed kPool: 128 + 128 + 1
constexpr int64_t kHeadDim = 128;
constexpr int64_t kSsmKeyDim = 4;
constexpr int64_t kSsmValueDim = 4;
constexpr int32_t kKvGroup = 0;
constexpr int32_t kLinearGroup = 1;
constexpr int32_t kIndexHeads = 1;  // the indexer cache is a single shared head
constexpr int32_t kSsmHeads = 8;

// One peer's parallel geometry.
struct SideSpec {
  int32_t cp_size = 1;
  int32_t tp_size = 1;
  int32_t kv_split_size = 1;
  const char* name = "";
};

// One cache tensor family. `canonical` is the number of canonical blocks for a
// block-scoped group and the number of sequence slots for a scoped one.
struct RoleSpec {
  int32_t role = 0;
  int32_t group_id = 0;
  int32_t global_heads = 1;
  bool sequence_scoped = false;
  bool full_sequence_replica = false;
  int64_t canonical = 0;
};

// One rank's published cache plus the host memory behind it.
struct RankCache {
  WorkerCacheLayoutManifest manifest;
  std::vector<CacheTensorDeclaration> declarations;
  PeerDirectory directory;
  std::map<uint64_t, std::vector<uint8_t>> buffers;
  int32_t cp_rank = 0;
  int32_t tp_rank = 0;
  int32_t local_rank = 0;
  bool ok = false;
  std::string error;
};

struct Scenario {
  SideSpec source;
  SideSpec destination;
  bool enable_mla = true;
};

std::vector<RoleSpec> role_specs(bool enable_mla) {
  RoleSpec key;
  key.role = static_cast<int32_t>(KVCacheTensorRole::KEY);
  key.group_id = kKvGroup;
  // An MLA latent cache keeps one logical head; a plain attention cache keeps
  // one per KV head.
  key.global_heads = enable_mla ? 1 : 8;
  key.canonical = kCanonicalBlocks;

  RoleSpec index;
  index.role = static_cast<int32_t>(KVCacheTensorRole::INDEX);
  index.group_id = kKvGroup;
  index.global_heads = kIndexHeads;
  // The indexer pool has to see the whole sequence on every rank: its top-k
  // reads historical gate and valid values.
  index.full_sequence_replica = true;
  index.canonical = kCanonicalBlocks;

  RoleSpec ssm;
  ssm.role = static_cast<int32_t>(KVCacheTensorRole::SSM);
  ssm.group_id = kLinearGroup;
  ssm.global_heads = kSsmHeads;
  ssm.sequence_scoped = true;
  ssm.canonical = kSlots;

  // An MLA instance describes every tensor as a whole resource, so the
  // convolution state of the linear layers arrives as a packed single-head row
  // (see PeerDirectoryTest.RejectsCompositeCacheGroups for the composite
  // descriptor a non-MLA instance produces instead).
  if (!enable_mla) {
    return {key, index, ssm};
  }
  RoleSpec conv;
  conv.role = static_cast<int32_t>(KVCacheTensorRole::CONV);
  conv.group_id = kLinearGroup;
  conv.global_heads = kSsmHeads;
  conv.sequence_scoped = true;
  conv.canonical = kSlots;
  return {key, index, ssm, conv};
}

CacheTensorDeclaration make_declaration(const SideSpec& side,
                                        const RoleSpec& role) {
  CacheTensorDeclaration declaration;
  declaration.cache_namespace = CacheNamespace::MAIN;
  declaration.role = role.role;
  declaration.group_id = role.group_id;
  declaration.topology.dp_size = 1;
  declaration.topology.cp_size = side.cp_size;
  declaration.topology.tp_size = side.tp_size;
  declaration.topology.kv_split_size = side.kv_split_size;
  declaration.topology.tokens_per_block = static_cast<int32_t>(kTokensPerBlock);
  declaration.group.global_head_count = role.global_heads;
  declaration.group.head_bytes = 0;
  declaration.group.sequence_scoped = role.sequence_scoped;
  declaration.group.full_sequence_replica = role.full_sequence_replica;
  return declaration;
}

void set_coordinates(WorkerCacheLayoutManifest* manifest,
                     const SideSpec& side,
                     int32_t cp_rank,
                     int32_t tp_rank) {
  manifest->coordinates.dp_rank = 0;
  manifest->coordinates.dp_size = 1;
  manifest->coordinates.tp_rank = tp_rank;
  manifest->coordinates.tp_size = side.tp_size;
  manifest->coordinates.cp_rank = cp_rank;
  manifest->coordinates.cp_size = side.cp_size;
  manifest->coordinates.kv_split_rank =
      side.kv_split_size <= side.cp_size &&
              side.cp_size % side.kv_split_size == 0
          ? cp_rank / (side.cp_size / side.kv_split_size)
          : cp_rank * side.tp_size + tp_rank;
  manifest->coordinates.kv_split_size = side.kv_split_size;
  manifest->layout_family = "token_head_dim";
  manifest->backend = "npu";
  manifest->fingerprint = "integration";
}

// Shapes follow the production allocator: the MLA latent and the indexer pool
// are whole resources, an attention group is [rows, tokens, local heads,
// head_dim], and the linear state is [slots, local heads, key, value].
std::vector<int64_t> role_shape(const RoleSpec& role,
                                int32_t local_heads,
                                int64_t rows,
                                bool enable_mla) {
  if (role.role == static_cast<int32_t>(KVCacheTensorRole::CONV)) {
    // [slot, state, packed key_a | key_b | value] features, one local head.
    return {kSlots, 1, (2 * local_heads + local_heads) * kSsmKeyDim};
  }
  if (role.sequence_scoped) {
    return {kSlots, local_heads, kSsmKeyDim, kSsmValueDim};
  }
  if (role.role == static_cast<int32_t>(KVCacheTensorRole::INDEX)) {
    return {rows, kTokensPerBlock, 1, kIndexValues};
  }
  if (enable_mla) {
    return {rows, kTokensPerBlock, 1, kMlaValues};
  }
  return {rows, kTokensPerBlock, local_heads, kHeadDim};
}

// Mirrors what register_kv_cache publishes for one described tensor.
CacheTensorManifest make_tensor_manifest(const KVCacheTensor& described,
                                         uint64_t rows_per_resource,
                                         uint64_t block_token_capacity,
                                         uint64_t buffer_id) {
  const torch::Tensor& tensor = described.tensor;
  CacheTensorManifest manifest;
  manifest.cache_namespace = CacheNamespace::MAIN;
  manifest.layer_id = 0;
  manifest.role = static_cast<int32_t>(
      static_cast<KVCacheTensorRole::Value>(described.role));
  manifest.group_id = described.group_id;
  manifest.mooncake_buffer_id = buffer_id;
  manifest.scalar_type = static_cast<int32_t>(tensor.scalar_type());
  manifest.element_bytes = static_cast<uint64_t>(tensor.element_size());
  manifest.shape = tensor.sizes().vec();
  manifest.stride = tensor.strides().vec();
  manifest.storage_offset_bytes = 0;
  manifest.contiguous = tensor.is_contiguous();
  manifest.resource_count =
      static_cast<uint64_t>(tensor.size(0)) / rows_per_resource;
  manifest.physical_rows_per_resource = rows_per_resource;
  manifest.resource_stride_bytes = static_cast<uint64_t>(tensor.stride(0)) *
                                   manifest.element_bytes * rows_per_resource;
  manifest.buffer_bytes = static_cast<uint64_t>(tensor.nbytes());
  manifest.block_token_capacity = block_token_capacity;
  manifest.explicit_resource_offsets = false;
  manifest.shard = described.shard_descriptor.value();
  return manifest;
}

const CacheTensorManifest* find_tensor(const RankCache& rank,
                                       const RoleSpec& role) {
  for (const CacheTensorManifest& tensor : rank.manifest.tensors) {
    if (tensor.role == role.role && tensor.group_id == role.group_id) {
      return &tensor;
    }
  }
  return nullptr;
}

// Sub-units one cache resource holds, read from the manifest alone.
uint64_t resource_units(const CacheTensorManifest& tensor) {
  if (tensor.shard.resource_scope == CacheResourceScope::SEQUENCE) {
    return tensor.physical_rows_per_resource;
  }
  return tensor.block_token_capacity;
}

// Bytes of one head, derived without the adapter.
uint64_t head_bytes_of(const CacheTensorManifest& tensor,
                       uint64_t units,
                       int32_t local_heads) {
  const LogicalSpan& span = tensor.shard.spans.front();
  const bool whole_resource =
      tensor.shard.spans.size() == 1 && span.repeat_count == 1 &&
      span.bytes_per_region == tensor.resource_stride_bytes;
  if (whole_resource && units > 0 && local_heads > 0) {
    return tensor.resource_stride_bytes /
           (units * static_cast<uint64_t>(local_heads));
  }
  return span.bytes_per_region;
}

// The canonical content of one cache resource. It depends only on the logical
// identity (resource, head, sub-unit, byte inside the head), so a byte that
// ends up anywhere else cannot match its expectation.
uint8_t content_byte(int32_t group_id,
                     int64_t resource,
                     int32_t head,
                     uint64_t unit,
                     uint64_t offset) {
  uint64_t mixed = 0x9E3779B97F4A7C15ULL;
  mixed = (mixed ^ (static_cast<uint64_t>(group_id) + 1U)) * 0x100000001B3ULL;
  mixed = (mixed ^ (static_cast<uint64_t>(resource) + 0x51ED2701ULL)) *
          0xBF58476D1CE4E5B9ULL;
  mixed =
      (mixed ^ ((static_cast<uint64_t>(head) + 1U) * 0x94D049BB133111EBULL)) *
      0xBF58476D1CE4E5B9ULL;
  mixed =
      (mixed ^ ((unit + 1U) * 0x2545F4914F6CDD1DULL)) * 0xBF58476D1CE4E5B9ULL;
  mixed = (mixed ^ (offset + 0x9E3779B9ULL)) * 0xBF58476D1CE4E5B9ULL;
  const uint8_t value = static_cast<uint8_t>((mixed >> 33) & 0xFFU);
  return value == kPoison ? static_cast<uint8_t>(value ^ 0x5AU) : value;
}

// Fills one tensor's whole buffer with the content of the rows this rank owns.
// The descriptor decides where a byte lives; the model decides which canonical
// resource a physical row is.
std::vector<uint8_t> canonical_content(const CacheTensorManifest& tensor,
                                       int32_t first_head,
                                       int32_t local_heads,
                                       int32_t split,
                                       int32_t slice) {
  const uint64_t units = resource_units(tensor);
  const uint64_t head_bytes = head_bytes_of(tensor, units, local_heads);
  const uint64_t stride = tensor.resource_stride_bytes;
  std::vector<uint8_t> buffer(
      static_cast<size_t>(tensor.resource_count * stride), kPoison);
  if (units == 0 || local_heads <= 0 || head_bytes == 0) {
    return buffer;
  }
  const uint64_t unit_bytes = static_cast<uint64_t>(local_heads) * head_bytes;
  const CanonicalBlock canonical(static_cast<int32_t>(units), split);

  for (uint64_t row = 0; row < tensor.resource_count; ++row) {
    const int64_t resource =
        canonical.canonical_of_row(static_cast<int64_t>(row), slice);
    const uint64_t base = row * stride;
    for (const LogicalSpan& span : tensor.shard.spans) {
      const bool whole_resource = tensor.shard.spans.size() == 1 &&
                                  span.repeat_count == 1 &&
                                  span.bytes_per_region == stride;
      if (whole_resource) {
        // The whole resource is this rank's local heads, head by head.
        for (uint64_t byte = 0; byte < span.bytes_per_region; ++byte) {
          const uint64_t unit = byte / unit_bytes;
          const uint64_t within = byte % unit_bytes;
          const int32_t head =
              first_head + static_cast<int32_t>(within / head_bytes);
          buffer[static_cast<size_t>(base + byte)] = content_byte(
              tensor.group_id, resource, head, unit, within % head_bytes);
        }
        continue;
      }
      const int32_t head = static_cast<int32_t>(span.logical_offset_bytes /
                                                span.bytes_per_region);
      for (uint64_t repeat = 0; repeat < span.repeat_count; ++repeat) {
        const uint64_t at = base + span.physical_offset_bytes +
                            repeat * span.physical_stride_bytes;
        for (uint64_t offset = 0; offset < span.bytes_per_region; ++offset) {
          buffer[static_cast<size_t>(at + offset)] =
              content_byte(tensor.group_id, resource, head, repeat, offset);
        }
      }
    }
  }
  return buffer;
}

// Builds one rank: real tensors, real descriptors, the manifest the rank would
// publish, the directory it would derive, and the host buffer the transport
// would read and write.
void make_rank(const SideSpec& side,
               int32_t cp_rank,
               int32_t tp_rank,
               bool enable_mla,
               RankCache* rank) {
  rank->cp_rank = cp_rank;
  rank->tp_rank = tp_rank;
  rank->local_rank = cp_rank * side.tp_size + tp_rank;
  set_coordinates(&rank->manifest, side, cp_rank, tp_rank);

  uint64_t buffer_id = 0;
  for (const RoleSpec& role : role_specs(enable_mla)) {
    const CacheTensorDeclaration declaration = make_declaration(side, role);
    KvRedundancy redundancy;
    std::string error;
    ASSERT_TRUE(KvRedundancy::derive(
        declaration.topology, declaration.group, &redundancy, &error))
        << error;
    const KvLayoutIndex index(declaration.topology, redundancy);
    const int32_t local_heads = redundancy.local_head_count();
    const int32_t split = redundancy.split();
    const int64_t rows = role.canonical / split;
    ASSERT_GT(rows, 0);
    ASSERT_EQ(role.canonical % split, 0);

    CacheTensorLayoutContext context;
    context.tp_rank = tp_rank;
    context.tp_size = side.tp_size;
    context.block_token_capacity = kTokensPerBlock;
    context.kv_head_count = role.sequence_scoped ? 0 : role.global_heads;
    context.index_head_count = kIndexValues / 2;
    context.linear_key_head_count =
        role.role == static_cast<int32_t>(KVCacheTensorRole::CONV)
            ? role.global_heads
            : 0;
    context.linear_value_head_count = role.global_heads;
    context.linear_key_head_dim = kSsmKeyDim;
    context.linear_ssm_checkpoint_stride = 1;
    context.enable_mla = enable_mla;
    context.head_major_layout = false;

    KVCacheTensor tensor{
        static_cast<KVCacheTensorRole::Value>(role.role),
        torch::zeros(role_shape(role, local_heads, rows, enable_mla),
                     torch::kBFloat16),
        role.group_id,
        role.sequence_scoped};
    ASSERT_TRUE(describe_cache_tensor(context, &tensor, &error)) << error;
    ASSERT_TRUE(tensor.shard_descriptor.has_value());

    rank->manifest.tensors.emplace_back(make_tensor_manifest(
        tensor, /*rows_per_resource=*/1, kTokensPerBlock, buffer_id));
    rank->declarations.emplace_back(declaration);
    const CacheTensorManifest& manifest_tensor = rank->manifest.tensors.back();
    rank->buffers[buffer_id] =
        canonical_content(manifest_tensor,
                          index.head_begin(index.head_class_of(tp_rank)),
                          local_heads,
                          split,
                          index.slice_of(cp_rank, tp_rank));
    ++buffer_id;
  }

  ASSERT_TRUE(PeerDirectory::describe(rank->manifest,
                                      rank->declarations,
                                      /*row_bases=*/{},
                                      &rank->directory,
                                      &rank->error))
      << rank->error;
  rank->ok = true;
}

// Moves every canonical resource the two peers have to exchange for one role.
// The byte ranges come from the binder; the bytes move with memcpy.
uint64_t transfer_role(std::vector<RankCache>& sources,
                       std::vector<RankCache>& destinations,
                       const SideSpec& source_side,
                       const SideSpec& destination_side,
                       const RoleSpec& role) {
  const CacheTensorDeclaration source_declaration =
      make_declaration(source_side, role);
  const CacheTensorDeclaration destination_declaration =
      make_declaration(destination_side, role);
  KvRedundancy source_redundancy;
  KvRedundancy destination_redundancy;
  std::string error;
  EXPECT_TRUE(KvRedundancy::derive(source_declaration.topology,
                                   source_declaration.group,
                                   &source_redundancy,
                                   &error))
      << error;
  EXPECT_TRUE(KvRedundancy::derive(destination_declaration.topology,
                                   destination_declaration.group,
                                   &destination_redundancy,
                                   &error))
      << error;
  const KvLayoutIndex source_index(source_declaration.topology,
                                   source_redundancy);
  const KvLayoutIndex destination_index(destination_declaration.topology,
                                        destination_redundancy);

  std::vector<RouteEdge> edges;
  EXPECT_TRUE(PdRouteTable::build(source_declaration.topology,
                                  source_declaration.group,
                                  destination_declaration.topology,
                                  destination_declaration.group,
                                  &edges,
                                  &error))
      << error;
  EXPECT_TRUE(PdRouteTable::validate(edges,
                                     source_declaration.topology,
                                     source_declaration.group,
                                     destination_declaration.topology,
                                     destination_declaration.group,
                                     &error))
      << error;

  // Only the replica-0 rank of each (head class, slice) writes, so the writers
  // are exactly head classes times slices.
  std::vector<int32_t> writers;
  for (const RouteEdge& edge : edges) {
    if (std::find(writers.begin(), writers.end(), edge.src_local_rank) ==
        writers.end()) {
      writers.emplace_back(edge.src_local_rank);
    }
  }
  EXPECT_EQ(writers.size(),
            static_cast<size_t>(source_redundancy.head_class_count() *
                                source_redundancy.split()));

  const int32_t source_split = source_redundancy.split();
  const int32_t destination_split = destination_redundancy.split();
  uint64_t moved = 0;
  for (RankCache& source : sources) {
    const int32_t head_class = source_index.head_class_of(source.tp_rank);
    const int32_t slice = source_index.slice_of(source.cp_rank, source.tp_rank);
    int32_t writer = 0;
    if (!source_index.writer_of(/*dp_rank=*/0, head_class, slice, &writer)) {
      continue;
    }
    if (writer != source.local_rank) {
      continue;
    }
    const PeerCacheView* local = source.directory.find(
        CacheNamespace::MAIN, 0, role.role, role.group_id);
    if (local == nullptr) {
      ADD_FAILURE() << "the source rank has no view for role " << role.role;
      continue;
    }
    for (RankCache& destination : destinations) {
      const int32_t destination_slice =
          destination_index.slice_of(destination.cp_rank, destination.tp_rank);
      std::vector<int64_t> resources;
      for (int64_t resource = 0; resource < role.canonical; ++resource) {
        if (resource % source_split == slice &&
            resource % destination_split == destination_slice) {
          resources.emplace_back(resource);
        }
      }
      if (resources.empty()) {
        continue;
      }
      bool connected = false;
      for (const RouteEdge& edge : edges) {
        if (edge.src_local_rank == source.local_rank &&
            edge.dst_local_rank == destination.local_rank) {
          connected = true;
          break;
        }
      }
      if (!connected) {
        continue;
      }
      const PeerCacheView* remote = destination.directory.find(
          CacheNamespace::MAIN, 0, role.role, role.group_id);
      if (remote == nullptr) {
        ADD_FAILURE() << "the destination rank has no view for role "
                      << role.role;
        continue;
      }
      std::vector<RouteRegion> regions;
      EXPECT_TRUE(RouteBinder::bind(edges,
                                    destination.local_rank,
                                    resources,
                                    *local,
                                    *remote,
                                    &regions,
                                    &error))
          << error;
      for (const RouteRegion& region : regions) {
        const std::vector<uint8_t>& from =
            source.buffers.at(region.local_buffer_id);
        std::vector<uint8_t>& to =
            destination.buffers.at(region.remote_buffer_id);
        EXPECT_LE(region.local_offset + region.length, from.size());
        EXPECT_LE(region.remote_offset + region.length, to.size());
        if (region.local_offset + region.length > from.size() ||
            region.remote_offset + region.length > to.size()) {
          continue;
        }
        std::memcpy(to.data() + region.remote_offset,
                    from.data() + region.local_offset,
                    region.length);
        moved += region.length;
      }
    }
  }
  return moved;
}

// Compares every destination byte against the canonical content of the resource
// its row holds.
void verify_role(const std::vector<RankCache>& destinations,
                 const SideSpec& destination_side,
                 const RoleSpec& role) {
  const CacheTensorDeclaration declaration =
      make_declaration(destination_side, role);
  KvRedundancy redundancy;
  std::string error;
  ASSERT_TRUE(KvRedundancy::derive(
      declaration.topology, declaration.group, &redundancy, &error))
      << error;
  const KvLayoutIndex index(declaration.topology, redundancy);
  const int32_t local_heads = redundancy.local_head_count();
  const int32_t split = redundancy.split();

  for (const RankCache& rank : destinations) {
    const CacheTensorManifest* tensor = find_tensor(rank, role);
    ASSERT_NE(tensor, nullptr);
    const PeerCacheView* view =
        rank.directory.find(CacheNamespace::MAIN, 0, role.role, role.group_id);
    ASSERT_NE(view, nullptr);
    // The adapter's head size has to agree with the descriptor arithmetic.
    EXPECT_EQ(view->group.head_bytes,
              head_bytes_of(*tensor, resource_units(*tensor), local_heads));

    const int32_t head_class = index.head_class_of(rank.tp_rank);
    const std::vector<uint8_t> expected =
        canonical_content(*tensor,
                          index.head_begin(head_class),
                          local_heads,
                          split,
                          index.slice_of(rank.cp_rank, rank.tp_rank));
    const std::vector<uint8_t>& actual = rank.buffers.at(view->entry.buffer_id);
    ASSERT_EQ(actual.size(), expected.size());

    size_t mismatches = 0;
    std::string first;
    for (size_t byte = 0; byte < actual.size(); ++byte) {
      if (actual[byte] == expected[byte]) {
        continue;
      }
      ++mismatches;
      if (first.empty()) {
        first = "row " +
                std::to_string(byte / view->entry.resource_stride_bytes) +
                " byte " +
                std::to_string(byte % view->entry.resource_stride_bytes) +
                " expected " + std::to_string(expected[byte]) + " got " +
                std::to_string(actual[byte]);
      }
    }
    EXPECT_EQ(mismatches, 0U)
        << destination_side.name << " cp " << rank.cp_rank << " tp "
        << rank.tp_rank << " role " << role.role << ": " << mismatches
        << " bytes differ, first at " << first;
  }
}

void run_scenario(const Scenario& scenario) {
  std::vector<RankCache> sources;
  std::vector<RankCache> destinations;
  for (int32_t cp_rank = 0; cp_rank < scenario.source.cp_size; ++cp_rank) {
    for (int32_t tp_rank = 0; tp_rank < scenario.source.tp_size; ++tp_rank) {
      RankCache rank;
      make_rank(scenario.source, cp_rank, tp_rank, scenario.enable_mla, &rank);
      ASSERT_TRUE(rank.ok) << rank.error;
      sources.emplace_back(std::move(rank));
    }
  }
  for (int32_t cp_rank = 0; cp_rank < scenario.destination.cp_size; ++cp_rank) {
    for (int32_t tp_rank = 0; tp_rank < scenario.destination.tp_size;
         ++tp_rank) {
      RankCache rank;
      make_rank(
          scenario.destination, cp_rank, tp_rank, scenario.enable_mla, &rank);
      ASSERT_TRUE(rank.ok) << rank.error;
      destinations.emplace_back(std::move(rank));
    }
  }

  for (const RoleSpec& role : role_specs(scenario.enable_mla)) {
    const uint64_t moved = transfer_role(
        sources, destinations, scenario.source, scenario.destination, role);
    std::printf(
        "[%s cp%d tp%d kv%d -> %s cp%d tp%d kv%d] role %d moved %llu bytes\n",
        scenario.source.name,
        scenario.source.cp_size,
        scenario.source.tp_size,
        scenario.source.kv_split_size,
        scenario.destination.name,
        scenario.destination.cp_size,
        scenario.destination.tp_size,
        scenario.destination.kv_split_size,
        role.role,
        static_cast<unsigned long long>(moved));
    EXPECT_GT(moved, 0U);
    verify_role(destinations, scenario.destination, role);
  }
}

// The degenerate anchor every step has to keep working: both peers split the
// same way, so every canonical block stays on the rank that already holds it.
TEST(PdRouteIntegrationTest, AnchoredEqualSplitReproducesTheSourceLayout) {
  Scenario scenario;
  scenario.source =
      SideSpec{/*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/4, "source"};
  scenario.destination = SideSpec{
      /*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/4, "destination"};
  scenario.enable_mla = true;
  run_scenario(scenario);
}

// The pilot shape the legacy planner cannot express: the source splits the
// sequence four ways while the destination splits it twice, so a destination
// row gathers blocks that were not neighbours on the source.
TEST(PdRouteIntegrationTest, TargetSplitMismatchFoldsFourSlicesIntoTwo) {
  Scenario scenario;
  scenario.source =
      SideSpec{/*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/4, "source"};
  scenario.destination = SideSpec{
      /*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/2, "destination"};
  scenario.enable_mla = true;
  run_scenario(scenario);
}

// The other direction of the same mismatch: the source splits the sequence
// twice and the destination four times, so one source slice fans out into two
// destination slices.
TEST(PdRouteIntegrationTest, ReverseSplitMismatchExpandsTwoSlicesIntoFour) {
  Scenario scenario;
  scenario.source =
      SideSpec{/*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/2, "source"};
  scenario.destination = SideSpec{
      /*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/4, "destination"};
  scenario.enable_mla = true;
  run_scenario(scenario);
}

// A plain attention instance: the heads themselves are sharded, so the route
// has to intersect head classes whose sizes differ between the peers, and the
// indexer pool still has to reach every destination replica.
TEST(PdRouteIntegrationTest, ShardedHeadsReshardAcrossHeadClasses) {
  Scenario scenario;
  scenario.source =
      SideSpec{/*cp_size=*/4, /*tp_size=*/8, /*kv_split_size=*/4, "source"};
  scenario.destination = SideSpec{
      /*cp_size=*/4, /*tp_size=*/4, /*kv_split_size=*/2, "destination"};
  scenario.enable_mla = false;
  run_scenario(scenario);
}

}  // namespace

}  // namespace xllm
