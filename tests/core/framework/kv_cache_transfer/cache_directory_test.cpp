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

#include "framework/kv_cache_transfer/cache_directory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache_tensor_role.h"

namespace xllm {

namespace {

// The manifests below are written by hand, following the formulas of
// `cache_layout_builder.cpp` and the field assignments of
// `mooncake_kv_cache_transfer.cpp::register_kv_cache`; they are a convention
// fixture, not builder output. `cache_layout_builder_test.cpp` covers the
// builder itself.
constexpr uint64_t kElementBytes = 2;
constexpr int64_t kHeadDim = 8;
constexpr int32_t kKeyRole = static_cast<int32_t>(KVCacheTensorRole::KEY);
constexpr int32_t kIndexRole = static_cast<int32_t>(KVCacheTensorRole::INDEX);
constexpr int32_t kSsmRole = static_cast<int32_t>(KVCacheTensorRole::SSM);
constexpr int32_t kConvRole = static_cast<int32_t>(KVCacheTensorRole::CONV);
constexpr int32_t kLinearGroup = 1;

LogicalSpan make_span(const std::string& logical_tensor,
                      uint64_t logical_offset,
                      uint64_t physical_offset,
                      uint64_t bytes_per_region,
                      uint64_t repeat_count,
                      uint64_t logical_stride,
                      uint64_t physical_stride,
                      int32_t owner_tp_rank) {
  LogicalSpan span;
  span.logical_tensor = logical_tensor;
  span.logical_offset_bytes = logical_offset;
  span.physical_offset_bytes = physical_offset;
  span.bytes_per_region = bytes_per_region;
  span.repeat_count = repeat_count;
  span.logical_stride_bytes = logical_stride;
  span.physical_stride_bytes = physical_stride;
  span.owner_tp_rank = owner_tp_rank;
  return span;
}

// Fills the physical fields the same way register_kv_cache does: contiguous
// strides, one resource every `rows_per_resource` physical rows.
void set_geometry(CacheTensorManifest* tensor,
                  const std::vector<int64_t>& shape,
                  uint64_t element_bytes,
                  uint64_t rows_per_resource) {
  std::vector<int64_t> stride(shape.size(), 1);
  uint64_t elements = 1;
  for (size_t reverse = shape.size(); reverse > 0; --reverse) {
    const size_t index = reverse - 1;
    stride[index] = static_cast<int64_t>(elements);
    elements *= static_cast<uint64_t>(shape[index]);
  }
  tensor->shape = shape;
  tensor->stride = std::move(stride);
  tensor->element_bytes = element_bytes;
  tensor->scalar_type = 0;
  tensor->contiguous = true;
  tensor->storage_offset_bytes = 0;
  tensor->physical_rows_per_resource = rows_per_resource;
  tensor->resource_count = static_cast<uint64_t>(shape[0]) / rows_per_resource;
  tensor->resource_stride_bytes = static_cast<uint64_t>(tensor->stride[0]) *
                                  element_bytes * rows_per_resource;
  tensor->buffer_bytes = tensor->resource_count * tensor->resource_stride_bytes;
}

void set_coordinates(WorkerCacheLayoutManifest* manifest,
                     int32_t tp_rank,
                     int32_t tp_size,
                     int32_t cp_rank,
                     int32_t cp_size,
                     int32_t kv_split_size) {
  manifest->coordinates.dp_rank = 0;
  manifest->coordinates.dp_size = 1;
  manifest->coordinates.tp_rank = tp_rank;
  manifest->coordinates.tp_size = tp_size;
  manifest->coordinates.cp_rank = cp_rank;
  manifest->coordinates.cp_size = cp_size;
  // The runtime publishes its DCP rank here. Nothing in the directory reads it:
  // the sequence slice of a rank is derived from the topology (see
  // KvLayoutIndex::slice_of).
  manifest->coordinates.kv_split_rank =
      (cp_rank * tp_size + tp_rank) % kv_split_size;
  manifest->coordinates.kv_split_size = kv_split_size;
  manifest->layout_family = "token_head_dim";
  manifest->backend = "npu";
  manifest->fingerprint = "fixture";
}

// Mirrors describe_attention_heads: one span per local head inside a
// token-major [rows, tokens, local_heads, head_dim] cache tensor.
WorkerCacheLayoutManifest make_attention_manifest(int32_t tp_rank,
                                                  int32_t tp_size,
                                                  int32_t cp_rank,
                                                  int32_t cp_size,
                                                  int32_t kv_split_size,
                                                  int32_t global_heads,
                                                  int64_t tokens_per_block,
                                                  int64_t rows) {
  const bool sharded = global_heads >= tp_size;
  const int32_t local_heads = sharded ? global_heads / tp_size : 1;
  const int32_t replica_count = sharded ? 1 : tp_size / global_heads;
  const int32_t first_global =
      sharded ? tp_rank * local_heads : tp_rank / replica_count;
  const uint64_t head_bytes = static_cast<uint64_t>(kHeadDim) * kElementBytes;

  WorkerCacheLayoutManifest manifest;
  set_coordinates(&manifest, tp_rank, tp_size, cp_rank, cp_size, kv_split_size);
  CacheTensorManifest tensor;
  tensor.role = kKeyRole;
  tensor.group_id = 0;
  tensor.layer_id = 0;
  tensor.mooncake_buffer_id = 3;
  tensor.block_token_capacity = static_cast<uint64_t>(tokens_per_block);
  set_geometry(&tensor,
               {rows, tokens_per_block, local_heads, kHeadDim},
               kElementBytes,
               /*rows_per_resource=*/1);

  LogicalShardDescriptor descriptor;
  descriptor.kind =
      sharded ? LogicalShardKind::SHARDED : LogicalShardKind::REPLICATED;
  descriptor.resource_scope = CacheResourceScope::BLOCK;
  descriptor.spans.reserve(static_cast<size_t>(local_heads));
  for (int32_t local_head = 0; local_head < local_heads; ++local_head) {
    const int32_t global_head = first_global + local_head;
    descriptor.spans.emplace_back(make_span(
        /*logical_tensor=*/"KEY",
        /*logical_offset=*/static_cast<uint64_t>(global_head) * head_bytes,
        /*physical_offset=*/static_cast<uint64_t>(local_head) * head_bytes,
        /*bytes_per_region=*/head_bytes,
        /*repeat_count=*/static_cast<uint64_t>(tokens_per_block),
        /*logical_stride=*/static_cast<uint64_t>(global_heads) * head_bytes,
        /*physical_stride=*/static_cast<uint64_t>(local_heads) * head_bytes,
        /*owner_tp_rank=*/sharded ? tp_rank : global_head * replica_count));
  }
  tensor.shard = std::move(descriptor);
  manifest.tensors.emplace_back(std::move(tensor));
  return manifest;
}

// Mirrors describe_replicated_tensor: the whole cache resource is one span.
WorkerCacheLayoutManifest make_replicated_manifest(
    int32_t tp_rank,
    int32_t tp_size,
    int32_t cp_rank,
    int32_t cp_size,
    int32_t kv_split_size,
    int32_t role,
    int32_t group_id,
    const std::vector<int64_t>& shape,
    uint64_t rows_per_resource,
    int64_t block_token_capacity) {
  WorkerCacheLayoutManifest manifest;
  set_coordinates(&manifest, tp_rank, tp_size, cp_rank, cp_size, kv_split_size);
  CacheTensorManifest tensor;
  tensor.role = role;
  tensor.group_id = group_id;
  tensor.layer_id = 0;
  tensor.mooncake_buffer_id = 7;
  tensor.block_token_capacity = static_cast<uint64_t>(block_token_capacity);
  set_geometry(&tensor, shape, kElementBytes, rows_per_resource);

  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::REPLICATED;
  descriptor.resource_scope = CacheResourceScope::BLOCK;
  descriptor.spans.emplace_back(
      make_span(/*logical_tensor=*/"KEY",
                /*logical_offset=*/0,
                /*physical_offset=*/0,
                /*bytes_per_region=*/tensor.resource_stride_bytes,
                /*repeat_count=*/1,
                /*logical_stride=*/0,
                /*physical_stride=*/0,
                /*owner_tp_rank=*/0));
  tensor.shard = std::move(descriptor);
  manifest.tensors.emplace_back(std::move(tensor));
  return manifest;
}

CacheTensorDeclaration make_declaration(int32_t role,
                                        int32_t group_id,
                                        int32_t cp_size,
                                        int32_t tp_size,
                                        int32_t kv_split_size,
                                        int64_t tokens_per_block,
                                        int32_t global_head_count,
                                        uint64_t head_bytes,
                                        bool sequence_scoped,
                                        bool full_sequence_replica) {
  CacheTensorDeclaration declaration;
  declaration.cache_namespace = CacheNamespace::MAIN;
  declaration.role = role;
  declaration.group_id = group_id;
  declaration.topology.dp_size = 1;
  declaration.topology.cp_size = cp_size;
  declaration.topology.tp_size = tp_size;
  declaration.topology.kv_split_size = kv_split_size;
  declaration.topology.tokens_per_block =
      static_cast<int32_t>(tokens_per_block);
  declaration.group.global_head_count = global_head_count;
  declaration.group.head_bytes = head_bytes;
  declaration.group.sequence_scoped = sequence_scoped;
  declaration.group.full_sequence_replica = full_sequence_replica;
  return declaration;
}

bool describe(const WorkerCacheLayoutManifest& manifest,
              const std::vector<CacheTensorDeclaration>& declarations,
              PeerDirectory* directory,
              std::string* error) {
  static const std::vector<CacheRowBases> kBases;
  return PeerDirectory::describe(
      manifest, declarations, kBases, directory, error);
}

// 6513 rows of 128 tokens times the MLA latent width is exactly the row count
// the production DCP4 instance reports, and 6513 == 26052 / 4 is what makes the
// KV-split interpretation of that window testable here.
TEST(PeerDirectoryTest, DescribesMlaLatentAcrossTheKvSplit) {
  const WorkerCacheLayoutManifest manifest = make_replicated_manifest(
      /*tp_rank=*/3,
      /*tp_size=*/8,
      /*cp_rank=*/2,
      /*cp_size=*/4,
      /*kv_split_size=*/4,
      kKeyRole,
      /*group_id=*/0,
      /*shape=*/{6513, 128, 1, 576},
      /*rows_per_resource=*/1,
      /*block_token_capacity=*/128);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/4,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/1,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  ASSERT_TRUE(describe(manifest, declarations, &directory, &error)) << error;
  ASSERT_EQ(directory.size(), 1U);
  const PeerCacheView* view =
      directory.find(CacheNamespace::MAIN, 0, kKeyRole, 0);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->entry.buffer_id, 7U);
  EXPECT_EQ(view->entry.resource_count, 6513U);
  EXPECT_EQ(view->entry.resource_stride_bytes, 147456U);
  EXPECT_EQ(view->entry.units_per_resource, 128U);
  EXPECT_FALSE(view->entry.explicit_offsets);
  EXPECT_TRUE(view->row_offsets.empty());
  // 147456 / 128 tokens == 576 values * 2 bytes: one MLA latent head.
  EXPECT_EQ(view->group.head_bytes, 1152U);
  EXPECT_EQ(view->group.global_head_count, 1);
  EXPECT_EQ(directory.find(CacheNamespace::MAIN, 0, kIndexRole, 0), nullptr);

  KvRedundancy redundancy;
  ASSERT_TRUE(
      KvRedundancy::derive(view->topology, view->group, &redundancy, &error))
      << error;
  EXPECT_EQ(redundancy.split(), 4);
  EXPECT_EQ(redundancy.replica_count(), 8);
  EXPECT_EQ(redundancy.local_head_count(), 1);
}

TEST(PeerDirectoryTest, DescribesThePackedIndexerPool) {
  // [26052, 128, 1, 257] is the measured production shape: 257 = 128 + 128 + 1
  // packed kPool values per token, and 26052 = 6513 * 4 canonical blocks.
  const WorkerCacheLayoutManifest manifest = make_replicated_manifest(
      /*tp_rank=*/5,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      kIndexRole,
      /*group_id=*/0,
      /*shape=*/{26052, 128, 1, 257},
      /*rows_per_resource=*/1,
      /*block_token_capacity=*/128);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kIndexRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/1,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/true)};
  PeerDirectory directory;
  std::string error;

  ASSERT_TRUE(describe(manifest, declarations, &directory, &error)) << error;
  const PeerCacheView* view =
      directory.find(CacheNamespace::MAIN, 0, kIndexRole, 0);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->entry.resource_count, 26052U);
  // 257 * 2 bytes per token, and the whole sequence on every rank.
  EXPECT_EQ(view->group.head_bytes, 514U);
  KvRedundancy redundancy;
  ASSERT_TRUE(
      KvRedundancy::derive(view->topology, view->group, &redundancy, &error))
      << error;
  EXPECT_EQ(redundancy.split(), 1);
  EXPECT_TRUE(redundancy.full_sequence_replica());
}

TEST(PeerDirectoryTest, DescribesShardedKvHeads) {
  // 32 global heads over 8 TP ranks and 4 CP ranks: four local heads per rank,
  // one head class per rank, and the sequence split across the CP ranks.
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/2,
      /*tp_size=*/8,
      /*cp_rank=*/1,
      /*cp_size=*/4,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/4,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  ASSERT_TRUE(describe(manifest, declarations, &directory, &error)) << error;
  const PeerCacheView* view =
      directory.find(CacheNamespace::MAIN, 0, kKeyRole, 0);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->entry.resource_count, 6513U);
  EXPECT_EQ(view->entry.resource_stride_bytes, 8192U);
  EXPECT_EQ(view->entry.units_per_resource, 128U);
  EXPECT_EQ(view->group.head_bytes, 16U);
  KvRedundancy redundancy;
  ASSERT_TRUE(
      KvRedundancy::derive(view->topology, view->group, &redundancy, &error))
      << error;
  EXPECT_EQ(redundancy.local_head_count(), 4);
  EXPECT_EQ(redundancy.head_class_count(), 8);
  EXPECT_EQ(redundancy.split(), 4);
}

TEST(PeerDirectoryTest, DescribesCheckpointedSsmSlots) {
  // SSM state is sequence scoped: the resource is one slot holding four
  // checkpointed rows, and the four value heads are split over two TP ranks.
  WorkerCacheLayoutManifest manifest;
  set_coordinates(&manifest,
                  /*tp_rank=*/1,
                  /*tp_size=*/2,
                  /*cp_rank=*/0,
                  /*cp_size=*/1,
                  /*kv_split_size=*/4);
  CacheTensorManifest tensor;
  tensor.role = kSsmRole;
  tensor.group_id = kLinearGroup;
  tensor.mooncake_buffer_id = 11;
  tensor.block_token_capacity = 128;
  set_geometry(&tensor,
               /*shape=*/{8, 2, 3, 4},
               kElementBytes,
               /*rows_per_resource=*/4);
  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::SHARDED;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  // describe_ssm: head stride 3 * 4 values, repeat over the four checkpoint
  // rows, logical stride over all four value heads.
  descriptor.spans.emplace_back(make_span("SSM",
                                          /*logical_offset=*/2 * 24,
                                          /*physical_offset=*/0,
                                          /*bytes_per_region=*/24,
                                          /*repeat_count=*/4,
                                          /*logical_stride=*/4 * 24,
                                          /*physical_stride=*/48,
                                          /*owner_tp_rank=*/1));
  descriptor.spans.emplace_back(make_span("SSM",
                                          /*logical_offset=*/3 * 24,
                                          /*physical_offset=*/24,
                                          /*bytes_per_region=*/24,
                                          /*repeat_count=*/4,
                                          /*logical_stride=*/4 * 24,
                                          /*physical_stride=*/48,
                                          /*owner_tp_rank=*/1));
  tensor.shard = std::move(descriptor);
  manifest.tensors.emplace_back(std::move(tensor));

  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kSsmRole,
                       kLinearGroup,
                       /*cp_size=*/1,
                       /*tp_size=*/2,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/4,
                       /*head_bytes=*/24,
                       /*sequence_scoped=*/true,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  ASSERT_TRUE(describe(manifest, declarations, &directory, &error)) << error;
  const PeerCacheView* view =
      directory.find(CacheNamespace::MAIN, 0, kSsmRole, kLinearGroup);
  ASSERT_NE(view, nullptr);
  EXPECT_EQ(view->entry.resource_count, 2U);
  EXPECT_EQ(view->entry.resource_stride_bytes, 192U);
  EXPECT_EQ(view->entry.units_per_resource, 4U);
  EXPECT_EQ(view->group.head_bytes, 24U);
  EXPECT_TRUE(view->group.sequence_scoped);
}

TEST(PeerDirectoryTest, DescribesPageMappedRowsThroughExplicitBases) {
  WorkerCacheLayoutManifest manifest = make_replicated_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/1,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/1,
      kKeyRole,
      /*group_id=*/0,
      /*shape=*/{4, 128, 1, 576},
      /*rows_per_resource=*/1,
      /*block_token_capacity=*/128);
  manifest.tensors[0].explicit_resource_offsets = true;
  manifest.tensors[0].mooncake_buffer_id = 0;
  manifest.tensors[0].buffer_bytes = 1U << 20;

  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/1,
                       /*kv_split_size=*/1,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/1,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  CacheRowBases bases;
  bases.role = kKeyRole;
  bases.group_id = 0;
  bases.row_offsets = {4096, 8192, 16384, 32768};
  const std::vector<CacheRowBases> row_bases = {bases};
  PeerDirectory directory;
  std::string error;

  ASSERT_TRUE(PeerDirectory::describe(
      manifest, declarations, row_bases, &directory, &error))
      << error;
  const PeerCacheView* view =
      directory.find(CacheNamespace::MAIN, 0, kKeyRole, 0);
  ASSERT_NE(view, nullptr);
  EXPECT_TRUE(view->entry.explicit_offsets);
  EXPECT_EQ(view->row_offsets, bases.row_offsets);
  EXPECT_EQ(view->entry.buffer_bytes, 1U << 20);
}

TEST(PeerDirectoryTest, RejectsCompositeCacheGroups) {
  // describe_conv packs conv_key_a, conv_key_b and conv_value into one row, and
  // the canonical route carries one head interval per edge.
  WorkerCacheLayoutManifest manifest;
  set_coordinates(&manifest,
                  /*tp_rank=*/1,
                  /*tp_size=*/2,
                  /*cp_rank=*/0,
                  /*cp_size=*/1,
                  /*kv_split_size=*/4);
  CacheTensorManifest tensor;
  tensor.role = kConvRole;
  tensor.group_id = kLinearGroup;
  tensor.mooncake_buffer_id = 12;
  tensor.block_token_capacity = 128;
  set_geometry(&tensor,
               /*shape=*/{2, 5, 15},
               kElementBytes,
               /*rows_per_resource=*/1);
  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::COMPOSITE;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.emplace_back(make_span("conv_key_a", 0, 0, 6, 5, 24, 30, 1));
  descriptor.spans.emplace_back(make_span("conv_key_b", 0, 6, 6, 5, 24, 30, 1));
  descriptor.spans.emplace_back(
      make_span("conv_value", 0, 12, 6, 5, 12, 30, 1));
  tensor.shard = std::move(descriptor);
  manifest.tensors.emplace_back(std::move(tensor));

  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kConvRole,
                       kLinearGroup,
                       /*cp_size=*/1,
                       /*tp_size=*/2,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/6,
                       /*head_bytes=*/6,
                       /*sequence_scoped=*/true,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("composite"), std::string::npos) << error;
  EXPECT_EQ(directory.size(), 0U);
}

TEST(PeerDirectoryTest, RejectsATensorTheModelDoesNotDeclare) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kIndexRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/1,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/true)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("does not declare"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsATopologyThatDiffersFromTheCoordinates) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/4,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("coordinates"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsAScopeThatDiffersFromTheDescriptor) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/true,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("scoped"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsABlockSizeThatDiffersFromTheManifest) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/64,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("tokens per block"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsALocalHeadCountThatDiffers) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  // 8 global heads over 8 ranks leave one local head, not four.
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/8,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("spans"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsADeclaredHeadSizeThatDiffers) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/32,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("bytes per head"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsTheDescriptorOfAnotherRank) {
  // The manifest belongs to tp rank 2, which owns heads [8, 12); the same
  // descriptor published by that rank must not claim class 0.
  WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/2,
      /*tp_size=*/8,
      /*cp_rank=*/1,
      /*cp_size=*/4,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  for (LogicalSpan& span : manifest.tensors[0].shard.spans) {
    span.logical_offset_bytes -= 8U * 16U;
    span.owner_tp_rank = 0;
  }
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/4,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("owns class"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, DescribesAWholeResourceReplicaOfAReplicatedHeadGroup) {
  // In an MLA instance every tensor goes through describe_replicated_tensor,
  // so a linear-state slot arrives as a whole-resource span even though the
  // group has eight global heads: each rank keeps exactly one of them, and the
  // rank is what says which.
  WorkerCacheLayoutManifest manifest;
  set_coordinates(&manifest,
                  /*tp_rank=*/3,
                  /*tp_size=*/8,
                  /*cp_rank=*/0,
                  /*cp_size=*/1,
                  /*kv_split_size=*/4);
  CacheTensorManifest tensor;
  tensor.role = kSsmRole;
  tensor.group_id = kLinearGroup;
  tensor.mooncake_buffer_id = 9;
  tensor.block_token_capacity = 128;
  set_geometry(&tensor,
               /*shape=*/{8, 1, 4, 4},
               kElementBytes,
               /*rows_per_resource=*/1);
  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::REPLICATED;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.emplace_back(make_span("SSM",
                                          /*logical_offset=*/0,
                                          /*physical_offset=*/0,
                                          /*bytes_per_region=*/32,
                                          /*repeat_count=*/1,
                                          /*logical_stride=*/0,
                                          /*physical_stride=*/0,
                                          /*owner_tp_rank=*/0));
  tensor.shard = std::move(descriptor);
  manifest.tensors.emplace_back(std::move(tensor));

  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kSsmRole,
                       kLinearGroup,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/8,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/true,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  ASSERT_TRUE(describe(manifest, declarations, &directory, &error)) << error;
  const PeerCacheView* view =
      directory.find(CacheNamespace::MAIN, 0, kSsmRole, kLinearGroup);
  ASSERT_NE(view, nullptr);
  // 32 bytes over one sub-unit is one value head of the linear state.
  EXPECT_EQ(view->group.head_bytes, 32U);
  EXPECT_EQ(view->entry.units_per_resource, 1U);
  EXPECT_EQ(view->entry.resource_count, 8U);
  KvRedundancy redundancy;
  ASSERT_TRUE(
      KvRedundancy::derive(view->topology, view->group, &redundancy, &error))
      << error;
  EXPECT_EQ(redundancy.local_head_count(), 1);
  EXPECT_EQ(redundancy.head_class_count(), 8);
}

TEST(PeerDirectoryTest, RejectsAWholeResourceDescriptorWithSeveralLocalHeads) {
  // The same shape over four TP ranks leaves two local heads per rank, and a
  // whole-resource span cannot say how they are ordered.
  WorkerCacheLayoutManifest manifest;
  set_coordinates(&manifest,
                  /*tp_rank=*/1,
                  /*tp_size=*/4,
                  /*cp_rank=*/0,
                  /*cp_size=*/1,
                  /*kv_split_size=*/4);
  CacheTensorManifest tensor;
  tensor.role = kSsmRole;
  tensor.group_id = kLinearGroup;
  tensor.mooncake_buffer_id = 9;
  tensor.block_token_capacity = 128;
  set_geometry(&tensor,
               /*shape=*/{8, 2, 4, 4},
               kElementBytes,
               /*rows_per_resource=*/1);
  LogicalShardDescriptor descriptor;
  descriptor.kind = LogicalShardKind::REPLICATED;
  descriptor.resource_scope = CacheResourceScope::SEQUENCE;
  descriptor.spans.emplace_back(make_span("SSM",
                                          /*logical_offset=*/0,
                                          /*physical_offset=*/0,
                                          /*bytes_per_region=*/64,
                                          /*repeat_count=*/1,
                                          /*logical_stride=*/0,
                                          /*physical_stride=*/0,
                                          /*owner_tp_rank=*/0));
  tensor.shard = std::move(descriptor);
  manifest.tensors.emplace_back(std::move(tensor));

  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kSsmRole,
                       kLinearGroup,
                       /*cp_size=*/1,
                       /*tp_size=*/4,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/8,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/true,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("one span per local head"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsALayoutThatIsNotTokenMajor) {
  // A head-major tensor strides over the tokens inside one head, which the
  // binder's per-head addressing does not model.
  WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  for (LogicalSpan& span : manifest.tensors[0].shard.spans) {
    span.physical_stride_bytes = 16;
  }
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("stride"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsPageMappedRowsWithoutBases) {
  WorkerCacheLayoutManifest manifest = make_replicated_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/1,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/1,
      kKeyRole,
      /*group_id=*/0,
      /*shape=*/{4, 128, 1, 576},
      /*rows_per_resource=*/1,
      /*block_token_capacity=*/128);
  manifest.tensors[0].explicit_resource_offsets = true;
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/1,
                       /*kv_split_size=*/1,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/1,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(describe(manifest, declarations, &directory, &error));
  EXPECT_NE(error.find("explicit"), std::string::npos) << error;
}

TEST(PeerDirectoryTest, RejectsBasesForAStridedTensor) {
  const WorkerCacheLayoutManifest manifest = make_attention_manifest(
      /*tp_rank=*/0,
      /*tp_size=*/8,
      /*cp_rank=*/0,
      /*cp_size=*/1,
      /*kv_split_size=*/4,
      /*global_heads=*/32,
      /*tokens_per_block=*/128,
      /*rows=*/6513);
  const std::vector<CacheTensorDeclaration> declarations = {
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/8,
                       /*kv_split_size=*/4,
                       /*tokens_per_block=*/128,
                       /*global_head_count=*/32,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false)};
  CacheRowBases bases;
  bases.role = kKeyRole;
  bases.group_id = 0;
  bases.row_offsets = {0, 1, 2};
  const std::vector<CacheRowBases> row_bases = {bases};
  PeerDirectory directory;
  std::string error;

  EXPECT_FALSE(PeerDirectory::describe(
      manifest, declarations, row_bases, &directory, &error));
  EXPECT_NE(error.find("by resource stride"), std::string::npos) << error;
}

// The adapter is the seam to the binder, so one case walks a whole route:
// 32 source ranks (CP4 x TP8) over 2 heads filling one destination rank that
// holds both heads and the whole sequence. The source slice is the cp rank and
// the writers sit at tp 0 of each cp rank. Every canonical block has to arrive
// exactly once, and the regions of one block have to tile its destination row.
TEST(PeerDirectoryTest, RoutesCanonicalBlocksBetweenPublishedLayouts) {
  constexpr int32_t kSourceCp = 4;
  constexpr int32_t kSourceTp = 8;
  constexpr int32_t kSourceSplit = 4;
  constexpr int32_t kGlobalHeads = 2;
  constexpr int64_t kTokensPerBlock = 4;

  const CacheTensorDeclaration source_declaration =
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/kSourceCp,
                       /*tp_size=*/kSourceTp,
                       /*kv_split_size=*/kSourceSplit,
                       /*tokens_per_block=*/kTokensPerBlock,
                       /*global_head_count=*/kGlobalHeads,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false);
  const CacheTensorDeclaration destination_declaration =
      make_declaration(kKeyRole,
                       /*group_id=*/0,
                       /*cp_size=*/1,
                       /*tp_size=*/1,
                       /*kv_split_size=*/1,
                       /*tokens_per_block=*/kTokensPerBlock,
                       /*global_head_count=*/kGlobalHeads,
                       /*head_bytes=*/0,
                       /*sequence_scoped=*/false,
                       /*full_sequence_replica=*/false);

  std::vector<PeerCacheView> sources;
  sources.reserve(static_cast<size_t>(kSourceCp * kSourceTp));
  for (int32_t cp_rank = 0; cp_rank < kSourceCp; ++cp_rank) {
    for (int32_t tp_rank = 0; tp_rank < kSourceTp; ++tp_rank) {
      // One source row per rank: four canonical blocks over a 4-way split.
      const WorkerCacheLayoutManifest manifest =
          make_attention_manifest(tp_rank,
                                  kSourceTp,
                                  cp_rank,
                                  kSourceCp,
                                  kSourceSplit,
                                  kGlobalHeads,
                                  kTokensPerBlock,
                                  /*rows=*/1);
      PeerDirectory directory;
      std::string error;
      ASSERT_TRUE(describe(manifest, {source_declaration}, &directory, &error))
          << error;
      const PeerCacheView* view =
          directory.find(CacheNamespace::MAIN, 0, kKeyRole, 0);
      ASSERT_NE(view, nullptr);
      sources.emplace_back(*view);
    }
  }

  const WorkerCacheLayoutManifest destination_manifest =
      make_attention_manifest(
          /*tp_rank=*/0,
          /*tp_size=*/1,
          /*cp_rank=*/0,
          /*cp_size=*/1,
          /*kv_split_size=*/1,
          kGlobalHeads,
          kTokensPerBlock,
          /*rows=*/4);
  PeerDirectory destination;
  std::string error;
  ASSERT_TRUE(describe(
      destination_manifest, {destination_declaration}, &destination, &error))
      << error;
  const PeerCacheView* remote =
      destination.find(CacheNamespace::MAIN, 0, kKeyRole, 0);
  ASSERT_NE(remote, nullptr);

  std::vector<RouteEdge> edges;
  ASSERT_TRUE(PdRouteTable::build(sources[0].topology,
                                  sources[0].group,
                                  remote->topology,
                                  remote->group,
                                  &edges,
                                  &error))
      << error;
  ASSERT_TRUE(PdRouteTable::validate(edges,
                                     sources[0].topology,
                                     sources[0].group,
                                     remote->topology,
                                     remote->group,
                                     &error))
      << error;

  KvRedundancy source_redundancy;
  ASSERT_TRUE(KvRedundancy::derive(
      sources[0].topology, sources[0].group, &source_redundancy, &error))
      << error;
  const KvLayoutIndex source_index(sources[0].topology, source_redundancy);
  const std::vector<int64_t> canonical_blocks = {0, 1, 2, 3};
  uint64_t written_bytes = 0;
  for (int32_t cp_rank = 0; cp_rank < kSourceCp; ++cp_rank) {
    for (int32_t tp_rank = 0; tp_rank < kSourceTp; ++tp_rank) {
      const size_t src_rank =
          static_cast<size_t>(cp_rank * kSourceTp + tp_rank);
      // The view knows the rank it was published by, so a mixed edge table is
      // safe to pass.
      ASSERT_EQ(sources[src_rank].local_rank, static_cast<int32_t>(src_rank));
      const int32_t slice = source_index.slice_of(cp_rank, tp_rank);
      const int32_t head_class = source_index.head_class_of(tp_rank);
      int32_t writer = -1;
      ASSERT_TRUE(
          source_index.writer_of(/*dp_rank=*/0, head_class, slice, &writer));
      if (writer != static_cast<int32_t>(src_rank)) {
        continue;  // only the replica-0 rank of a (head class, slice) writes
      }
      std::vector<int64_t> owned;
      for (int64_t block : canonical_blocks) {
        if (block % kSourceSplit == slice) {
          owned.emplace_back(block);
        }
      }
      ASSERT_FALSE(owned.empty());
      std::vector<RouteRegion> regions;
      ASSERT_TRUE(RouteBinder::bind(edges,
                                    /*dst_local_rank=*/0,
                                    owned,
                                    sources[src_rank],
                                    *remote,
                                    &regions,
                                    &error))
          << error;
      for (const RouteRegion& region : regions) {
        EXPECT_EQ(region.local_buffer_id, sources[src_rank].entry.buffer_id);
        EXPECT_EQ(region.remote_buffer_id, remote->entry.buffer_id);
        written_bytes += region.length;
      }
    }
  }
  // Each of the four destination rows is covered exactly once: 2 head classes
  // times 4 canonical blocks, 4 tokens * 2 heads * 16 bytes per block.
  EXPECT_EQ(written_bytes, 4U * 4U * 2U * 16U);
  EXPECT_EQ(remote->entry.resource_stride_bytes, 128U);

  // Edges of another source rank are skipped, so a pair handed the wrong source
  // view fails coverage instead of writing the wrong bytes.
  RouteEdge foreign;
  foreign.src_local_rank = 8;  // cp 1, tp 0: another writer of slice 1
  foreign.dst_local_rank = 0;
  foreign.head_begin = 0;
  foreign.head_end = 1;
  foreign.src_slice = 1;
  foreign.dst_slice = 0;
  std::vector<RouteRegion> regions;
  EXPECT_FALSE(RouteBinder::bind(std::vector<RouteEdge>{foreign},
                                 /*dst_local_rank=*/0,
                                 /*canonical_blocks=*/{1},
                                 sources[0],
                                 *remote,
                                 &regions,
                                 &error));
  EXPECT_NE(error.find("not carried by any edge"), std::string::npos) << error;

  // Binding a destination view that belongs to another rank is refused
  // outright: the remote offsets would address a buffer this call cannot reach.
  PeerCacheView wrong_rank = *remote;
  wrong_rank.local_rank = 1;
  EXPECT_FALSE(RouteBinder::bind(edges,
                                 /*dst_local_rank=*/0,
                                 /*canonical_blocks=*/{1},
                                 sources[1],
                                 wrong_rank,
                                 &regions,
                                 &error));
  EXPECT_NE(error.find("belongs to rank 1"), std::string::npos) << error;
}

}  // namespace

}  // namespace xllm
