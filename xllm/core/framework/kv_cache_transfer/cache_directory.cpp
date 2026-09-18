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

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace xllm {

namespace {

// A group whose whole sequence fits in one head, which is the only case a
// whole-resource span can describe.
constexpr int32_t kSingleHead = 1;

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool multiply_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs;
}

std::string tensor_id(const CacheTensorManifest& tensor) {
  return "layer " + std::to_string(tensor.layer_id) + ", role " +
         std::to_string(tensor.role) + ", group " +
         std::to_string(tensor.group_id);
}

bool same_tensor_id(const CacheTensorManifest& lhs,
                    const CacheTensorManifest& rhs) {
  return lhs.cache_namespace == rhs.cache_namespace &&
         lhs.layer_id == rhs.layer_id && lhs.role == rhs.role &&
         lhs.group_id == rhs.group_id;
}

bool declaration_matches(const CacheTensorDeclaration& declaration,
                         const CacheTensorManifest& tensor) {
  return declaration.cache_namespace == tensor.cache_namespace &&
         declaration.role == tensor.role &&
         declaration.group_id == tensor.group_id;
}

bool row_bases_match(const CacheRowBases& bases,
                     const CacheTensorManifest& tensor) {
  return bases.cache_namespace == tensor.cache_namespace &&
         bases.layer_id == tensor.layer_id && bases.role == tensor.role &&
         bases.group_id == tensor.group_id;
}

// Physical geometry of the local heads of one rank, as the descriptor states
// it. `whole_resource` marks the degenerate descriptor whose single span covers
// the entire cache resource and therefore carries no head axis of its own.
struct HeadGeometry {
  uint64_t head_bytes = 0;
  int32_t head_begin = 0;
  int32_t local_head_count = 0;
  bool whole_resource = false;
};

// Reads the head axis of a descriptor and proves that it is the layout
// RouteBinder addresses: one span per local head, heads contiguous inside a
// resource, and a global head range that is exactly one head class.
bool derive_head_geometry(const WorkerCacheLayoutManifest& manifest,
                          const CacheTensorManifest& tensor,
                          const CacheTensorDeclaration& declaration,
                          const KvRedundancy& redundancy,
                          uint64_t units,
                          HeadGeometry* geometry,
                          std::string* error) {
  const std::string id = tensor_id(tensor);
  const std::vector<LogicalSpan>& spans = tensor.shard.spans;

  for (const LogicalSpan& span : spans) {
    if (span.logical_tensor != spans.front().logical_tensor) {
      set_error(
          error,
          id + ": the descriptor names both " + spans.front().logical_tensor +
              " and " + span.logical_tensor +
              ", and the canonical route addresses one logical tensor per "
              "edge");
      return false;
    }
    if (span.bytes_per_region == 0 ||
        span.bytes_per_region != spans.front().bytes_per_region) {
      set_error(error, id + ": logical spans disagree on the size of one head");
      return false;
    }
  }

  const uint64_t head_bytes = spans.front().bytes_per_region;
  const int32_t global_heads = declaration.group.global_head_count;
  const int32_t local_heads = redundancy.local_head_count();
  const int32_t tp_redundancy = redundancy.tp_redundancy();

  const bool whole_resource = spans.size() == 1 &&
                              spans.front().repeat_count == 1 &&
                              head_bytes == tensor.resource_stride_bytes;
  if (whole_resource) {
    // The descriptor gives no head axis: the whole resource is one local head's
    // data, `units` sub-units of it. Requiring a single local head is what
    // keeps the byte order inside the resource unconstrained -- with one head
    // there is nothing to order -- which is also what makes a packed component
    // layout (CONV) safe to move as one run. A group with more local heads has
    // to publish one span per head instead.
    if (local_heads != kSingleHead || units == 0 ||
        tensor.resource_stride_bytes % units != 0) {
      set_error(error,
                id + ": a whole-resource descriptor holds no head axis and " +
                    std::to_string(local_heads) +
                    " local heads; publish one span per local head instead");
      return false;
    }
    // Which head it holds comes from the rank, because the descriptor cannot
    // say: every replica of a whole resource publishes the same span.
    int32_t head_begin = 0;
    if (tensor.cache_namespace == CacheNamespace::MAIN) {
      head_begin = manifest.coordinates.tp_rank / tp_redundancy;
    } else if (global_heads != kSingleHead) {
      set_error(error,
                id + ": a whole-resource descriptor of a draft body whose rank "
                     "is unknown cannot be placed in the global head range");
      return false;
    }
    geometry->head_bytes = tensor.resource_stride_bytes / units;
    geometry->head_begin = head_begin;
    geometry->local_head_count = kSingleHead;
    geometry->whole_resource = true;
    return true;
  }

  if (spans.size() != static_cast<size_t>(local_heads)) {
    set_error(error,
              id + ": the descriptor holds " + std::to_string(spans.size()) +
                  " spans but the group has " + std::to_string(local_heads) +
                  " local heads on this topology");
    return false;
  }

  std::vector<size_t> order(spans.size());
  for (size_t index = 0; index < order.size(); ++index) {
    order[index] = index;
  }
  std::sort(order.begin(), order.end(), [&spans](size_t lhs, size_t rhs) {
    return spans[lhs].logical_offset_bytes < spans[rhs].logical_offset_bytes;
  });

  const uint64_t begin_bytes = spans[order.front()].logical_offset_bytes;
  if (begin_bytes % head_bytes != 0 ||
      begin_bytes / head_bytes >
          static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    set_error(
        error,
        id + ": the first logical span does not start on a head boundary");
    return false;
  }
  const int32_t head_begin = static_cast<int32_t>(begin_bytes / head_bytes);
  if (head_begin % local_heads != 0) {
    set_error(error,
              id + ": the descriptor starts at global head " +
                  std::to_string(head_begin) +
                  ", which is not the start of a " +
                  std::to_string(local_heads) + "-head class");
    return false;
  }
  const int32_t head_class = head_begin / local_heads;
  if (head_class >= redundancy.head_class_count()) {
    set_error(error,
              id + ": the descriptor claims head class " +
                  std::to_string(head_class) + " of " +
                  std::to_string(redundancy.head_class_count()));
    return false;
  }
  // The worker coordinates of a MAIN manifest describe the rank that published
  // it, so the head class must be that rank's. A SPEC_DRAFT manifest carries
  // MAIN's coordinates while the draft body keeps its own placement in the
  // descriptor, so there the class is taken as declared.
  if (tensor.cache_namespace == CacheNamespace::MAIN &&
      head_class != manifest.coordinates.tp_rank / tp_redundancy) {
    set_error(error,
              id + ": the descriptor holds head class " +
                  std::to_string(head_class) + " but tp rank " +
                  std::to_string(manifest.coordinates.tp_rank) +
                  " owns class " +
                  std::to_string(manifest.coordinates.tp_rank / tp_redundancy));
    return false;
  }

  const int32_t owner_tp_rank = head_class * tp_redundancy;
  const uint64_t logical_stride =
      static_cast<uint64_t>(global_heads) * head_bytes;
  const uint64_t physical_stride =
      static_cast<uint64_t>(local_heads) * head_bytes;
  for (size_t position = 0; position < order.size(); ++position) {
    const LogicalSpan& span = spans[order[position]];
    const uint64_t local_head = static_cast<uint64_t>(position);
    if (span.repeat_count != units) {
      set_error(error,
                id + ": a logical span repeats " +
                    std::to_string(span.repeat_count) +
                    " times but one cache resource holds " +
                    std::to_string(units) + " sub-units");
      return false;
    }
    if (span.logical_offset_bytes !=
        (static_cast<uint64_t>(head_begin) + local_head) * head_bytes) {
      set_error(error,
                id + ": logical span " + std::to_string(position) +
                    " is not the next global head");
      return false;
    }
    if (span.physical_offset_bytes != local_head * head_bytes) {
      set_error(error,
                id + ": logical span " + std::to_string(position) +
                    " is not the next local head inside the cache resource");
      return false;
    }
    // A span that covers a single sub-unit has no stride to describe, so its
    // stride fields are not part of the layout.
    if (span.repeat_count > 1 &&
        (span.logical_stride_bytes != logical_stride ||
         span.physical_stride_bytes != physical_stride)) {
      set_error(error,
                id + ": logical span " + std::to_string(position) +
                    " does not stride over the global heads and sub-units of "
                    "the canonical layout");
      return false;
    }
    if (span.owner_tp_rank != owner_tp_rank) {
      set_error(error,
                id + ": logical span " + std::to_string(position) +
                    " names owner tp rank " +
                    std::to_string(span.owner_tp_rank) + " but head class " +
                    std::to_string(head_class) + " is written by tp rank " +
                    std::to_string(owner_tp_rank));
      return false;
    }
  }

  geometry->head_bytes = head_bytes;
  geometry->head_begin = head_begin;
  geometry->local_head_count = local_heads;
  geometry->whole_resource = false;
  return true;
}

bool describe_tensor(const WorkerCacheLayoutManifest& manifest,
                     const CacheTensorManifest& tensor,
                     const CacheTensorDeclaration& declaration,
                     const CacheRowBases* row_bases,
                     PeerCacheView* view,
                     std::string* error) {
  const std::string id = tensor_id(tensor);

  if (tensor.layer_id < 0) {
    set_error(error, id + ": layer id must not be negative");
    return false;
  }
  if (!tensor.contiguous) {
    set_error(error,
              id + ": a non-contiguous cache tensor has no canonical block "
                   "layout");
    return false;
  }
  if (tensor.resource_count == 0 || tensor.resource_stride_bytes == 0) {
    set_error(error, id + ": the cache tensor holds no cache resource");
    return false;
  }
  if (multiply_overflows(tensor.resource_count, tensor.resource_stride_bytes) ||
      tensor.resource_count * tensor.resource_stride_bytes >
          tensor.buffer_bytes) {
    set_error(error, id + ": the cache buffer is smaller than its resources");
    return false;
  }

  const LogicalShardDescriptor& descriptor = tensor.shard;
  if (descriptor.spans.empty()) {
    set_error(error, id + ": the descriptor has no logical span");
    return false;
  }
  if (descriptor.kind == LogicalShardKind::COMPOSITE) {
    set_error(error,
              id +
                  ": a composite descriptor interleaves several logical "
                  "tensors in one physical row, and the canonical route "
                  "carries one head interval per edge; keep such groups on the "
                  "legacy planner or extend the edge with per-component byte "
                  "offsets");
    return false;
  }

  const bool sequence_scoped =
      descriptor.resource_scope == CacheResourceScope::SEQUENCE;
  if (sequence_scoped != declaration.group.sequence_scoped) {
    set_error(error,
              id + ": the manifest describes a " +
                  std::string(sequence_scoped ? "sequence" : "block") +
                  "-scoped cache tensor but the model declares a " +
                  std::string(declaration.group.sequence_scoped ? "sequence"
                                                                : "block") +
                  "-scoped group");
    return false;
  }

  KvRedundancy redundancy;
  std::string reason;
  if (!KvRedundancy::derive(
          declaration.topology, declaration.group, &redundancy, &reason)) {
    set_error(error, id + ": " + reason);
    return false;
  }

  // The slice a rank physically holds is its runtime DCP rank, and the manifest
  // publishes that rank in its coordinates. A group whose effective split is
  // the configured one therefore has to agree with the coordinates exactly: the
  // route places canonical blocks by slice while the runtime places them by DCP
  // rank, and the two only line up when this holds. A group that keeps the
  // whole sequence (split 1) has no slice to disagree about.
  if (tensor.cache_namespace == CacheNamespace::MAIN &&
      redundancy.split() == std::max(declaration.topology.kv_split_size, 1)) {
    const KvLayoutIndex index(declaration.topology, redundancy);
    const int32_t slice = index.slice_of(manifest.coordinates.cp_rank,
                                         manifest.coordinates.tp_rank);
    if (slice != manifest.coordinates.kv_split_rank) {
      set_error(error,
                id + ": rank (cp " +
                    std::to_string(manifest.coordinates.cp_rank) + ", tp " +
                    std::to_string(manifest.coordinates.tp_rank) +
                    ") holds sequence slice " + std::to_string(slice) +
                    " but publishes DCP rank " +
                    std::to_string(manifest.coordinates.kv_split_rank) +
                    "; the split is not placed where the runtime places it");
      return false;
    }
  }

  uint64_t units = 0;
  if (sequence_scoped) {
    // One resource is one sequence slot, holding `physical_rows_per_resource`
    // checkpointed rows.
    units = tensor.physical_rows_per_resource;
    if (units == 0) {
      set_error(error,
                id + ": a sequence-scoped cache tensor must declare its "
                     "physical rows per resource");
      return false;
    }
  } else {
    // One resource is one canonical block, holding `tokens_per_block` tokens.
    // The two peers therefore have to agree on the token capacity before any
    // route can be built.
    if (declaration.topology.tokens_per_block <= 0) {
      set_error(error,
                id + ": a block-scoped group must declare tokens_per_block");
      return false;
    }
    units = tensor.block_token_capacity;
    if (units != static_cast<uint64_t>(declaration.topology.tokens_per_block)) {
      set_error(error,
                id + ": the manifest stores " + std::to_string(units) +
                    " tokens per block but the model declares " +
                    std::to_string(declaration.topology.tokens_per_block));
      return false;
    }
  }

  HeadGeometry geometry;
  if (!derive_head_geometry(
          manifest, tensor, declaration, redundancy, units, &geometry, error)) {
    return false;
  }

  if (declaration.group.head_bytes != 0 &&
      declaration.group.head_bytes != geometry.head_bytes) {
    set_error(error,
              id + ": the model declares " +
                  std::to_string(declaration.group.head_bytes) +
                  " bytes per head but the descriptor holds " +
                  std::to_string(geometry.head_bytes));
    return false;
  }

  if (tensor.explicit_resource_offsets) {
    if (row_bases == nullptr) {
      set_error(error,
                id + ": the cache tensor addresses its rows through explicit "
                     "offsets but none were supplied");
      return false;
    }
    if (row_bases->row_offsets.size() < tensor.resource_count) {
      set_error(error,
                id + ": the cache tensor holds " +
                    std::to_string(tensor.resource_count) + " rows but only " +
                    std::to_string(row_bases->row_offsets.size()) +
                    " explicit row bases were supplied");
      return false;
    }
  } else if (row_bases != nullptr) {
    set_error(error,
              id + ": explicit row bases were supplied for a cache tensor that "
                   "addresses its rows by resource stride");
    return false;
  }

  view->topology = declaration.topology;
  view->group = declaration.group;
  view->group.head_bytes = geometry.head_bytes;
  // The worker coordinates describe the publishing rank, except for a draft
  // body, whose placement lives in its own descriptor and whose coordinates are
  // MAIN's. Leaving the rank unknown there keeps bind() from comparing the two
  // frames; the caller filters the edges instead.
  view->local_rank =
      tensor.cache_namespace == CacheNamespace::MAIN
          ? manifest.coordinates.cp_rank * declaration.topology.tp_size +
                manifest.coordinates.tp_rank
          : -1;
  view->entry.cache_namespace = tensor.cache_namespace;
  view->entry.layer_id = tensor.layer_id;
  view->entry.role = tensor.role;
  view->entry.group_id = tensor.group_id;
  view->entry.buffer_id = tensor.mooncake_buffer_id;
  view->entry.resource_count = tensor.resource_count;
  view->entry.resource_stride_bytes = tensor.resource_stride_bytes;
  view->entry.buffer_bytes = tensor.buffer_bytes;
  view->entry.units_per_resource = units;
  view->entry.explicit_offsets = tensor.explicit_resource_offsets;
  if (row_bases != nullptr) {
    view->row_offsets = row_bases->row_offsets;
  } else {
    view->row_offsets.clear();
  }
  return true;
}

}  // namespace

bool declare_cache_group(const CacheTensorLayoutContext& context,
                         KVCacheTensorRole role,
                         GroupTopology* group,
                         std::string* error) {
  if (group == nullptr) {
    set_error(error, "group geometry output must not be null");
    return false;
  }
  const std::string role_name(role.to_string());

  // Mirrors the dispatch order of describe_cache_tensor(): the attention roles
  // first, then the indexer, then the recurrent state.
  if (is_kv_head_role(role)) {
    if (context.kv_head_count <= 0) {
      set_error(
          error,
          "role " + role_name +
              ": the model declares no KV head count, so the group has no "
              "head geometry to route");
      return false;
    }
    // An MLA instance describes these tensors as one whole resource, which
    // holds a single latent head: `local_heads` has to stay 1 for the binder to
    // place it.
    group->global_head_count =
        context.enable_mla ? 1 : static_cast<int32_t>(context.kv_head_count);
    group->sequence_scoped = false;
    group->full_sequence_replica = false;
    return true;
  }

  if (role == KVCacheTensorRole::INDEX ||
      role == KVCacheTensorRole::INDEX_SCALE) {
    // describe_attention_heads(global_head_count=1): the cached key is one
    // shared logical head on every rank.
    group->global_head_count = 1;
    group->sequence_scoped = false;
    group->full_sequence_replica = true;
    return true;
  }

  if (role == KVCacheTensorRole::SSM && context.linear_value_head_count > 0) {
    group->global_head_count =
        static_cast<int32_t>(context.linear_value_head_count);
    group->sequence_scoped = true;
    group->full_sequence_replica = false;
    return true;
  }

  if (role == KVCacheTensorRole::CONV && context.linear_key_head_count > 0 &&
      context.linear_value_head_count > 0) {
    // Without MLA this role publishes a composite descriptor, which the
    // canonical route refuses on purpose (see describe_tensor). Declaring the
    // geometry anyway keeps that refusal in one place.
    group->global_head_count =
        static_cast<int32_t>(context.linear_value_head_count);
    group->sequence_scoped = true;
    group->full_sequence_replica = false;
    return true;
  }

  set_error(
      error,
      "role " + role_name +
          ": this role has no declared group geometry, so the canonical "
          "route cannot serve it; either declare its head count and scope "
          "or keep the family on the legacy planner");
  return false;
}

bool canonical_blocks_of_request(
    const std::vector<CacheGroupRequest>& groups,
    const std::vector<CacheTensorDeclaration>& local,
    std::vector<int64_t>* canonical_blocks,
    std::string* error) {
  if (canonical_blocks == nullptr) {
    set_error(error, "canonical block output must not be null");
    return false;
  }
  canonical_blocks->clear();
  for (const CacheGroupRequest& group : groups) {
    bool declared = false;
    bool sequence_scoped = false;
    int32_t split = 1;
    for (const CacheTensorDeclaration& declaration : local) {
      if (declaration.group_id != group.group_id) {
        continue;
      }
      if (!declared) {
        declared = true;
        sequence_scoped = declaration.group.sequence_scoped;
        // The configured split is instance wide, which is exactly the DCP size
        // the runtime places the logical blocks with.
        split = std::max(declaration.topology.kv_split_size, 1);
        continue;
      }
      if (declaration.group.sequence_scoped != sequence_scoped) {
        set_error(error,
                  "cache group " + std::to_string(group.group_id) +
                      " has both sequence-scoped and block-scoped families, so "
                      "its request ids have no single meaning");
        return false;
      }
    }
    if (!declared) {
      set_error(error,
                "the model declares no cache family for cache group " +
                    std::to_string(group.group_id));
      return false;
    }

    if (sequence_scoped) {
      for (uint64_t id : group.ids) {
        canonical_blocks->emplace_back(static_cast<int64_t>(id));
      }
      continue;
    }
    // One logical block spans one canonical block per DCP rank.
    const int64_t factor = static_cast<int64_t>(split);
    for (uint64_t id : group.ids) {
      const int64_t base = static_cast<int64_t>(id) * factor;
      for (int64_t offset = 0; offset < factor; ++offset) {
        canonical_blocks->emplace_back(base + offset);
      }
    }
  }

  std::sort(canonical_blocks->begin(), canonical_blocks->end());
  canonical_blocks->erase(
      std::unique(canonical_blocks->begin(), canonical_blocks->end()),
      canonical_blocks->end());
  return true;
}

bool PeerDirectory::describe(
    const WorkerCacheLayoutManifest& manifest,
    const std::vector<CacheTensorDeclaration>& declarations,
    const std::vector<CacheRowBases>& row_bases,
    PeerDirectory* directory,
    std::string* error) {
  if (directory == nullptr) {
    set_error(error, "the peer directory output must not be null");
    return false;
  }
  directory->views_.clear();
  if (manifest.schema_version != kCacheLayoutSchemaVersion) {
    set_error(error,
              "unsupported cache layout schema version " +
                  std::to_string(manifest.schema_version) + ", expected " +
                  std::to_string(kCacheLayoutSchemaVersion));
    return false;
  }
  if (declarations.empty()) {
    set_error(error, "no cache tensor declaration was supplied");
    return false;
  }
  for (size_t lhs = 0; lhs < declarations.size(); ++lhs) {
    for (size_t rhs = lhs + 1; rhs < declarations.size(); ++rhs) {
      if (declarations[lhs].cache_namespace ==
              declarations[rhs].cache_namespace &&
          declarations[lhs].role == declarations[rhs].role &&
          declarations[lhs].group_id == declarations[rhs].group_id) {
        set_error(error,
                  "cache tensor declarations are duplicated for role " +
                      std::to_string(declarations[lhs].role) + " group " +
                      std::to_string(declarations[lhs].group_id));
        return false;
      }
    }
  }
  for (size_t lhs = 0; lhs < row_bases.size(); ++lhs) {
    for (size_t rhs = lhs + 1; rhs < row_bases.size(); ++rhs) {
      if (row_bases[lhs].cache_namespace == row_bases[rhs].cache_namespace &&
          row_bases[lhs].layer_id == row_bases[rhs].layer_id &&
          row_bases[lhs].role == row_bases[rhs].role &&
          row_bases[lhs].group_id == row_bases[rhs].group_id) {
        set_error(error,
                  "explicit row bases are duplicated for layer " +
                      std::to_string(row_bases[lhs].layer_id) + " role " +
                      std::to_string(row_bases[lhs].role));
        return false;
      }
    }
  }

  std::vector<PeerCacheView> views;
  views.reserve(manifest.tensors.size());
  for (size_t index = 0; index < manifest.tensors.size(); ++index) {
    const CacheTensorManifest& tensor = manifest.tensors[index];
    for (size_t other = index + 1; other < manifest.tensors.size(); ++other) {
      if (same_tensor_id(tensor, manifest.tensors[other])) {
        set_error(
            error,
            tensor_id(tensor) + " is published more than once by the peer");
        return false;
      }
    }
    const CacheTensorDeclaration* declaration = nullptr;
    for (const CacheTensorDeclaration& candidate : declarations) {
      if (declaration_matches(candidate, tensor)) {
        declaration = &candidate;
        break;
      }
    }
    if (declaration == nullptr) {
      set_error(error,
                tensor_id(tensor) +
                    " is published by the peer but the model does not declare "
                    "it");
      return false;
    }
    if (tensor.cache_namespace == CacheNamespace::MAIN &&
        (declaration->topology.dp_size != manifest.coordinates.dp_size ||
         declaration->topology.cp_size != manifest.coordinates.cp_size ||
         declaration->topology.tp_size != manifest.coordinates.tp_size ||
         declaration->topology.kv_split_size !=
             manifest.coordinates.kv_split_size)) {
      set_error(error,
                tensor_id(tensor) +
                    ": the declared topology differs from the coordinates the "
                    "peer published");
      return false;
    }

    const CacheRowBases* bases = nullptr;
    for (const CacheRowBases& candidate : row_bases) {
      if (row_bases_match(candidate, tensor)) {
        bases = &candidate;
        break;
      }
    }

    PeerCacheView view;
    if (!describe_tensor(manifest, tensor, *declaration, bases, &view, error)) {
      return false;
    }
    views.emplace_back(std::move(view));
  }

  directory->views_ = std::move(views);
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

const PeerCacheView* PeerDirectory::find(CacheNamespace cache_namespace,
                                         int64_t layer_id,
                                         int32_t role,
                                         int32_t group_id) const {
  for (const PeerCacheView& view : views_) {
    if (view.entry.cache_namespace == cache_namespace &&
        view.entry.layer_id == layer_id && view.entry.role == role &&
        view.entry.group_id == group_id) {
      return &view;
    }
  }
  return nullptr;
}

}  // namespace xllm
