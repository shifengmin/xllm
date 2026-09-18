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

#include "framework/kv_cache_transfer/route_binder.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace xllm {

namespace {

// Head geometry of one rank of one peer.
struct RankGeometry {
  int32_t head_begin = 0;
  int32_t local_head_count = 0;
};

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool add_overflows(uint64_t lhs, uint64_t rhs) {
  return rhs > std::numeric_limits<uint64_t>::max() - lhs;
}

bool multiply_overflows(uint64_t lhs, uint64_t rhs) {
  return lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs;
}

bool derive_side(const KvTopology& topology,
                 const GroupTopology& group,
                 const char* side,
                 KvRedundancy* redundancy,
                 std::string* error) {
  std::string reason;
  if (!KvRedundancy::derive(topology, group, redundancy, &reason)) {
    set_error(error, std::string(side) + " side: " + reason);
    return false;
  }
  return true;
}

// Per-rank head geometry, indexed by local rank.
std::vector<RankGeometry> describe_ranks(const KvTopology& topology,
                                         const KvRedundancy& redundancy,
                                         const KvLayoutIndex& index) {
  const int32_t ranks = topology.cp_size * topology.tp_size;
  std::vector<RankGeometry> geometry(static_cast<size_t>(ranks));
  for (int32_t rank = 0; rank < ranks; ++rank) {
    const int32_t tp_rank = rank % topology.tp_size;
    const int32_t head_class = index.head_class_of(tp_rank);
    RankGeometry& rank_geometry = geometry[static_cast<size_t>(rank)];
    rank_geometry.head_begin = index.head_begin(head_class);
    rank_geometry.local_head_count = redundancy.local_head_count();
  }
  return geometry;
}

bool row_base(const BufferDirectoryEntry& entry,
              const std::vector<uint64_t>& row_offsets,
              uint64_t row,
              const char* side,
              uint64_t* base,
              std::string* error) {
  if (entry.explicit_offsets) {
    if (row >= row_offsets.size()) {
      set_error(error,
                std::string(side) + " row " + std::to_string(row) +
                    " has no explicit offset");
      return false;
    }
    *base = row_offsets[static_cast<size_t>(row)];
    return true;
  }
  if (multiply_overflows(row, entry.resource_stride_bytes)) {
    set_error(error, std::string(side) + " resource offset overflows");
    return false;
  }
  *base = row * entry.resource_stride_bytes;
  return true;
}

// Merges regions that are adjacent in both buffers, so that a run of canonical
// blocks that happens to be contiguous is one region regardless of how the
// route table happened to split it into edges.
void compact_regions(std::vector<RouteRegion>* regions) {
  std::sort(regions->begin(),
            regions->end(),
            [](const RouteRegion& lhs, const RouteRegion& rhs) {
              return std::tie(lhs.local_buffer_id,
                              lhs.remote_buffer_id,
                              lhs.local_offset,
                              lhs.remote_offset,
                              lhs.length) < std::tie(rhs.local_buffer_id,
                                                     rhs.remote_buffer_id,
                                                     rhs.local_offset,
                                                     rhs.remote_offset,
                                                     rhs.length);
            });

  std::vector<RouteRegion> merged;
  merged.reserve(regions->size());
  for (const RouteRegion& region : *regions) {
    if (!merged.empty()) {
      RouteRegion& previous = merged.back();
      const bool adjacent =
          previous.local_buffer_id == region.local_buffer_id &&
          previous.remote_buffer_id == region.remote_buffer_id &&
          !add_overflows(previous.local_offset, previous.length) &&
          !add_overflows(previous.remote_offset, previous.length) &&
          previous.local_offset + previous.length == region.local_offset &&
          previous.remote_offset + previous.length == region.remote_offset &&
          !add_overflows(previous.length, region.length);
      if (adjacent) {
        previous.length += region.length;
        continue;
      }
    }
    merged.emplace_back(region);
  }
  regions->swap(merged);
}

// The physical row one canonical block occupies in one peer's cache buffer.
//
// A canonical block is a *position* (the request's `p`-th block through slice
// `j`), and no pool lays its rows out one per canonical block, so the row has
// to come from the family's own storage layout:
//
//   - A split family keeps one canonical block per row, and its rows are the
//     logical blocks: row `p + 1` of the pool, because the pool reserves row 0
//     for the padding block. A canonical block whose position is `p * split +
//     j` therefore lands on row `canonical / split + 1` -- the same one-based
//     row space the request's ids live in, which is why the source side of a
//     homogeneous pair got this right by accident.
//   - A family that keeps the whole sequence on every rank packs each slice of
//   a
//     logical block as its own row, exactly as the runtime expands the indexer
//     block table (`row = id * dcp_size + slice`). Its rows are scaled by the
//     instance's *configured* kv-split -- its own effective split is 1, since
//     every rank keeps everything -- and sit one reserved block further along,
//     so the row is `canonical + kv_split_size`.
//   - A sequence-scoped family has no block dimension at all: its canonical id
//     is a sequence slot, which is already a row.
//
// `split` is the family's effective split (KvRedundancy::split()). Getting this
// wrong is invisible within one instance and shifts every block on the peer as
// soon as the two splits differ, so it is spelled out once, here.
uint64_t peer_row(const PeerCacheView& view, int32_t split, int64_t block) {
  if (view.group.sequence_scoped) {
    return static_cast<uint64_t>(block);
  }
  if (view.group.full_sequence_replica) {
    const int32_t width = std::max(view.topology.kv_split_size, 1);
    return static_cast<uint64_t>(block) + static_cast<uint64_t>(width);
  }
  return static_cast<uint64_t>(block) / static_cast<uint64_t>(split) + 1;
}

}  // namespace

void BufferDirectory::add(BufferDirectoryEntry entry) {
  entries_.emplace_back(std::move(entry));
}

const BufferDirectoryEntry* BufferDirectory::find(
    CacheNamespace cache_namespace,
    int64_t layer_id,
    int32_t role,
    int32_t group_id) const {
  for (const BufferDirectoryEntry& entry : entries_) {
    if (entry.cache_namespace == cache_namespace &&
        entry.layer_id == layer_id && entry.role == role &&
        entry.group_id == group_id) {
      return &entry;
    }
  }
  return nullptr;
}

bool RouteBinder::bind(const std::vector<RouteEdge>& edges,
                       int32_t dst_local_rank,
                       const std::vector<int64_t>& canonical_blocks,
                       const PeerCacheView& local,
                       const PeerCacheView& remote,
                       std::vector<RouteRegion>* regions,
                       std::string* error) {
  if (regions == nullptr) {
    set_error(error, "region output must not be null");
    return false;
  }
  regions->clear();

  for (size_t index = 1; index < canonical_blocks.size(); ++index) {
    if (canonical_blocks[index] <= canonical_blocks[index - 1]) {
      set_error(error,
                "canonical blocks must be strictly ascending: " +
                    std::to_string(canonical_blocks[index - 1]) + " then " +
                    std::to_string(canonical_blocks[index]));
      return false;
    }
  }

  // The canonical unit is shared: both peers must agree on the global head
  // count, on how many sub-units one resource holds, and on the size of a head.
  if (local.group.global_head_count != remote.group.global_head_count) {
    set_error(error, "peers disagree on the global head count");
    return false;
  }
  if (local.group.sequence_scoped != remote.group.sequence_scoped) {
    set_error(error, "peers disagree on the resource scope of the cache group");
    return false;
  }
  if (local.group.head_bytes != remote.group.head_bytes) {
    set_error(error, "peers disagree on the size of one cache head");
    return false;
  }
  if (local.entry.units_per_resource != remote.entry.units_per_resource ||
      local.entry.units_per_resource == 0) {
    set_error(error,
              "peers disagree on the number of sub-units per cache resource "
              "(canonical block size)");
    return false;
  }

  KvRedundancy local_redundancy;
  KvRedundancy remote_redundancy;
  if (!derive_side(
          local.topology, local.group, "local", &local_redundancy, error)) {
    return false;
  }
  if (!derive_side(
          remote.topology, remote.group, "remote", &remote_redundancy, error)) {
    return false;
  }

  const KvLayoutIndex local_index(local.topology, local_redundancy);
  const KvLayoutIndex remote_index(remote.topology, remote_redundancy);
  const std::vector<RankGeometry> local_ranks =
      describe_ranks(local.topology, local_redundancy, local_index);
  const std::vector<RankGeometry> remote_ranks =
      describe_ranks(remote.topology, remote_redundancy, remote_index);
  const int32_t local_split = local_redundancy.split();
  const int32_t remote_split = remote_redundancy.split();
  const uint64_t units = local.entry.units_per_resource;
  const uint64_t head_bytes = local.group.head_bytes;

  if (dst_local_rank < 0 ||
      dst_local_rank >= static_cast<int32_t>(remote_ranks.size())) {
    set_error(error,
              "destination rank " + std::to_string(dst_local_rank) +
                  " is outside the remote instance");
    return false;
  }
  // A view that names its rank has to be the rank being bound, otherwise the
  // remote offsets below would address another rank's buffer. (A source view
  // that names its rank only filters the edges, so it is checked per edge.)
  if (remote.local_rank >= 0 && remote.local_rank != dst_local_rank) {
    set_error(error,
              "the destination view belongs to rank " +
                  std::to_string(remote.local_rank) + " but rank " +
                  std::to_string(dst_local_rank) + " was requested");
    return false;
  }

  // Bucket the request's canonical blocks by source slice, so that each edge
  // walks only the blocks it owns instead of the whole request (F7).
  std::vector<std::vector<size_t>> buckets(static_cast<size_t>(local_split));
  for (std::vector<size_t>& bucket : buckets) {
    bucket.reserve(canonical_blocks.size() / static_cast<size_t>(local_split) +
                   1);
  }
  std::vector<uint8_t> matched(canonical_blocks.size(), 0);
  for (size_t index = 0; index < canonical_blocks.size(); ++index) {
    const int64_t block = canonical_blocks[index];
    if (block < 0) {
      set_error(error, "canonical block must not be negative");
      return false;
    }
    buckets[static_cast<size_t>(block % local_split)].emplace_back(index);
  }

  const size_t regions_per_edge =
      canonical_blocks.size() / static_cast<size_t>(local_split) + 1;
  regions->reserve(edges.size() * regions_per_edge *
                   static_cast<size_t>(units));

  for (const RouteEdge& edge : edges) {
    // A bind call is scoped to one destination rank, because the transport
    // below writes one peer's buffer at a time.
    if (edge.dst_local_rank != dst_local_rank) {
      continue;
    }
    if (edge.src_local_rank < 0 ||
        edge.src_local_rank >= static_cast<int32_t>(local_ranks.size())) {
      set_error(error,
                "edge source rank " + std::to_string(edge.src_local_rank) +
                    " is outside the local instance");
      return false;
    }
    if (local.local_rank >= 0 && local.local_rank != edge.src_local_rank) {
      // The table may cover every source rank; the work of the other ranks is
      // carried by their own bind calls. Coverage below still fails if a
      // requested block has no writer of this rank.
      continue;
    }
    if (edge.src_slice < 0 || edge.src_slice >= local_split) {
      set_error(error,
                "edge source slice " + std::to_string(edge.src_slice) +
                    " is outside the local group split");
      return false;
    }
    if (edge.dst_slice < 0 || edge.dst_slice >= remote_split) {
      set_error(error,
                "edge destination slice " + std::to_string(edge.dst_slice) +
                    " is outside the remote group split");
      return false;
    }
    if (edge.head_begin < 0 || edge.head_begin >= edge.head_end ||
        edge.head_end > local.group.global_head_count) {
      set_error(error, "edge head range is outside the global head range");
      return false;
    }

    const RankGeometry& local_rank =
        local_ranks[static_cast<size_t>(edge.src_local_rank)];
    const RankGeometry& remote_rank =
        remote_ranks[static_cast<size_t>(edge.dst_local_rank)];
    // The edge must belong to this rank, otherwise the local head offsets below
    // would silently address the wrong heads.
    if (edge.head_begin < local_rank.head_begin ||
        edge.head_end > local_rank.head_begin + local_rank.local_head_count ||
        edge.head_begin < remote_rank.head_begin ||
        edge.head_end > remote_rank.head_begin + remote_rank.local_head_count) {
      set_error(error,
                "edge head range straddles the head class of one of its ranks");
      return false;
    }
    // The slice an edge carries is a property of the rank it comes from and of
    // the rank it goes to; a mismatch means the table and the directories
    // describe different layouts.
    if (local_index.slice_of(
            /*cp_rank=*/edge.src_local_rank / local.topology.tp_size,
            /*tp_rank=*/edge.src_local_rank % local.topology.tp_size) !=
            edge.src_slice ||
        remote_index.slice_of(
            /*cp_rank=*/edge.dst_local_rank / remote.topology.tp_size,
            /*tp_rank=*/edge.dst_local_rank % remote.topology.tp_size) !=
            edge.dst_slice) {
      set_error(error,
                "edge slice does not match the sequence slice of its ranks");
      return false;
    }

    const uint64_t head_count =
        static_cast<uint64_t>(edge.head_end - edge.head_begin);
    if (multiply_overflows(head_count, head_bytes)) {
      set_error(error, "bound region length overflows");
      return false;
    }
    const uint64_t length = head_count * head_bytes;
    const uint64_t local_head_offset =
        static_cast<uint64_t>(edge.head_begin - local_rank.head_begin) *
        head_bytes;
    const uint64_t remote_head_offset =
        static_cast<uint64_t>(edge.head_begin - remote_rank.head_begin) *
        head_bytes;

    const std::vector<size_t>& bucket =
        buckets[static_cast<size_t>(edge.src_slice)];
    for (size_t block_index : bucket) {
      const int64_t block = canonical_blocks[block_index];
      // Which slice a block belongs to on the destination side is the one
      // thing the caller has to get right: `remote_row` below is the row this
      // rank holds it in, so a block of another slice would be written into
      // this rank's buffer and never reach the rank that owns it. The route
      // derives the destination slice from the block, so a mismatch is a bug in
      // the caller, not a shape the binder can absorb.
      if (block % remote_split != edge.dst_slice) {
        set_error(error,
                  "canonical block " + std::to_string(block) +
                      " belongs to destination slice " +
                      std::to_string(block % remote_split) + ", not slice " +
                      std::to_string(edge.dst_slice) + " of destination rank " +
                      std::to_string(dst_local_rank));
        return false;
      }
      const uint64_t local_row = peer_row(local, local_split, block);
      const uint64_t remote_row = peer_row(remote, remote_split, block);
      if (local_row >= local.entry.resource_count ||
          remote_row >= remote.entry.resource_count) {
        set_error(error,
                  "canonical block " + std::to_string(block) +
                      " maps to physical row " + std::to_string(local_row) +
                      "/" + std::to_string(remote_row) +
                      " outside its cache buffer");
        return false;
      }

      uint64_t local_base = 0;
      uint64_t remote_base = 0;
      if (!row_base(local.entry,
                    local.row_offsets,
                    local_row,
                    "local",
                    &local_base,
                    error) ||
          !row_base(remote.entry,
                    remote.row_offsets,
                    remote_row,
                    "remote",
                    &remote_base,
                    error)) {
        return false;
      }

      const uint64_t local_unit_stride =
          static_cast<uint64_t>(local_rank.local_head_count) * head_bytes;
      const uint64_t remote_unit_stride =
          static_cast<uint64_t>(remote_rank.local_head_count) * head_bytes;
      for (uint64_t unit = 0; unit < units; ++unit) {
        if (multiply_overflows(unit, local_unit_stride) ||
            multiply_overflows(unit, remote_unit_stride)) {
          set_error(error, "sub-unit offset overflows");
          return false;
        }
        RouteRegion region;
        region.local_buffer_id = local.entry.buffer_id;
        region.remote_buffer_id = remote.entry.buffer_id;
        region.length = length;
        if (add_overflows(local_base, local_head_offset) ||
            add_overflows(local_base + local_head_offset,
                          unit * local_unit_stride) ||
            add_overflows(remote_base, remote_head_offset) ||
            add_overflows(remote_base + remote_head_offset,
                          unit * remote_unit_stride)) {
          set_error(error, "bound region offset overflows");
          return false;
        }
        region.local_offset =
            local_base + local_head_offset + unit * local_unit_stride;
        region.remote_offset =
            remote_base + remote_head_offset + unit * remote_unit_stride;
        if (region.local_offset > local.entry.buffer_bytes ||
            region.length > local.entry.buffer_bytes - region.local_offset ||
            region.remote_offset > remote.entry.buffer_bytes ||
            region.length > remote.entry.buffer_bytes - region.remote_offset) {
          set_error(error, "bound region exceeds its cache buffer");
          return false;
        }
        regions->emplace_back(region);
      }

      matched[block_index] = 1;
    }
  }

  for (size_t index = 0; index < canonical_blocks.size(); ++index) {
    if (matched[index] == 0) {
      set_error(error,
                "canonical block " + std::to_string(canonical_blocks[index]) +
                    " is not carried by any edge to destination rank " +
                    std::to_string(dst_local_rank));
      return false;
    }
  }

  compact_regions(regions);
  return true;
}

}  // namespace xllm
