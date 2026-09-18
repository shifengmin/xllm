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

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache_transfer/cache_layout.h"
#include "framework/kv_cache_transfer/kv_redundancy.h"
#include "framework/kv_cache_transfer/pd_route_table.h"

namespace xllm {

// One contiguous byte range of one buffer pair. Every peer-dependent number has
// been resolved by the time a RouteRegion exists, so this is all the transport
// layer below has to understand.
struct RouteRegion {
  uint64_t local_buffer_id = 0;
  uint64_t local_offset = 0;
  uint64_t remote_buffer_id = 0;
  uint64_t remote_offset = 0;
  uint64_t length = 0;
};

// Physical geometry of one cache tensor on one peer.
//
// resource_count counts physical rows: canonical blocks for a split block
// group, sequence slots for a sequence-scoped one. units_per_resource is the
// number of repeated sub-units one resource holds -- the tokens of a canonical
// block, or the checkpoint rows of a recurrent-state slot -- and is peer
// independent, because the canonical unit it belongs to is.
struct BufferDirectoryEntry {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t role = 0;
  int32_t group_id = 0;
  uint64_t buffer_id = 0;
  uint64_t resource_count = 0;
  uint64_t resource_stride_bytes = 0;
  uint64_t buffer_bytes = 0;
  uint64_t units_per_resource = 1;
  // Page-mapped buffers (GlobalXTensor) address a row through an explicit byte
  // base instead of row * resource_stride_bytes.
  bool explicit_offsets = false;
};

// The physical cache buffers one peer published, addressed the way a request
// addresses them.
class BufferDirectory final {
 public:
  void add(BufferDirectoryEntry entry);

  const BufferDirectoryEntry* find(CacheNamespace cache_namespace,
                                   int64_t layer_id,
                                   int32_t role,
                                   int32_t group_id) const;

  size_t size() const { return entries_.size(); }
  const BufferDirectoryEntry& at(size_t index) const {
    return entries_.at(index);
  }

 private:
  std::vector<BufferDirectoryEntry> entries_;
};

// Everything binding needs to know about one peer's view of one cache tensor:
// its instance topology, this group's geometry, and the physical buffer.
struct PeerCacheView {
  KvTopology topology;
  GroupTopology group;
  BufferDirectoryEntry entry;
  // Page bases indexed by physical row; used only when entry.explicit_offsets
  // is set, and therefore empty otherwise.
  std::vector<uint64_t> row_offsets;
};

// Turns canonical blocks into byte ranges.
//
// This is the single place where the peer-independent coordinates of the route
// meet the peer-dependent physical layout, and therefore the only place where a
// folding mistake could live: the canonical block decides both rows, and the
// head range decides the offset inside each row. Nothing is intersected, and no
// region is inferred from the other side's resource geometry.
//
// Assumption inherited from the physical layouts in use: the local heads of one
// rank are contiguous inside a resource, so a head range is one contiguous run
// of `(head_end - head_begin) * head_bytes` bytes per sub-unit. The legacy
// planner encodes the same layout as one span per head, which is what the
// equivalence test compares against.
class RouteBinder final {
 public:
  // Binds the transfer into one destination rank.
  //
  // `canonical_blocks` are exactly the canonical blocks this call has to carry
  // to `dst_local_rank`, strictly ascending; every one of them must be covered
  // by an edge of `edges` that ends at that rank. Callers therefore group the
  // work by (source rank, destination rank) pair, which is also how the
  // transport below addresses peers. Sequence-scoped groups pass slot ids
  // instead: their split is 1, so the row is the id.
  //
  // Emits one region per (edge, canonical block, sub-unit) and then merges
  // regions that are adjacent in both buffers, so the output shape does not
  // depend on the split width.
  static bool bind(const std::vector<RouteEdge>& edges,
                   int32_t dst_local_rank,
                   const std::vector<int64_t>& canonical_blocks,
                   const PeerCacheView& local,
                   const PeerCacheView& remote,
                   std::vector<RouteRegion>* regions,
                   std::string* error);
};

}  // namespace xllm
