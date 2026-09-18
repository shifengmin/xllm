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

#include "framework/kv_cache_transfer/pd_route_table.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

namespace xllm {

namespace {

// One intersecting pair of head classes: the part of the global head range that
// both peers cover, and the class index each side needs for the slice lookup.
struct HeadPair {
  int32_t src_head_class = 0;
  int32_t dst_head_class = 0;
  int32_t head_begin = 0;
  int32_t head_end = 0;
};

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

int32_t greatest_common_divisor(int32_t lhs, int32_t rhs) {
  while (rhs != 0) {
    const int32_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

int32_t local_rank_count(const KvTopology& topology) {
  return topology.cp_size * topology.tp_size;
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

// Every head class of the source that overlaps a head class of the
// destination. Head routing is exactly this intersection: a head can only be
// transferred between the two ranks that hold it.
std::vector<HeadPair> head_pairs(const KvLayoutIndex& src_index,
                                 const KvLayoutIndex& dst_index,
                                 int32_t src_head_classes,
                                 int32_t dst_head_classes) {
  std::vector<HeadPair> pairs;
  pairs.reserve(static_cast<size_t>(src_head_classes) *
                static_cast<size_t>(dst_head_classes));
  for (int32_t src_head_class = 0; src_head_class < src_head_classes;
       ++src_head_class) {
    const int32_t src_begin = src_index.head_begin(src_head_class);
    const int32_t src_end = src_index.head_end(src_head_class);
    for (int32_t dst_head_class = 0; dst_head_class < dst_head_classes;
         ++dst_head_class) {
      const int32_t begin =
          std::max(src_begin, dst_index.head_begin(dst_head_class));
      const int32_t end = std::min(src_end, dst_index.head_end(dst_head_class));
      if (begin >= end) {
        continue;
      }
      HeadPair pair;
      pair.src_head_class = src_head_class;
      pair.dst_head_class = dst_head_class;
      pair.head_begin = begin;
      pair.head_end = end;
      pairs.emplace_back(pair);
    }
  }
  return pairs;
}

// Destination slices reachable from one source slice. The source slice holds
// the canonical blocks {src_slice + k * src_split}, and a block b lives on the
// destination slice b % dst_split, so the reachable set cycles with period
// dst_split / gcd(src_split, dst_split).
std::vector<int32_t> destination_slices(int32_t src_split,
                                        int32_t dst_split,
                                        int32_t src_slice) {
  const int32_t period =
      dst_split / greatest_common_divisor(src_split, dst_split);
  std::vector<int32_t> slices;
  slices.reserve(static_cast<size_t>(period));
  for (int32_t k = 0; k < period; ++k) {
    slices.emplace_back((src_slice + k * src_split) % dst_split);
  }
  std::sort(slices.begin(), slices.end());
  return slices;
}

}  // namespace

bool PdRouteTable::build(const KvTopology& src_topology,
                         const GroupTopology& src_group,
                         const KvTopology& dst_topology,
                         const GroupTopology& dst_group,
                         std::vector<RouteEdge>* edges,
                         std::string* error) {
  if (edges == nullptr) {
    set_error(error, "edge output must not be null");
    return false;
  }
  edges->clear();

  // Both peers address the same logical heads: the global head range is a
  // property of the model instance, not of the parallel layout.
  if (src_group.global_head_count != dst_group.global_head_count) {
    set_error(error,
              "peers disagree on the global head count (" +
                  std::to_string(src_group.global_head_count) + " vs " +
                  std::to_string(dst_group.global_head_count) + ")");
    return false;
  }
  if (src_group.sequence_scoped != dst_group.sequence_scoped) {
    set_error(error,
              "peers disagree on the resource scope of the cache group "
              "(sequence scoped vs block scoped)");
    return false;
  }

  KvRedundancy src_redundancy;
  KvRedundancy dst_redundancy;
  if (!derive_side(src_topology, src_group, "source", &src_redundancy, error)) {
    return false;
  }
  if (!derive_side(
          dst_topology, dst_group, "destination", &dst_redundancy, error)) {
    return false;
  }

  const int32_t src_split = src_redundancy.split();
  const int32_t dst_split = dst_redundancy.split();
  if (src_split % dst_split != 0 && dst_split % src_split != 0) {
    set_error(error,
              "sequence slices cannot be nested: source split (" +
                  std::to_string(src_split) + ") and destination split (" +
                  std::to_string(dst_split) + ") must divide one another");
    return false;
  }

  const KvLayoutIndex src_index(src_topology, src_redundancy);
  const KvLayoutIndex dst_index(dst_topology, dst_redundancy);
  const std::vector<HeadPair> pairs =
      head_pairs(src_index,
                 dst_index,
                 src_redundancy.head_class_count(),
                 dst_redundancy.head_class_count());

  const int32_t period =
      dst_split / greatest_common_divisor(src_split, dst_split);
  edges->reserve(static_cast<size_t>(pairs.size()) *
                 static_cast<size_t>(src_split) * static_cast<size_t>(period) *
                 static_cast<size_t>(dst_redundancy.replica_count()));

  std::vector<int32_t> replicas;
  for (const HeadPair& pair : pairs) {
    for (int32_t src_slice = 0; src_slice < src_split; ++src_slice) {
      // Replica deduplication: among the identical copies of (head class,
      // slice) only the replica-0 rank writes.
      int32_t src_rank = 0;
      if (!src_index.writer_of(
              /*dp_rank=*/0, pair.src_head_class, src_slice, &src_rank)) {
        set_error(error,
                  "source has no writer for head class " +
                      std::to_string(pair.src_head_class) + " slice " +
                      std::to_string(src_slice));
        return false;
      }
      const std::vector<int32_t> dst_slices =
          destination_slices(src_split, dst_split, src_slice);
      for (int32_t dst_slice : dst_slices) {
        // Every replica has to be filled: the copies live in different ranks'
        // memory even though they are byte identical.
        if (!dst_index.replicas_of(
                /*dp_rank=*/0, pair.dst_head_class, dst_slice, &replicas)) {
          set_error(error,
                    "destination has no rank for head class " +
                        std::to_string(pair.dst_head_class) + " slice " +
                        std::to_string(dst_slice));
          return false;
        }
        for (int32_t dst_rank : replicas) {
          RouteEdge edge;
          edge.src_local_rank = src_rank;
          edge.dst_local_rank = dst_rank;
          edge.head_begin = pair.head_begin;
          edge.head_end = pair.head_end;
          edge.src_slice = src_slice;
          edge.dst_slice = dst_slice;
          edges->emplace_back(edge);
        }
      }
    }
  }
  return true;
}

bool PdRouteTable::validate(const std::vector<RouteEdge>& edges,
                            const KvTopology& src_topology,
                            const GroupTopology& src_group,
                            const KvTopology& dst_topology,
                            const GroupTopology& dst_group,
                            std::string* error) {
  KvRedundancy src_redundancy;
  KvRedundancy dst_redundancy;
  if (!derive_side(src_topology, src_group, "source", &src_redundancy, error) ||
      !derive_side(
          dst_topology, dst_group, "destination", &dst_redundancy, error)) {
    return false;
  }
  const KvLayoutIndex src_index(src_topology, src_redundancy);
  const KvLayoutIndex dst_index(dst_topology, dst_redundancy);
  const int32_t src_ranks = local_rank_count(src_topology);
  const int32_t dst_ranks = local_rank_count(dst_topology);
  const int32_t src_split = src_redundancy.split();
  const int32_t dst_split = dst_redundancy.split();
  const int32_t heads = dst_group.global_head_count;

  // A destination rank is fed by several source ranks, one per source slice, so
  // coverage is counted per (destination rank, source slice, global head).
  std::vector<uint8_t> coverage(static_cast<size_t>(dst_ranks) *
                                    static_cast<size_t>(src_split) *
                                    static_cast<size_t>(heads),
                                0);
  // The route must also deduplicate the source: one writer per (head, slice).
  std::vector<int32_t> writers(
      static_cast<size_t>(src_split) * static_cast<size_t>(heads), -1);

  for (const RouteEdge& edge : edges) {
    if (edge.src_local_rank < 0 || edge.src_local_rank >= src_ranks) {
      set_error(error,
                "edge source rank " + std::to_string(edge.src_local_rank) +
                    " is outside the instance");
      return false;
    }
    if (edge.dst_local_rank < 0 || edge.dst_local_rank >= dst_ranks) {
      set_error(error,
                "edge destination rank " + std::to_string(edge.dst_local_rank) +
                    " is outside the instance");
      return false;
    }
    if (edge.src_slice < 0 || edge.src_slice >= src_split) {
      set_error(error,
                "edge source slice " + std::to_string(edge.src_slice) +
                    " is outside the source group split");
      return false;
    }
    if (edge.dst_slice < 0 || edge.dst_slice >= dst_split) {
      set_error(error,
                "edge destination slice " + std::to_string(edge.dst_slice) +
                    " is outside the destination group split");
      return false;
    }
    if (edge.head_begin < 0 || edge.head_end > heads ||
        edge.head_begin >= edge.head_end) {
      set_error(error,
                "edge head range [" + std::to_string(edge.head_begin) + ", " +
                    std::to_string(edge.head_end) + ") is outside [0, " +
                    std::to_string(heads) + ")");
      return false;
    }

    // Both ranks must own the range, otherwise the head offsets derived while
    // binding would address heads the rank does not hold.
    const int32_t src_head_class =
        src_index.head_class_of(edge.src_local_rank % src_topology.tp_size);
    const int32_t dst_head_class =
        dst_index.head_class_of(edge.dst_local_rank % dst_topology.tp_size);
    if (edge.head_begin < src_index.head_begin(src_head_class) ||
        edge.head_end > src_index.head_end(src_head_class) ||
        edge.head_begin < dst_index.head_begin(dst_head_class) ||
        edge.head_end > dst_index.head_end(dst_head_class)) {
      set_error(error,
                "edge head range straddles the head class of one of its ranks");
      return false;
    }

    // The slices an edge connects are properties of its ranks, and the block
    // mapping must connect them.
    if (src_index.slice_of(edge.src_local_rank / src_topology.tp_size,
                           edge.src_local_rank % src_topology.tp_size) !=
        edge.src_slice) {
      set_error(error, "edge source slice does not match its source rank");
      return false;
    }
    if (dst_index.slice_of(edge.dst_local_rank / dst_topology.tp_size,
                           edge.dst_local_rank % dst_topology.tp_size) !=
        edge.dst_slice) {
      set_error(error,
                "edge destination slice does not match its destination rank");
      return false;
    }
    const std::vector<int32_t> reachable =
        destination_slices(src_split, dst_split, edge.src_slice);
    if (std::find(reachable.begin(), reachable.end(), edge.dst_slice) ==
        reachable.end()) {
      set_error(error,
                "edge connects source slice " + std::to_string(edge.src_slice) +
                    " to an unreachable destination slice " +
                    std::to_string(edge.dst_slice));
      return false;
    }

    for (int32_t head = edge.head_begin; head < edge.head_end; ++head) {
      const size_t key = (static_cast<size_t>(edge.dst_local_rank) *
                              static_cast<size_t>(src_split) +
                          static_cast<size_t>(edge.src_slice)) *
                             static_cast<size_t>(heads) +
                         static_cast<size_t>(head);
      if (coverage[key] != 0) {
        set_error(error,
                  "destination rank " + std::to_string(edge.dst_local_rank) +
                      " source slice " + std::to_string(edge.src_slice) +
                      " head " + std::to_string(head) + " is written twice");
        return false;
      }
      coverage[key] = 1;

      const size_t writer_key =
          static_cast<size_t>(edge.src_slice) * static_cast<size_t>(heads) +
          static_cast<size_t>(head);
      if (writers[writer_key] < 0) {
        writers[writer_key] = edge.src_local_rank;
      } else if (writers[writer_key] != edge.src_local_rank) {
        set_error(error,
                  "source slice " + std::to_string(edge.src_slice) + " head " +
                      std::to_string(head) + " has two writers");
        return false;
      }
    }
  }

  for (int32_t rank = 0; rank < dst_ranks; ++rank) {
    const int32_t head_class =
        dst_index.head_class_of(rank % dst_topology.tp_size);
    const int32_t owned_begin = dst_index.head_begin(head_class);
    const int32_t owned_end = dst_index.head_end(head_class);
    const int32_t slice = dst_index.slice_of(rank / dst_topology.tp_size,
                                             rank % dst_topology.tp_size);
    for (int32_t src_slice = 0; src_slice < src_split; ++src_slice) {
      const std::vector<int32_t> reachable =
          destination_slices(src_split, dst_split, src_slice);
      const bool expected =
          std::find(reachable.begin(), reachable.end(), slice) !=
          reachable.end();
      for (int32_t head = 0; head < heads; ++head) {
        const bool owned = head >= owned_begin && head < owned_end;
        const int32_t count = coverage[(static_cast<size_t>(rank) *
                                            static_cast<size_t>(src_split) +
                                        static_cast<size_t>(src_slice)) *
                                           static_cast<size_t>(heads) +
                                       static_cast<size_t>(head)];
        if (expected && owned && count != 1) {
          set_error(error,
                    "destination rank " + std::to_string(rank) +
                        " source slice " + std::to_string(src_slice) +
                        " head " + std::to_string(head) + " has " +
                        std::to_string(count) + " writers instead of 1");
          return false;
        }
        if ((!expected || !owned) && count != 0) {
          set_error(error,
                    "route writes head " + std::to_string(head) +
                        " of source "
                        "slice " +
                        std::to_string(src_slice) + " onto rank " +
                        std::to_string(rank) + " which does not receive it");
          return false;
        }
      }
    }
  }

  // Every source (slice, head) the instance holds must be routed somewhere.
  for (int32_t src_slice = 0; src_slice < src_split; ++src_slice) {
    for (int32_t head = 0; head < heads; ++head) {
      if (writers[static_cast<size_t>(src_slice) * static_cast<size_t>(heads) +
                  static_cast<size_t>(head)] < 0) {
        set_error(error,
                  "source slice " + std::to_string(src_slice) + " head " +
                      std::to_string(head) + " has no writer");
        return false;
      }
    }
  }
  return true;
}

}  // namespace xllm
