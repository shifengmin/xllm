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

#include <cstdint>
#include <string>
#include <vector>

#include "framework/kv_cache_transfer/kv_redundancy.h"

namespace xllm {

// One directed edge of the KV route: "the source-local rank transfers the
// global heads [head_begin, head_end) of every canonical block whose source
// slice is src_slice into the canonical blocks whose destination slice is
// dst_slice on the destination-local rank".
//
// Ranks are local to one DP group: global_rank = dp * (cp_size * tp_size) +
// local_rank. The DP pairing is a scheduling decision (TransferKVInfo.dp_rank)
// and is deliberately absent here, so one table serves every DP pair and the
// edge count does not grow with the DP width.
struct RouteEdge {
  int32_t src_local_rank = 0;  // cp_rank * tp_size + tp_rank
  int32_t dst_local_rank = 0;  // cp_rank * tp_size + tp_rank
  int32_t head_begin = 0;      // global KV head, half open
  int32_t head_end = 0;
  int32_t src_slice = 0;  // sequence slice t on the source side
  int32_t dst_slice = 0;  // sequence slice t on the destination side
};

// The transfer route between two PD peers for one cache group.
//
// Routing is the cartesian product of two independent maps: head classes
// intersect over global head intervals, and canonical blocks move between
// sequence slices. Replicas are deduplicated on the source side (only the
// replica-0 rank writes) and enumerated in full on the destination side (every
// replica has to be filled).
//
// build() is a pure function of the two topologies, so both peers derive the
// same table independently. That is what turns the table into something the two
// sides can cross-check instead of a plan that has to be shipped over RPC.
class PdRouteTable final {
 public:
  // Enumerates every edge, ordered by (source head class, destination head
  // class, source slice, destination slice, destination replica).
  static bool build(const KvTopology& src_topology,
                    const GroupTopology& src_group,
                    const KvTopology& dst_topology,
                    const GroupTopology& dst_group,
                    std::vector<RouteEdge>* edges,
                    std::string* error);

  // Post-condition of build() re-derived from the two topologies: for every
  // destination rank and every (head, source slice) pair exactly one edge
  // carries it, no edge writes a head its destination rank does not own, and
  // every source slice has a single writer. O(edges) plus
  // O(dst_ranks * src_split * G) scratch space.
  static bool validate(const std::vector<RouteEdge>& edges,
                       const KvTopology& src_topology,
                       const GroupTopology& src_group,
                       const KvTopology& dst_topology,
                       const GroupTopology& dst_group,
                       std::string* error);
};

}  // namespace xllm
