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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {

namespace {

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool same_topology(const KvTopology& lhs, const KvTopology& rhs) {
  return lhs.dp_size == rhs.dp_size && lhs.cp_size == rhs.cp_size &&
         lhs.tp_size == rhs.tp_size && lhs.kv_split_size == rhs.kv_split_size &&
         lhs.tokens_per_block == rhs.tokens_per_block;
}

bool same_group_topology(const GroupTopology& lhs, const GroupTopology& rhs) {
  return lhs.global_head_count == rhs.global_head_count &&
         lhs.head_bytes == rhs.head_bytes &&
         lhs.sequence_scoped == rhs.sequence_scoped &&
         lhs.full_sequence_replica == rhs.full_sequence_replica;
}

// A cache tensor family: one published tensor of one (namespace, layer, role,
// group). Two peers route a family together only when both published it.
bool same_family(const PeerCacheView& lhs, const PeerCacheView& rhs) {
  return lhs.entry.cache_namespace == rhs.entry.cache_namespace &&
         lhs.entry.layer_id == rhs.entry.layer_id &&
         lhs.entry.role == rhs.entry.role &&
         lhs.entry.group_id == rhs.entry.group_id;
}

std::string family_label(const PeerCacheView& view) {
  return "role " + std::to_string(view.entry.role) + " group " +
         std::to_string(view.entry.group_id) + " (layer " +
         std::to_string(view.entry.layer_id) + ")";
}

bool ascending_blocks(const std::vector<int64_t>& canonical_blocks,
                      std::string* error) {
  for (size_t index = 0; index < canonical_blocks.size(); ++index) {
    if (canonical_blocks[index] < 0) {
      set_error(error, "canonical block must not be negative");
      return false;
    }
    if (index > 0 && canonical_blocks[index] <= canonical_blocks[index - 1]) {
      set_error(error,
                "canonical blocks must be strictly ascending: " +
                    std::to_string(canonical_blocks[index - 1]) + " then " +
                    std::to_string(canonical_blocks[index]));
      return false;
    }
  }
  return true;
}

// The distinct cache tensor families a view list covers, in first-seen order.
std::vector<const PeerCacheView*> distinct_families(
    const std::vector<PeerCacheView>& views) {
  std::vector<const PeerCacheView*> families;
  families.reserve(views.size());
  for (size_t index = 0; index < views.size(); ++index) {
    bool seen = false;
    for (size_t earlier = 0; earlier < index; ++earlier) {
      if (same_family(views[index], views[earlier])) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      families.emplace_back(&views[index]);
    }
  }
  return families;
}

// The view one rank of `views` published for the family `family`. `*out` is
// nullptr when that rank published none.
//
// A view that does not name its rank is accepted only when it is the only such
// view of the family: a draft body's published coordinates describe MAIN, so it
// cannot say which rank published it, and one candidate is the most that can be
// resolved. Anything else would address whichever rank happened to come first.
// The function returns false only on that ambiguity.
bool find_rank_view(const std::vector<PeerCacheView>& views,
                    const PeerCacheView& family,
                    int32_t rank,
                    const PeerCacheView** out,
                    std::string* error) {
  *out = nullptr;
  int32_t unnamed_count = 0;
  const PeerCacheView* unnamed = nullptr;
  for (const PeerCacheView& view : views) {
    if (!same_family(view, family)) {
      continue;
    }
    if (view.local_rank == rank) {
      *out = &view;
      return true;
    }
    if (view.local_rank >= 0) {
      continue;
    }
    unnamed = &view;
    ++unnamed_count;
  }
  if (unnamed_count > 1) {
    set_error(error,
              family_label(family) + ": " + std::to_string(unnamed_count) +
                  " published views do not name their rank, so peer rank " +
                  std::to_string(rank) + " cannot be addressed");
    return false;
  }
  *out = unnamed;
  return true;
}

bool derive_group(const PeerCacheView& view,
                  const char* side,
                  KvRedundancy* redundancy,
                  std::string* error) {
  std::string reason;
  if (!KvRedundancy::derive(view.topology, view.group, redundancy, &reason)) {
    set_error(error, family_label(view) + ": " + side + " side: " + reason);
    return false;
  }
  return true;
}

// The sequence slice a rank holds, i.e. its runtime DCP rank.
int32_t rank_slice(const KvTopology& topology,
                   const KvRedundancy& redundancy,
                   int32_t local_rank) {
  const KvLayoutIndex index(topology, redundancy);
  return index.slice_of(local_rank / topology.tp_size,
                        local_rank % topology.tp_size);
}

// RouteBinder binds destination-last -- its `local` view is the writer -- which
// is the PUSH orientation. PULL is the same pair of ranges read the other way,
// so only the two halves swap.
void orient_for_pull(std::vector<RouteRegion>* regions) {
  for (RouteRegion& region : *regions) {
    std::swap(region.local_buffer_id, region.remote_buffer_id);
    std::swap(region.local_offset, region.remote_offset);
  }
}

// Rewrites the model-side declarations so they describe the peer instead of us.
//
// The parallel coordinates of a published layout are a property of the instance
// that published it, not of the model: the pilot pushes PREFILL cp=2/kv_split=2
// slices into a DECODE instance that is cp=1/kv_split=1, which is the whole
// point of the canonical route. Requiring the two to be equal -- which is what
// passing our own declarations does -- rejects exactly the pair the route
// exists to connect ("the declared topology differs from the coordinates the
// peer published"). Only the coordinates are adopted: the group geometry and
// the token capacity per block stay ours, because both instances run the same
// model and PeerDirectory::describe still reconciles them against the
// descriptor the peer published, so a real disagreement is still refused.
//
// SPEC_DRAFT families keep their own topology: the manifest coordinates always
// describe MAIN (see CacheTensorDeclaration).
std::vector<CacheTensorDeclaration> declarations_for_peer(
    const std::vector<CacheTensorDeclaration>& declarations,
    const ParallelCoordinates& coordinates) {
  std::vector<CacheTensorDeclaration> adjusted = declarations;
  for (CacheTensorDeclaration& declaration : adjusted) {
    if (declaration.cache_namespace != CacheNamespace::MAIN) {
      continue;
    }
    declaration.topology.dp_size = coordinates.dp_size;
    declaration.topology.cp_size = coordinates.cp_size;
    declaration.topology.tp_size = coordinates.tp_size;
    declaration.topology.kv_split_size = coordinates.kv_split_size;
  }
  return adjusted;
}

}  // namespace

bool flatten_route_for_layers(
    const std::vector<RouteLeg>& legs,
    const std::unordered_map<uint64_t, int64_t>& layer_of_buffer,
    std::vector<RouteLayerBatch>* batches,
    std::string* error) {
  if (batches == nullptr) {
    set_error(error, "the layer batch output must not be null");
    return false;
  }
  batches->clear();

  // A batch is one (layer, peer, source rank) triple: the transport call
  // carries a single peer address, so two legs to the same peer stay apart when
  // they come from different ranks.
  std::vector<RouteLayerBatch> ordered;
  for (const RouteLeg& leg : legs) {
    for (const RouteRegion& region : leg.regions) {
      const auto layer_it = layer_of_buffer.find(region.local_buffer_id);
      if (layer_it == layer_of_buffer.end()) {
        set_error(error,
                  "region of leg to peer rank " +
                      std::to_string(leg.peer_local_rank) + " names buffer " +
                      std::to_string(region.local_buffer_id) +
                      ", which belongs to no layer");
        return false;
      }
      const int64_t layer_id = layer_it->second;
      auto batch_it =
          std::find_if(ordered.begin(),
                       ordered.end(),
                       [layer_id, &leg](const RouteLayerBatch& batch) {
                         return batch.layer_id == layer_id &&
                                batch.peer_addr == leg.peer_addr;
                       });
      if (batch_it == ordered.end()) {
        RouteLayerBatch batch;
        batch.layer_id = layer_id;
        batch.peer_addr = leg.peer_addr;
        ordered.emplace_back(std::move(batch));
        batch_it = std::prev(ordered.end());
      }
      batch_it->regions.emplace_back(region);
    }
  }

  // Layer order is the order the push loop synchronizes in; the legs of one
  // layer keep the plan's order.
  std::stable_sort(ordered.begin(),
                   ordered.end(),
                   [](const RouteLayerBatch& lhs, const RouteLayerBatch& rhs) {
                     return lhs.layer_id < rhs.layer_id;
                   });
  for (RouteLayerBatch& batch : ordered) {
    if (!batch.regions.empty()) {
      batches->emplace_back(std::move(batch));
    }
  }
  return true;
}

bool build_route_peer(
    const std::vector<std::string>& instance_addrs,
    int32_t dp_rank,
    int32_t local_rank_count,
    const std::vector<const WorkerCacheLayoutManifest*>& manifests,
    const std::vector<CacheTensorDeclaration>& declarations,
    const std::vector<CacheRowBases>& row_bases,
    RoutePeer* peer,
    std::string* error) {
  if (peer == nullptr) {
    set_error(error, "the route peer output must not be null");
    return false;
  }
  peer->addrs.clear();
  peer->views.clear();
  if (local_rank_count <= 0) {
    set_error(error, "the peer instance must own at least one rank");
    return false;
  }
  if (dp_rank < 0) {
    set_error(error, "the destination DP rank must not be negative");
    return false;
  }
  if (manifests.size() != static_cast<size_t>(local_rank_count)) {
    set_error(error,
              "the peer instance has " + std::to_string(local_rank_count) +
                  " ranks but " + std::to_string(manifests.size()) +
                  " cache layouts were supplied");
    return false;
  }
  const int64_t begin = static_cast<int64_t>(dp_rank) * local_rank_count;
  if (begin + local_rank_count > static_cast<int64_t>(instance_addrs.size())) {
    set_error(error,
              "the peer instance publishes " +
                  std::to_string(instance_addrs.size()) +
                  " addresses, which do not cover DP group " +
                  std::to_string(dp_rank));
    return false;
  }

  peer->addrs.reserve(static_cast<size_t>(local_rank_count));
  for (int32_t local_rank = 0; local_rank < local_rank_count; ++local_rank) {
    const WorkerCacheLayoutManifest* manifest =
        manifests[static_cast<size_t>(local_rank)];
    if (manifest == nullptr) {
      set_error(error,
                "peer rank " + std::to_string(local_rank) +
                    " published no cache layout");
      return false;
    }
    PeerDirectory directory;
    std::string reason;
    // The peer's own coordinates decide how its buffers are laid out, so
    // reconcile its manifest against declarations that describe it rather than
    // against ours.
    const std::vector<CacheTensorDeclaration> peer_declarations =
        declarations_for_peer(declarations, manifest->coordinates);
    if (!PeerDirectory::describe(
            *manifest, peer_declarations, row_bases, &directory, &reason)) {
      set_error(error,
                "the cache layout of peer rank " + std::to_string(local_rank) +
                    " does not match the model: " + reason);
      return false;
    }
    for (size_t index = 0; index < directory.size(); ++index) {
      const PeerCacheView& view = directory.at(index);
      // A view that names its rank has to be the rank whose layout produced it,
      // otherwise the caller filed the manifests in the wrong order and the
      // route would address another rank's buffer.
      if (view.local_rank >= 0 && view.local_rank != local_rank) {
        set_error(error,
                  "the layout filed under peer rank " +
                      std::to_string(local_rank) + " describes rank " +
                      std::to_string(view.local_rank));
        return false;
      }
      peer->views.emplace_back(view);
    }
    peer->addrs.emplace_back(
        instance_addrs[static_cast<size_t>(begin + local_rank)]);
  }
  return true;
}

bool parse_pd_route_mode(const std::string& value, PdRouteMode* mode) {
  if (mode == nullptr) {
    return false;
  }
  if (value == "legacy") {
    *mode = PdRouteMode::LEGACY;
    return true;
  }
  if (value == "canonical") {
    *mode = PdRouteMode::CANONICAL;
    return true;
  }
  return false;
}

const char* pd_route_mode_name(PdRouteMode mode) {
  switch (mode) {
    case PdRouteMode::LEGACY:
      return "legacy";
    case PdRouteMode::CANONICAL:
      return "canonical";
  }
  return "legacy";
}

bool PdRouteCache::find_or_build(const KvTopology& src_topology,
                                 const GroupTopology& src_group,
                                 const KvTopology& dst_topology,
                                 const GroupTopology& dst_group,
                                 const std::vector<RouteEdge>** edges,
                                 std::string* error) {
  if (edges == nullptr) {
    set_error(error, "route cache output must not be null");
    return false;
  }
  *edges = nullptr;
  for (const Entry& entry : entries_) {
    if (same_topology(entry.src_topology, src_topology) &&
        same_group_topology(entry.src_group, src_group) &&
        same_topology(entry.dst_topology, dst_topology) &&
        same_group_topology(entry.dst_group, dst_group)) {
      *edges = &entry.edges;
      return true;
    }
  }

  Entry entry;
  entry.src_topology = src_topology;
  entry.src_group = src_group;
  entry.dst_topology = dst_topology;
  entry.dst_group = dst_group;
  std::string reason;
  if (!PdRouteTable::build(src_topology,
                           src_group,
                           dst_topology,
                           dst_group,
                           &entry.edges,
                           &reason)) {
    set_error(error, "the route table cannot be built: " + reason);
    return false;
  }
  // A table is a pure function of the two shapes, so a mistake in it is a
  // property of the shapes and would otherwise be re-discovered -- or missed --
  // once per peer pair. Validate it here, once.
  if (!PdRouteTable::validate(entry.edges,
                              src_topology,
                              src_group,
                              dst_topology,
                              dst_group,
                              &reason)) {
    set_error(error, "the route table is not complete: " + reason);
    return false;
  }

  entries_.emplace_back(std::move(entry));
  *edges = &entries_.back().edges;
  return true;
}

bool PdRouteTransfer::plan(PdRouteCache* cache,
                           RouteOpcode opcode,
                           int32_t local_rank,
                           const std::vector<int64_t>& canonical_blocks,
                           const std::vector<PeerCacheView>& local,
                           const RoutePeer& peer,
                           std::vector<RouteLeg>* legs,
                           std::string* error) {
  if (cache == nullptr) {
    set_error(error, "route cache must not be null");
    return false;
  }
  if (legs == nullptr) {
    set_error(error, "leg output must not be null");
    return false;
  }
  legs->clear();
  if (local_rank < 0) {
    set_error(error, "the local rank within the DP group must not be negative");
    return false;
  }
  if (!ascending_blocks(canonical_blocks, error)) {
    return false;
  }
  if (canonical_blocks.empty()) {
    return true;
  }

  const bool local_is_reader = opcode == RouteOpcode::PULL;
  const std::vector<const PeerCacheView*> families = distinct_families(local);
  for (const PeerCacheView* family : families) {
    const PeerCacheView* local_view = nullptr;
    if (!find_rank_view(local, *family, local_rank, &local_view, error)) {
      return false;
    }
    if (local_view == nullptr) {
      continue;
    }

    // The peer side of this family decides the table's other shape. Its ranks
    // must agree, otherwise there is no single shape to route against.
    const PeerCacheView* peer_shape = nullptr;
    for (const PeerCacheView& peer_view : peer.views) {
      if (!same_family(peer_view, *local_view)) {
        continue;
      }
      if (peer_shape == nullptr) {
        peer_shape = &peer_view;
        continue;
      }
      if (!same_topology(peer_shape->topology, peer_view.topology) ||
          !same_group_topology(peer_shape->group, peer_view.group)) {
        set_error(error,
                  family_label(*local_view) +
                      ": the peer published this family with two different "
                      "shapes, so it cannot be routed");
        return false;
      }
    }
    if (peer_shape == nullptr) {
      continue;
    }

    const PeerCacheView& writer_view =
        local_is_reader ? *peer_shape : *local_view;
    const PeerCacheView& reader_view =
        local_is_reader ? *local_view : *peer_shape;
    KvRedundancy writer_redundancy;
    KvRedundancy reader_redundancy;
    if (!derive_group(writer_view, "writer", &writer_redundancy, error) ||
        !derive_group(reader_view, "reader", &reader_redundancy, error)) {
      return false;
    }
    const int32_t writer_split = writer_redundancy.split();
    const int32_t reader_split = reader_redundancy.split();

    const std::vector<RouteEdge>* edges = nullptr;
    if (!cache->find_or_build(writer_view.topology,
                              writer_view.group,
                              reader_view.topology,
                              reader_view.group,
                              &edges,
                              error)) {
      return false;
    }

    // Which peer ranks this rank pairs with, taken from the route itself. A
    // cross product of the two view lists would include pairs whose head
    // classes do not intersect; those carry no bytes and are not peers.
    std::vector<int32_t> counter_ranks;
    for (const RouteEdge& edge : *edges) {
      const int32_t anchor =
          local_is_reader ? edge.dst_local_rank : edge.src_local_rank;
      if (anchor != local_rank) {
        continue;
      }
      const int32_t counter =
          local_is_reader ? edge.src_local_rank : edge.dst_local_rank;
      if (std::find(counter_ranks.begin(), counter_ranks.end(), counter) ==
          counter_ranks.end()) {
        counter_ranks.emplace_back(counter);
      }
    }
    std::sort(counter_ranks.begin(), counter_ranks.end());

    const int32_t local_slice =
        rank_slice(local_view->topology,
                   local_is_reader ? reader_redundancy : writer_redundancy,
                   local_rank);
    // Every canonical block this rank is the reader (PULL) or the writer
    // (PUSH) of has to end up in exactly one leg; the bitmap is what turns
    // "the route covered the request" from a hope into a check.
    std::vector<uint8_t> covered(canonical_blocks.size(), 0);
    const size_t family_leg_begin = legs->size();

    for (int32_t counter : counter_ranks) {
      const PeerCacheView* counter_view = nullptr;
      if (!find_rank_view(
              peer.views, *local_view, counter, &counter_view, error)) {
        return false;
      }
      if (counter_view == nullptr) {
        set_error(error,
                  family_label(*local_view) + ": the peer rank " +
                      std::to_string(counter) +
                      " the route pairs with published no view, so its buffer "
                      "cannot be addressed");
        return false;
      }
      const PeerCacheView& leg_writer_view =
          local_is_reader ? *counter_view : *local_view;
      const PeerCacheView& leg_reader_view =
          local_is_reader ? *local_view : *counter_view;
      const int32_t counter_slice =
          rank_slice(counter_view->topology,
                     local_is_reader ? writer_redundancy : reader_redundancy,
                     counter);
      const int32_t writer_slice =
          local_is_reader ? counter_slice : local_slice;
      const int32_t reader_slice =
          local_is_reader ? local_slice : counter_slice;

      // A canonical block moves between the slice that holds it and the slice
      // that needs it; a block whose two slices belong to other ranks is some
      // other leg's work.
      std::vector<int64_t> selected;
      std::vector<size_t> selected_indices;
      for (size_t index = 0; index < canonical_blocks.size(); ++index) {
        const int64_t block = canonical_blocks[index];
        if (block % writer_split != writer_slice ||
            block % reader_split != reader_slice) {
          continue;
        }
        selected.emplace_back(block);
        selected_indices.emplace_back(index);
      }
      if (selected.empty()) {
        continue;
      }

      std::vector<RouteRegion> regions;
      std::string reason;
      if (!RouteBinder::bind(
              *edges,
              /*dst_local_rank=*/local_is_reader ? local_rank : counter,
              selected,
              leg_writer_view,
              leg_reader_view,
              &regions,
              &reason)) {
        set_error(error, family_label(*local_view) + ": " + reason);
        return false;
      }
      for (size_t index : selected_indices) {
        covered[index] = 1;
      }
      if (regions.empty()) {
        continue;
      }
      if (local_is_reader) {
        orient_for_pull(&regions);
      }

      RouteLeg leg;
      leg.opcode = opcode;
      leg.local_rank = local_rank;
      leg.peer_local_rank = counter;
      if (counter >= static_cast<int32_t>(peer.addrs.size())) {
        set_error(error,
                  family_label(*local_view) + ": peer rank " +
                      std::to_string(counter) +
                      " has no address in the peer instance");
        return false;
      }
      leg.peer_addr = peer.addrs[static_cast<size_t>(counter)];
      leg.regions = std::move(regions);
      legs->emplace_back(std::move(leg));
    }

    // Completeness: this rank has to end up with everything it is the reader
    // of, and a rank that is not the writer of its slice must move nothing (the
    // redundant copies are byte identical, so a replica pushing would only
    // duplicate traffic).
    bool is_writer = false;
    if (!local_is_reader) {
      const KvLayoutIndex local_index(local_view->topology, writer_redundancy);
      int32_t writer_rank = -1;
      const int32_t local_tp_rank = local_rank % local_view->topology.tp_size;
      is_writer =
          local_index.writer_of(/*dp_rank=*/0,
                                local_index.head_class_of(local_tp_rank),
                                local_slice,
                                &writer_rank) &&
          writer_rank == local_rank;
    }
    if (!local_is_reader && !is_writer) {
      if (legs->size() != family_leg_begin) {
        set_error(error,
                  family_label(*local_view) + ": rank " +
                      std::to_string(local_rank) +
                      " is a redundant copy of this group's writer but the "
                      "route made it push");
        return false;
      }
      continue;
    }
    const int32_t local_split = local_is_reader ? reader_split : writer_split;
    for (size_t index = 0; index < canonical_blocks.size(); ++index) {
      if (covered[index] == 0 &&
          canonical_blocks[index] % local_split == local_slice) {
        set_error(error,
                  family_label(*local_view) + ": canonical block " +
                      std::to_string(canonical_blocks[index]) +
                      (local_is_reader ? " has no writer into rank "
                                       : " has no reader from rank ") +
                      std::to_string(local_rank));
        return false;
      }
    }
  }
  return true;
}

bool PdRouteTransfer::transfer(PdRouteCache* cache,
                               RouteOpcode opcode,
                               int32_t local_rank,
                               const std::vector<int64_t>& canonical_blocks,
                               const std::vector<PeerCacheView>& local,
                               const RoutePeer& peer,
                               const MoveFn& move,
                               std::vector<RouteLeg>* legs,
                               std::string* error) {
  std::vector<RouteLeg> planned;
  std::vector<RouteLeg>* output = legs != nullptr ? legs : &planned;
  if (!plan(cache,
            opcode,
            local_rank,
            canonical_blocks,
            local,
            peer,
            output,
            error)) {
    return false;
  }
  return apply(*output, move, error);
}

bool PdRouteTransfer::apply(const std::vector<RouteLeg>& legs,
                            const MoveFn& move,
                            std::string* error) {
  if (!move) {
    set_error(error, "the route transport must not be empty");
    return false;
  }
  for (const RouteLeg& leg : legs) {
    if (leg.regions.empty()) {
      continue;
    }
    if (leg.peer_addr.empty()) {
      set_error(error,
                "a leg to peer rank " + std::to_string(leg.peer_local_rank) +
                    " has no peer address");
      return false;
    }
    if (!move(leg.peer_addr, leg.regions, leg.opcode)) {
      set_error(error,
                "the transport failed for peer rank " +
                    std::to_string(leg.peer_local_rank) + " at " +
                    leg.peer_addr);
      return false;
    }
  }
  return true;
}

}  // namespace xllm
