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

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/kv_cache_transfer/cache_directory.h"
#include "framework/kv_cache_transfer/pd_route_table.h"
#include "framework/kv_cache_transfer/route_binder.h"

namespace xllm {

// Which implementation of the data plane moves the KV cache.
//
//   LEGACY     the rank-aligned strided remap (filter_kv_split_infos plus
//              rotate_dst_rank), which requires both peers to use the same
//              kv-split width and their ranks to line up one to one.
//   CANONICAL  the route tables, which derive the writer and the reader of
//              every canonical block from the two peers' layouts.
//
// The mode is chosen before execution and never falls back into the other path:
// a route that cannot be built has to fail, not silently downgrade.
enum class PdRouteMode : int8_t {
  LEGACY = 0,
  CANONICAL = 1,
};

// The mode a --pd_route value names. Returns false for anything else, so a typo
// fails instead of selecting a path the operator did not ask for.
bool parse_pd_route_mode(const std::string& value, PdRouteMode* mode);

// The name of a mode, i.e. the value that parses back to it.
const char* pd_route_mode_name(PdRouteMode mode);

// Which side of a PD pair initiates one transfer, and therefore how the two
// halves of every region are addressed.
//
//   PULL  the reader initiates: the transport reads from the peer, so
//         region.local_* is the reader's range and region.remote_* the
//         writer's. This is MooncakeTransferEngine::MoveOpcode::READ.
//   PUSH  the writer initiates: the transport writes to the peer, so
//         region.local_* is the writer's range and region.remote_* the
//         reader's. This is MoveOpcode::WRITE, and it is also the orientation
//         RouteBinder binds in.
//
// A leg is the same pair of byte ranges in both directions; only which side the
// local half names changes. That is why the two opcodes can be cross-checked
// against each other instead of each being trusted on its own.
enum class RouteOpcode : int8_t {
  PULL = 0,
  PUSH = 1,
};

// One peer instance of a PD pair.
struct RoutePeer {
  // addrs[local_rank] reaches local rank `local_rank` of the peer instance.
  std::vector<std::string> addrs;
  // Every cache tensor the peer instance published, from PeerDirectory. Views
  // may come from any of its ranks; PeerCacheView::local_rank says which.
  std::vector<PeerCacheView> views;
};

// Assembles the peer instance the canonical route moves against.
//
// `instance_addrs` is a peer instance's address list as the scheduler publishes
// it, indexed by the instance's *global* rank (`dp_rank * cp_size * tp_size +
// local_rank`). The route addresses ranks inside one DP group, so the entry for
// a local rank is read at `dp_rank * local_rank_count + local_rank`; using the
// local rank as the index would silently move bytes against another DP group's
// worker, which is why this conversion lives in one place instead of at every
// call site.
//
// `manifests` holds the cache layout each peer rank published, indexed by its
// local rank. Every rank is reconciled with the same model-side declarations,
// so a peer whose published layout disagrees with the model fails here rather
// than at the first transfer, and every view is checked to come from the rank
// it was filed under.
//
// The parallel coordinates inside those declarations are taken from the
// manifest being described, not from the caller: a peer is allowed to be
// configured differently from us -- that is what the canonical route is for --
// while the group geometry and the token capacity stay model-side and are still
// checked against the published descriptor.
bool build_route_peer(
    const std::vector<std::string>& instance_addrs,
    int32_t dp_rank,
    int32_t local_rank_count,
    const std::vector<const WorkerCacheLayoutManifest*>& manifests,
    const std::vector<CacheTensorDeclaration>& declarations,
    const std::vector<CacheRowBases>& row_bases,
    RoutePeer* peer,
    std::string* error);

// One (writer rank, reader rank) leg of the route, in transport-ready bytes.
struct RouteLeg {
  RouteOpcode opcode = RouteOpcode::PULL;
  // The rank on this side of the pair: the writer for PUSH, the reader for
  // PULL.
  int32_t local_rank = -1;
  int32_t peer_local_rank = -1;
  std::string peer_addr;
  std::vector<RouteRegion> regions;
};

// Route tables cached by the pair of (topology, group) they connect.
//
// The table is a pure function of the two sides' shapes, so one table serves
// every peer pair of the same shape and a request never rebuilds it (handover
// F8). Caching by peer instead would rebuild the same table once per
// destination worker and would also make the table unable to describe a pair
// whose peer has not been linked yet.
// One transport call the push loop makes: the byte regions to write to one
// peer while one layer is being published.
struct RouteLayerBatch {
  int64_t layer_id = 0;
  std::string peer_addr;
  std::vector<RouteRegion> regions;
};

// Flattens a planned route into the calls a layer-synchronized push makes.
//
// A region addresses a cache buffer, and every cache buffer belongs to exactly
// one layer, which is what makes the flattening possible: the route derives its
// regions from canonical blocks, and a canonical block spans every layer of the
// model. The output is ordered by ascending layer and, inside a layer, by the
// order the legs were planned in, so a caller can synchronize once per layer
// and then emit every peer's bytes for it before moving on. Layers with no
// regions produce no batch.
//
// `layer_of_buffer` maps the buffer id a region names to its layer; a region
// whose buffer is unknown is an error rather than a dropped write, because
// dropping it would leave the destination with a hole nothing checks.
bool flatten_route_for_layers(
    const std::vector<RouteLeg>& legs,
    const std::unordered_map<uint64_t, int64_t>& layer_of_buffer,
    std::vector<RouteLayerBatch>* batches,
    std::string* error);

class PdRouteCache final {
 public:
  // Returns the edges connecting `src` to `dst`, building and validating them
  // on first use. The pointer stays valid until this cache is destroyed; a
  // later call for a different shape may reallocate the returned vector's
  // storage, so callers must not hold it across another call.
  bool find_or_build(const KvTopology& src_topology,
                     const GroupTopology& src_group,
                     const KvTopology& dst_topology,
                     const GroupTopology& dst_group,
                     const std::vector<RouteEdge>** edges,
                     std::string* error);

  size_t size() const { return entries_.size(); }

 private:
  struct Entry {
    KvTopology src_topology;
    GroupTopology src_group;
    KvTopology dst_topology;
    GroupTopology dst_group;
    std::vector<RouteEdge> edges;
  };

  std::vector<Entry> entries_;
};

// The unified data-plane entry of the canonical route.
//
// One call plans and moves one request between this rank and one peer instance:
// the route table says which peer rank holds each canonical block and which
// one has to end up with it, RouteBinder turns that into byte ranges, and the
// transport moves them. Nothing here depends on the two sides having the same
// kv-split width or on their ranks lining up one to one -- the two properties
// the legacy filter_kv_split_infos / rotate_dst_rank pair silently assumes.
//
// The opcode is a parameter rather than a loop because a PD pair picks one
// direction operationally: a decode instance pulls, a prefill instance pushes.
class PdRouteTransfer final {
 public:
  // The transport, invoked once per leg with everything resolved: the peer
  // address, the byte ranges, and which side of the pair is local.
  using MoveFn = std::function<bool(const std::string& peer_addr,
                                    const std::vector<RouteRegion>& regions,
                                    RouteOpcode opcode)>;

  // Plans one opcode of one request for the rank `local_rank`.
  //
  // `canonical_blocks` are the canonical blocks of the request, ascending. A
  // canonical block is a *position* -- the request's `p`-th block through
  // sequence slice `j` -- and it is the only block identity both sides agree
  // on, so it is also the only input that has to be converted from a
  // request-local id: the caller converts its pool rows with
  // CanonicalBlock::canonical_of_row before calling here, rebasing them onto
  // the request's first block. Each side then resolves the position against its
  // own layout, which is why the two rows differ whenever the two splits do.
  //
  // `local` holds the views this side published and `peer` the ones the peer
  // instance published; both may cover several ranks. A family is routed only
  // when this rank published a view for it and at least one edge connects this
  // rank to a peer rank. Every leg is oriented for `opcode`, so the caller
  // hands the regions to the transport unchanged.
  static bool plan(PdRouteCache* cache,
                   RouteOpcode opcode,
                   int32_t local_rank,
                   const std::vector<int64_t>& canonical_blocks,
                   const std::vector<PeerCacheView>& local,
                   const RoutePeer& peer,
                   std::vector<RouteLeg>* legs,
                   std::string* error);

  // plan() followed by apply(): the single entry the data plane calls.
  static bool transfer(PdRouteCache* cache,
                       RouteOpcode opcode,
                       int32_t local_rank,
                       const std::vector<int64_t>& canonical_blocks,
                       const std::vector<PeerCacheView>& local,
                       const RoutePeer& peer,
                       const MoveFn& move,
                       std::vector<RouteLeg>* legs,
                       std::string* error);

  // Moves an already planned route: one transport call per non-empty leg, in
  // plan order. The first failed leg stops the route.
  static bool apply(const std::vector<RouteLeg>& legs,
                    const MoveFn& move,
                    std::string* error);
};

}  // namespace xllm
