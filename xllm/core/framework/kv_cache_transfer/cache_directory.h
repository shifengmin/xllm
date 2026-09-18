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
#include "framework/kv_cache_transfer/route_binder.h"

namespace xllm {

// Model-side declaration of one cache tensor family: the instance topology its
// tensors live in, plus the group geometry a published manifest cannot express.
//
// A manifest describes bytes -- shape, stride, spans -- and deliberately
// carries no block identity (see the verification plan §6.3). It does not say
// how many global heads a group exposes, whether the group is sequence scoped,
// or whether it keeps every canonical block on every rank, and those are
// exactly the inputs of the redundancy derivation. Declaring them here and
// reconciling them against the descriptor is what keeps a layout change from
// silently changing the route.
//
// One declaration covers every layer of one (cache namespace, role, group id):
// the geometry of a cache tensor family is a property of the model, not of the
// layer. Roles are opaque ids shared with the model side; this layer never
// interprets them.
struct CacheTensorDeclaration {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int32_t role = 0;
  int32_t group_id = 0;
  // Topology of the instance that owns the tensors. For the MAIN namespace it
  // must equal the coordinates the manifest publishes; a speculative draft body
  // declares its own, because the manifest coordinates always describe MAIN.
  KvTopology topology;
  // group.head_bytes is read from the descriptor. A non-zero declared value is
  // checked against it, which lets a caller pin the width of one head.
  GroupTopology group;
};

// Row bases of one page-mapped (GlobalXTensor) cache tensor, indexed by
// physical row. Only tensors whose manifest sets explicit_resource_offsets need
// an entry; supplying one for any other tensor is an error.
struct CacheRowBases {
  CacheNamespace cache_namespace = CacheNamespace::MAIN;
  int64_t layer_id = 0;
  int32_t role = 0;
  int32_t group_id = 0;
  std::vector<uint64_t> row_offsets;
};

// Translates the cache layout a peer published into the physical view the
// binder addresses, and reconciles it with what the model declares.
//
// This is the seam between the wire representation of a peer's cache and the
// peer-independent routing model: afterwards every routing decision is a pure
// function of checked numbers. Because the binder addresses a cache resource as
// `unit * local_heads * head_bytes + (head - first_local_head) * head_bytes`,
// a descriptor is accepted only when it provably describes that layout; the
// span arithmetic is what makes the acceptance a proof rather than a
// convention.
class PeerDirectory final {
 public:
  // Interprets `manifest` and produces one view per published cache tensor.
  // `declarations` must cover every tensor the manifest publishes, and
  // `row_bases` must cover exactly the tensors that address their rows through
  // explicit offsets. Returns false and fills `error` on the first
  // inconsistency; `directory` is left empty in that case.
  static bool describe(const WorkerCacheLayoutManifest& manifest,
                       const std::vector<CacheTensorDeclaration>& declarations,
                       const std::vector<CacheRowBases>& row_bases,
                       PeerDirectory* directory,
                       std::string* error);

  // The view of one cache tensor, or nullptr when this peer does not publish
  // it.
  const PeerCacheView* find(CacheNamespace cache_namespace,
                            int64_t layer_id,
                            int32_t role,
                            int32_t group_id) const;

  size_t size() const { return views_.size(); }
  const PeerCacheView& at(size_t index) const { return views_.at(index); }

 private:
  std::vector<PeerCacheView> views_;
};

}  // namespace xllm
