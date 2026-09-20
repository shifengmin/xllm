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

#include "framework/kv_cache/cache_layout_types.h"
#include "framework/kv_cache/kv_cache_tensor_role.h"
#include "framework/kv_cache_transfer/cache_layout.h"
#include "framework/kv_cache_transfer/kv_redundancy.h"
#include "framework/kv_cache_transfer/route_binder.h"

namespace xllm {

// Model-side group geometry of one cache tensor family, derived from the same
// inputs the descriptor builder uses.
//
// This is the declaration half of describe_cache_tensor(): the descriptor says
// how the bytes of one rank are laid out, and this says how many logical heads
// the group exposes, whether it has a block dimension at all, and whether the
// whole sequence stays on every rank. The two are derived from the same role
// and the same layout context, so a layout change that moves one has to move
// the other.
//
// Only the families whose geometry the model actually pins are declared. A role
// whose geometry is unknown is refused rather than guessed: a wrong head count
// or a wrong sequence scope routes the wrong blocks, and neither the manifest
// nor the route can detect it afterwards.
//
// `group->head_bytes` is left at 0: the descriptor owns the byte width of one
// head, and PeerDirectory fills it from there.
//
// Two decisions are worth spelling out:
//
//   - An MLA instance publishes every cache tensor as one whole resource, so
//   the
//     attention group exposes a single latent head per rank no matter how many
//     KV heads the model text declares. That is what keeps `local_heads == 1`,
//     which is the admission rule for a whole-resource descriptor and also the
//     pilot's `Hc = 1, D_tp = tp` contract.
//   - The DSA indexer pool (INDEX and its scale, which the allocator sizes from
//     the index block count) keeps every canonical block on every rank, because
//     its top-k reads historical gate and valid values. Declaring it split
//     would leave the destination's other rows uninitialised without any check
//     firing. The reverse mistake is caught: the destination buffer of a
//     genuinely split pool is too short for the rows the declaration implies,
//     and the binder rejects the row.
bool declare_cache_group(const CacheTensorLayoutContext& context,
                         KVCacheTensorRole role,
                         GroupTopology* group,
                         std::string* error);

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

// One cache group of a request, as the scheduler addressed it.
struct CacheGroupRequest {
  int32_t group_id = 0;
  // The rows the scheduler addressed, in sequence order.
  std::vector<uint64_t> ids;
  // The *position* of each id in the sequence's block table, aligned one-to-one
  // with it. A request's ids are pool rows: a prefix-cache hit hands out rows
  // from wherever the shared prefix already sits, and a later chunk of a
  // chunked prefill starts mid-sequence, so nothing about an id says which
  // position it covers. Position is what the route places blocks by, so a
  // block-scoped group has to carry it; a sequence-scoped group's id already is
  // the position (its slot), so it may leave this empty.
  std::vector<uint64_t> positions;
};

// The canonical blocks one request covers, derived from the groups the
// scheduler addressed.
//
// A canonical block is the coordinate both peers agree on: the request's
// `p`-th block seen through sequence slice `j`, that is `p * split + j`, where
// `p` counts from the request's first block and `j` is the DCP rank that holds
// the slice. It is a *position*, not an address: the two instances allocate
// their own pool rows, so their block ids differ while the position does not,
// and only the position survives the crossing.
//
// A block-scoped group's id names one *logical* block, which spans
// `topology.kv_split_size` canonical blocks -- one per DCP rank, and the group
// supplies the *position* of each id so the canonical id can be that position
// through slice `j`. Positions are the identity the runtime itself uses:
// KVShardLayout::globalize() turns a rank's local row into the global slot of
// position `p` through slice `j`, and the indexer pool expands the request's
// `p`-th block into the rows `(p + 1) * dcp_size + j` for every `j`
// (expand_kv_shard_indexer_block_table) -- both count from the sequence's first
// block in a row space whose row 0 the block manager reserves for its padding
// block. Each family turns the canonical id into its own row with its own
// layout (see RouteBinder::bind).
//
// Deriving the position from the id instead is what the first version did, and
// it holds only while the ids happen to be `position + 1`: a prefix-cache hit
// or a mid-sequence chunk makes them arbitrary, and the route would then place
// the whole request at the wrong offset without any check firing. Positions are
// therefore required, and their absence is an error rather than a fallback.
//
// A sequence-scoped group has no block dimension: its slot id is already the
// canonical unit, so it needs no positions and neither the rebase nor the
// expansion applies to it.
//
// `local` are the declarations of this rank's families; they say which groups
// exist and whether each is sequence scoped. Returns false, filling `error`,
// for a group the model does not declare, whose families disagree on scope, or
// whose block-scoped ids carry no usable positions.
bool canonical_blocks_of_request(
    const std::vector<CacheGroupRequest>& groups,
    const std::vector<CacheTensorDeclaration>& local,
    std::vector<int64_t>* canonical_blocks,
    std::string* error);

// Translates the cache layout a peer published into the physical view the
// binder addresses, and reconciles it with what the model declares.
//
// This is the seam between the wire representation of a peer's cache and the
// peer-independent routing model: afterwards every routing decision is a pure
// function of checked numbers. Because the binder addresses a cache resource as
// `unit * local_heads * head_bytes + (head - first_local_head) * head_bytes`,
// a descriptor is accepted only when it provably describes that layout; the
// span arithmetic is what makes the acceptance a proof rather than a
// convention. A descriptor whose single span covers the whole cache resource
// carries no head axis at all: it is accepted only for a group with one local
// head, and which head that is comes from the rank that published it.
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
