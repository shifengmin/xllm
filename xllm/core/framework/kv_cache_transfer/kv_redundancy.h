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

namespace xllm {

// Instance-wide parallel topology of one PD peer. Deliberately free of any I/O
// and of the cache-layout manifests: every routing decision below is a pure
// function of these numbers.
struct KvTopology {
  int32_t dp_size = 1;
  int32_t cp_size = 1;
  int32_t tp_size = 1;
  // Configured kv-split (DCP) width. A group whose redundancy cannot absorb it
  // falls back to a split of 1; see KvRedundancy::split().
  int32_t kv_split_size = 1;
  // Tokens covered by one cache resource (== CacheTensorManifest::
  // block_token_capacity). Peer independent: it is the canonical block size.
  int32_t tokens_per_block = 0;
};

// Per-cache-group geometry. G is the number of *global* logical heads this
// group exposes on the model instance, which is what decides how many ranks
// hold replicated copies of it.
//
// Routing is per (cache namespace, role, group id), not per group id: several
// roles may share one group id (e.g. MLA latent KEY and the indexer both live
// in the KV group), and they must then agree on global_head_count, otherwise a
// single group-level route cannot serve both.
struct GroupTopology {
  int32_t global_head_count = 0;
  uint64_t head_bytes = 0;
  // Sequence-scoped groups (SSM / CONV / LINEAR / EMBEDDING slots) have no
  // block dimension: the sequence slice index is always 0 for them.
  bool sequence_scoped = false;
  // Block-scoped groups that nevertheless keep the whole sequence on every
  // rank. The DSA indexer pool is the example: its top-k selection reads
  // historical gate/valid values from the whole sequence, so the pool is
  // replicated per rank even though a redundancy budget exists. This is
  // declared rather than derived because the reason is semantic (what the
  // kernel must read), not a property of the redundancy.
  //
  // Only an instance that does not shard the sequence can honour it: with
  // context parallelism every rank computes just its own sequence shard and
  // writes only that shard into its pool. See group_keeps_whole_sequence().
  bool full_sequence_replica = false;
};

// Whether this group really keeps every canonical block on every rank of one
// instance.
//
// The declaration above says what the model needs the pool to hold; only an
// instance that computes every token on every rank can hold it. With context
// parallelism each rank computes, and therefore writes, just its own sequence
// shard: the indexer pool is scattered through the same KV-shard slot mapping
// the K/V cache uses, so a rank's pool holds the slice of the canonical blocks
// its DCP rank owns, in the K/V pool's compact row space, exactly like a split
// family. Routing such a pool as a replica hands the peer one writer's shard as
// if it were the whole sequence.
bool group_keeps_whole_sequence(const KvTopology& topology,
                                const GroupTopology& group);

// KV redundancy derived from one instance topology plus one group geometry.
//
// Terminology (all quantities are derived, none are configured):
//   Hl    local heads per rank
//   D_tp  TP-induced redundancy: how many adjacent tp ranks hold identical KV
//   Hc    number of head classes; Hc * Hl == G always holds
//   D     composite redundancy, CP * D_tp
//   S_eff effective split of this group
//   N_rep residual redundancy after the split, D / S_eff
class KvRedundancy final {
 public:
  // Default constructible so it can serve as an output parameter. The values
  // are only meaningful once derive() has returned true.
  KvRedundancy() = default;

  // Validates the divisibility constraints and derives every field.
  //
  // C1: global_head_count % tp_size == 0 or tp_size % global_head_count == 0
  // C2: 1 <= S_eff <= D, and S_eff divides D
  // C3: S_eff is a DCP shape the runtime can express: it divides cp_size, it
  //     equals cp_size * tp_size, or -- when cp_size is 1 -- it divides
  //     tp_size (see ContextParallelTopology). A split the runtime cannot place
  //     would otherwise abort when the DCP process group is built.
  //
  // S_eff equals the configured kv_split_size when this group's redundancy can
  // absorb it. It is 1 -- every rank keeps the whole sequence -- in exactly
  // three cases: the group is sequence scoped (no block dimension to split), it
  // is declared as a full-sequence replica and the instance honours it (see
  // group_keeps_whole_sequence), or its redundancy is 1 (nothing to remove).
  // Any other mismatch between the configured split and the group redundancy
  // fails instead of degrading silently. Returns false and fills `error` on
  // violation.
  static bool derive(const KvTopology& topology,
                     const GroupTopology& group,
                     KvRedundancy* redundancy,
                     std::string* error);

  int32_t local_head_count() const { return local_head_count_; }
  int32_t tp_redundancy() const { return tp_redundancy_; }
  int32_t head_class_count() const { return head_class_count_; }
  int32_t redundancy() const { return redundancy_; }
  int32_t split() const { return split_; }
  int32_t replica_count() const { return replica_count_; }
  bool sequence_scoped() const { return sequence_scoped_; }
  bool full_sequence_replica() const { return full_sequence_replica_; }

 private:
  int32_t local_head_count_ = 0;
  int32_t tp_redundancy_ = 0;
  int32_t head_class_count_ = 0;
  int32_t redundancy_ = 0;
  int32_t split_ = 0;
  int32_t replica_count_ = 0;
  bool sequence_scoped_ = false;
  bool full_sequence_replica_ = false;
};

// Maps a rank to the three orthogonal routing indices and back.
//
//   h  head class            -> which heads the rank holds
//   t  sequence slice        -> which canonical blocks the rank holds
//   c  replica index         -> which redundant copy of them
//
// A rank holds (all heads of class h) x (all canonical blocks with slice t).
// c != 0 ranks are byte-identical replicas of the c == 0 rank in the same
// group and therefore never act as transfer writers.
//
// t is the rank's DCP rank, because that is the identity the runtime uses:
// KVShardLayout::globalize() maps its local row to the canonical block
// `row * split + dcp_rank`, and the decode path builds that layout from the
// DCP process group. ContextParallelTopology describes the shapes the runtime
// supports, and `derive` rejects every other split:
//
//   (a) split <= cp_size and cp_size % split == 0: DCP partitions the PCP
//       group, so t = cp_rank / (cp_size / split);
//   (b) split == cp_size * tp_size: DCP covers the whole DP-local domain, so
//       t = cp_rank * tp_size + tp_rank (every rank its own slice);
//   (c) cp_size == 1 and split divides tp_size: DCP is one of the consecutive
//       blocks of `split` TP ranks the domain is cut into, so t = tp_rank %
//       split and the `tp_size / split` groups repeat the same slices.
//
// Slices are what the two peers have to agree on: canonical block b lives on
// the rank whose t is `b % split`, on either side.
class KvLayoutIndex final {
 public:
  KvLayoutIndex(const KvTopology& topology, const KvRedundancy& redundancy);

  // rank = dp * (cp_size * tp_size) + cp * tp_size + tp
  int32_t rank(int32_t dp_rank, int32_t cp_rank, int32_t tp_rank) const;

  // Global head range owned by head class `head_class`.
  int32_t head_begin(int32_t head_class) const;
  int32_t head_end(int32_t head_class) const;

  int32_t head_class_of(int32_t tp_rank) const;
  // Sequence slice and redundant-copy index of one rank.
  int32_t slice_of(int32_t cp_rank, int32_t tp_rank) const;
  int32_t replica_of(int32_t cp_rank, int32_t tp_rank) const;

  // The single writer for (head class, slice): the replica-0 rank. Returns
  // false when the pair is not owned by any rank of this instance.
  bool writer_of(int32_t dp_rank,
                 int32_t head_class,
                 int32_t slice,
                 int32_t* rank) const;

  // Every rank holding (head class, slice), one per replica. Ascending by
  // replica index.
  bool replicas_of(int32_t dp_rank,
                   int32_t head_class,
                   int32_t slice,
                   std::vector<int32_t>* ranks) const;

  int32_t cp_size() const { return cp_size_; }
  int32_t tp_size() const { return tp_size_; }
  int32_t head_class_count() const { return head_class_count_; }
  int32_t split() const { return split_; }
  int32_t local_head_count() const { return local_head_count_; }

 private:
  int32_t dp_size_ = 0;
  int32_t cp_size_ = 0;
  int32_t tp_size_ = 0;
  int32_t tp_redundancy_ = 0;
  int32_t head_class_count_ = 0;
  int32_t local_head_count_ = 0;
  int32_t split_ = 0;
  int32_t replica_count_ = 0;
  // DCP shape (a) splits the PCP group into cp_size / split sequence-replica
  // groups; shape (b) has a single group covering the whole DP-local domain and
  // shape (c) has tp_size / split groups of `split` consecutive TP ranks.
  bool partitions_pcp_ = false;
  bool tiles_tp_ = false;
  int32_t sequence_groups_ = 1;
};

// Peer-independent block identity.
//
// The canonical block is the unit both peers agree on: it covers
// `tokens_per_block` tokens and is defined by a token range only, so it does
// not depend on either side's kv-split width or physical resource geometry.
// The local physical row index is derived from it, never the other way round.
class CanonicalBlock final {
 public:
  CanonicalBlock(int32_t tokens_per_block, int32_t split);

  int64_t block_of_token(int64_t token_index) const;
  int64_t token_begin(int64_t block) const;
  int64_t token_end(int64_t block) const;

  // The *position* row holding `block` on a rank whose slice is `slice`,
  // counted from the sequence's first block. A cache pool row can be one
  // further along: the xLLM block manager reserves row 0 for its padding block,
  // so the pool row of the same block is `local_row(block) + 1` for a split
  // family. That offset belongs to the peer-dependent side of the route, so it
  // is applied where a canonical block becomes a peer row (RouteBinder), not
  // here. Only meaningful for blocks this rank owns; see owns().
  int64_t local_row(int64_t block) const;
  // Inverse of local_row().
  int64_t canonical_of_row(int64_t row, int32_t slice) const;
  // True when `block` is assigned to the rank whose sequence slice is `slice`.
  bool owns(int64_t block, int32_t slice) const;

  int32_t tokens_per_block() const { return tokens_per_block_; }
  int32_t split() const { return split_; }

 private:
  int32_t tokens_per_block_ = 0;
  int32_t split_ = 1;
};

}  // namespace xllm
