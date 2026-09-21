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

#include "framework/kv_cache_transfer/kv_redundancy.h"

#include <algorithm>

namespace xllm {

namespace {

void set_error(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
}

// The runtime places the sequence split with the DCP process group, and
// ContextParallelTopology accepts exactly two shapes for it (see
// ContextParallelTopology::ContextParallelTopology):
//   (a) the DCP group partitions the PCP group: split divides cp_size;
//   (b) the DCP group covers the whole DP-local domain: split is cp_size *
//       tp_size.
bool dcp_partitions_pcp(const KvTopology& topology, int32_t split) {
  return split <= topology.cp_size && topology.cp_size % split == 0;
}

bool dcp_spans_domain(const KvTopology& topology, int32_t split) {
  return split == topology.cp_size * topology.tp_size;
}

// DCP shape (c): without PCP the DCP group can still be narrower than the TP
// axis. The runtime cuts the DP-local domain into `tp_size / split` consecutive
// DCP groups of `split` ranks each (the NPU DCP process group is indexed as
// `global_rank / dcp_size`), so a rank's slice is its TP rank modulo the split
// and the groups after the first repeat the same slices.
bool dcp_tiles_tp(const KvTopology& topology, int32_t split) {
  return topology.cp_size == 1 && split > 1 && split < topology.tp_size &&
         topology.tp_size % split == 0;
}

}  // namespace

bool group_keeps_whole_sequence(const KvTopology& topology,
                                const GroupTopology& group) {
  // CP is what shards the sequence across ranks: every rank then computes only
  // its own shard of it and writes just that shard into the pool. A DCP split
  // without CP does not do that -- every rank still sees every token -- so the
  // pool really is a replica there.
  return group.full_sequence_replica && topology.cp_size <= 1;
}

bool KvRedundancy::derive(const KvTopology& topology,
                          const GroupTopology& group,
                          KvRedundancy* redundancy,
                          std::string* error) {
  if (redundancy == nullptr) {
    set_error(error, "redundancy output must not be null");
    return false;
  }
  if (topology.cp_size <= 0 || topology.tp_size <= 0 || topology.dp_size <= 0) {
    set_error(error, "parallel sizes must be positive");
    return false;
  }
  if (group.global_head_count <= 0) {
    set_error(error, "global_head_count must be positive");
    return false;
  }

  // C1: the global head count must be evenly divisible by the TP width
  // (sharding) or divide it (replication).
  const int32_t tp_size = topology.tp_size;
  const int32_t global_heads = group.global_head_count;
  if (global_heads % tp_size != 0 && tp_size % global_heads != 0) {
    set_error(error,
              "global_head_count (" + std::to_string(global_heads) +
                  ") must be divisible by tp_size (" + std::to_string(tp_size) +
                  ") or divide it: neither sharding nor replication is "
                  "possible");
    return false;
  }

  // The declaration says what the model needs the pool to hold; the instance
  // decides whether it can hold it.
  const bool whole_sequence = group_keeps_whole_sequence(topology, group);

  KvRedundancy derived;
  derived.sequence_scoped_ = group.sequence_scoped;
  derived.full_sequence_replica_ = whole_sequence;
  // G >= TP shards the heads; G < TP replicates them.
  derived.local_head_count_ = std::max(global_heads / tp_size, 1);
  derived.tp_redundancy_ = std::max(tp_size / global_heads, 1);
  derived.head_class_count_ = tp_size / derived.tp_redundancy_;
  derived.redundancy_ = topology.cp_size * derived.tp_redundancy_;

  // C2: the split may never exceed this group's redundancy, and it must divide
  // it so that the redundant group decomposes into whole complete groups.
  //
  // Three cases legitimately keep the whole sequence on every rank:
  //   - sequence-scoped groups (SSM / CONV / linear state slots) have no block
  //     dimension to split at all;
  //   - groups whose instance really does keep it (the DSA indexer pool without
  //     CP; with CP the same pool is written per shard and splits like the K/V
  //     cache -- see group_keeps_whole_sequence);
  //   - D == 1 means the group has no redundancy to remove.
  // Any other mismatch is a configuration error: silently degrading to 1 would
  // leave the operator believing a wider split is active.
  const int32_t configured_split = std::max(topology.kv_split_size, 1);
  if (group.sequence_scoped || whole_sequence || derived.redundancy_ == 1) {
    derived.split_ = 1;
  } else if (configured_split <= derived.redundancy_ &&
             derived.redundancy_ % configured_split == 0) {
    derived.split_ = configured_split;
  } else {
    set_error(error,
              "kv_split_size (" + std::to_string(configured_split) +
                  ") must divide the group redundancy (" +
                  std::to_string(derived.redundancy_) +
                  ") and not exceed it; use 1 to disable the split for this "
                  "group");
    return false;
  }
  derived.replica_count_ = derived.redundancy_ / derived.split_;

  // C3: the runtime has to be able to place this split in its DCP topology.
  if (!dcp_partitions_pcp(topology, derived.split_) &&
      !dcp_spans_domain(topology, derived.split_) &&
      !dcp_tiles_tp(topology, derived.split_)) {
    set_error(error,
              "kv_split_size (" + std::to_string(derived.split_) +
                  ") is not a DCP shape the runtime supports: it must divide "
                  "cp_size (" +
                  std::to_string(topology.cp_size) +
                  "), equal cp_size * tp_size (" +
                  std::to_string(topology.cp_size * topology.tp_size) +
                  "), or divide tp_size (" + std::to_string(topology.tp_size) +
                  ") when cp_size is 1");
    return false;
  }

  // Post-condition: head classes tile the global head range exactly.
  if (derived.head_class_count_ * derived.local_head_count_ != global_heads) {
    set_error(error,
              "head classes do not tile the global head range: head classes (" +
                  std::to_string(derived.head_class_count_) +
                  ") * local heads (" +
                  std::to_string(derived.local_head_count_) +
                  ") != " + std::to_string(global_heads));
    return false;
  }

  *redundancy = derived;
  if (error != nullptr) {
    error->clear();
  }
  return true;
}

KvLayoutIndex::KvLayoutIndex(const KvTopology& topology,
                             const KvRedundancy& redundancy) {
  dp_size_ = topology.dp_size;
  cp_size_ = topology.cp_size;
  tp_size_ = topology.tp_size;
  tp_redundancy_ = redundancy.tp_redundancy();
  head_class_count_ = redundancy.head_class_count();
  local_head_count_ = redundancy.local_head_count();
  split_ = redundancy.split();
  replica_count_ = redundancy.replica_count();
  partitions_pcp_ = dcp_partitions_pcp(topology, split_);
  tiles_tp_ = dcp_tiles_tp(topology, split_);
  // How many ranks hold identical copies of one slice: the PCP group divided by
  // the split for shape (a), the TP axis divided by the split for shape (c),
  // and one for shape (b), where every rank owns its own slice.
  if (partitions_pcp_) {
    sequence_groups_ = cp_size_ / split_;
  } else {
    sequence_groups_ = tiles_tp_ ? tp_size_ / split_ : 1;
  }
}

int32_t KvLayoutIndex::rank(int32_t dp_rank,
                            int32_t cp_rank,
                            int32_t tp_rank) const {
  return dp_rank * (cp_size_ * tp_size_) + cp_rank * tp_size_ + tp_rank;
}

int32_t KvLayoutIndex::head_begin(int32_t head_class) const {
  return head_class * local_head_count_;
}

int32_t KvLayoutIndex::head_end(int32_t head_class) const {
  return (head_class + 1) * local_head_count_;
}

int32_t KvLayoutIndex::head_class_of(int32_t tp_rank) const {
  return tp_rank / tp_redundancy_;
}

int32_t KvLayoutIndex::slice_of(int32_t cp_rank, int32_t tp_rank) const {
  // The slice of a rank is its DCP rank, the identity KVShardLayout::globalize
  // inverts.
  if (partitions_pcp_) {
    return cp_rank / sequence_groups_;
  }
  if (tiles_tp_) {
    // Shape (c): the DCP group is the consecutive block of `split` TP ranks, so
    // the slice is the rank's position inside that block.
    return tp_rank % split_;
  }
  return cp_rank * tp_size_ + tp_rank;
}

int32_t KvLayoutIndex::replica_of(int32_t cp_rank, int32_t tp_rank) const {
  if (tiles_tp_) {
    // Which of the `tp_size / split` groups that repeat this slice the rank
    // belongs to.
    return tp_rank / split_;
  }
  if (!partitions_pcp_) {
    // Every rank holds its own slice of the single head class, so there is no
    // redundant copy and every rank is a writer.
    return 0;
  }
  const int32_t sequence_replica = cp_rank % sequence_groups_;
  const int32_t tp_replica = tp_rank % tp_redundancy_;
  return sequence_replica * tp_redundancy_ + tp_replica;
}

bool KvLayoutIndex::writer_of(int32_t dp_rank,
                              int32_t head_class,
                              int32_t slice,
                              int32_t* rank_out) const {
  if (rank_out == nullptr || dp_rank < 0 || dp_rank >= dp_size_ ||
      head_class < 0 || head_class >= head_class_count_ || slice < 0 ||
      slice >= split_) {
    return false;
  }
  // Replica 0 is the first sequence-replica group and the first TP replica.
  const int32_t tp_rank = head_class * tp_redundancy_;
  if (partitions_pcp_) {
    *rank_out = rank(dp_rank, slice * sequence_groups_, tp_rank);
    return true;
  }
  if (tiles_tp_) {
    // The first DCP group is the replica-0 one and a rank's position inside its
    // group is its slice, so the writer of a slice sits at that TP rank.
    if (slice / tp_redundancy_ != head_class) {
      return false;
    }
    *rank_out = rank(dp_rank, /*cp_rank=*/0, slice);
    return true;
  }
  // The whole DP-local domain is one DCP group, so the rank whose DCP rank is
  // the slice is its only holder.
  const int32_t cp_rank = slice / tp_size_;
  const int32_t holder_tp = slice % tp_size_;
  if (holder_tp / tp_redundancy_ != head_class) {
    return false;
  }
  *rank_out = rank(dp_rank, cp_rank, holder_tp);
  return true;
}

bool KvLayoutIndex::replicas_of(int32_t dp_rank,
                                int32_t head_class,
                                int32_t slice,
                                std::vector<int32_t>* ranks) const {
  if (ranks == nullptr || dp_rank < 0 || dp_rank >= dp_size_ ||
      head_class < 0 || head_class >= head_class_count_ || slice < 0 ||
      slice >= split_) {
    return false;
  }
  ranks->clear();
  if (tiles_tp_) {
    // One rank per repeating group, ascending by replica index.
    ranks->reserve(static_cast<size_t>(sequence_groups_));
    for (int32_t tp_rank = slice; tp_rank < tp_size_; tp_rank += split_) {
      if (head_class_of(tp_rank) != head_class) {
        continue;
      }
      ranks->emplace_back(rank(dp_rank, /*cp_rank=*/0, tp_rank));
    }
    return !ranks->empty();
  }
  if (!partitions_pcp_) {
    const int32_t cp_rank = slice / tp_size_;
    const int32_t holder_tp = slice % tp_size_;
    if (holder_tp / tp_redundancy_ != head_class) {
      return false;
    }
    ranks->emplace_back(rank(dp_rank, cp_rank, holder_tp));
    return true;
  }
  ranks->reserve(static_cast<size_t>(replica_count_));
  for (int32_t sequence_replica = 0; sequence_replica < sequence_groups_;
       ++sequence_replica) {
    const int32_t cp_rank = slice * sequence_groups_ + sequence_replica;
    for (int32_t tp_replica = 0; tp_replica < tp_redundancy_; ++tp_replica) {
      ranks->emplace_back(
          rank(dp_rank, cp_rank, head_class * tp_redundancy_ + tp_replica));
    }
  }
  return true;
}

CanonicalBlock::CanonicalBlock(int32_t tokens_per_block, int32_t split)
    : tokens_per_block_(tokens_per_block), split_(std::max(split, 1)) {}

int64_t CanonicalBlock::block_of_token(int64_t token_index) const {
  return token_index / tokens_per_block_;
}

int64_t CanonicalBlock::token_begin(int64_t block) const {
  return block * tokens_per_block_;
}

int64_t CanonicalBlock::token_end(int64_t block) const {
  return (block + 1) * tokens_per_block_;
}

int64_t CanonicalBlock::local_row(int64_t block) const {
  return block / split_;
}

int64_t CanonicalBlock::canonical_of_row(int64_t row, int32_t slice) const {
  return row * split_ + slice;
}

bool CanonicalBlock::owns(int64_t block, int32_t slice) const {
  return block % split_ == slice;
}

}  // namespace xllm
