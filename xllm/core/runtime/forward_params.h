/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "framework/model/model_input_params.h"
#include "framework/request/mm_batch_data.h"
#include "framework/request/mm_data.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/sampling_params.h"
#include "platform/device.h"

namespace xllm {

class WorkerType {
 public:
  enum Value : int8_t {
    INVALID = 0,
    LLM,     // LLM
    VLM,     // VLM
    DIT,     // DIT
    ELM,     // Embedding LM
    EVLM,    // Embedding VLM
    REC,     // Rec
    MMEVLM,  // Encoder Embedding VLM
  };

  constexpr WorkerType(Value v) : value_(v) {}
  WorkerType(const std::string& str) {
    if (str == "LLM") {
      value_ = LLM;
    } else if (str == "VLM") {
      value_ = VLM;
    } else if (str == "DIT") {
      value_ = DIT;
    } else if (str == "ELM") {
      value_ = ELM;
    } else if (str == "EVLM") {
      value_ = EVLM;
    } else if (str == "REC") {
      value_ = REC;
    } else if (str == "MMEVLM") {
      value_ = MMEVLM;
    } else {
      value_ = INVALID;
    }
  }

  WorkerType() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(WorkerType rhs) const { return value_ == rhs.value_; }
  bool operator!=(WorkerType rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == LLM) {
      return "LLM";
    } else if (this->value_ == VLM) {
      return "VLM";
    } else if (this->value_ == DIT) {
      return "DIT";
    } else if (this->value_ == ELM) {
      return "ELM";
    } else if (this->value_ == EVLM) {
      return "EVLM";
    } else if (this->value_ == REC) {
      return "REC";
    } else if (this->value_ == MMEVLM) {
      return "MMEVLM";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

// Step-level decode metadata for Rec multi-round (device loop).
struct StepDecodeMeta {
  int32_t batch_size = 0;
  int32_t beam_width = 1;
  int32_t current_round = 0;
  int32_t total_round = 0;
  // Planned decode kv cache shape: [batch_size * beam_width, n_kv_heads,
  // step_rounds, head_dim]
  std::vector<int64_t> full_kv_shape;
  // Flattened decode positions for each sequence.
  std::vector<int32_t> decode_positions_vec;
};

// Inputs for forward execution
struct ForwardInput {
  ForwardInput to(const torch::Device& device, torch::ScalarType dtype) const {
    ForwardInput inputs;
    inputs.token_ids = safe_to(token_ids, device, true);
    inputs.positions = safe_to(positions, device, true);
    // Convert positions to int64 on CUDA/ILU/MUSA to avoid repeated per-layer
    // type conversions in rope kernels.
    const auto dev = Device::type_str();
    if ((dev == "cuda" || dev == "ilu" || dev == "musa") &&
        inputs.positions.defined() &&
        inputs.positions.scalar_type() != torch::kInt64) {
      inputs.positions = inputs.positions.to(torch::kInt64);
    }
    inputs.input_params = input_params.to(device);
    inputs.sampling_params = sampling_params.to(device, dtype);
    inputs.decoder_sampling_params = decoder_sampling_params.to(device, dtype);
    inputs.transfer_kv_infos = transfer_kv_infos;
    inputs.eplb_info = eplb_info;
    inputs.acc_logprob = safe_to(acc_logprob, device, true);
    inputs.step_decode = step_decode;
    inputs.skip_sampling_for_logits_only = skip_sampling_for_logits_only;
    inputs.device_input_buffer = device_input_buffer;
    return inputs;
  }

  ForwardInput cp_partition(int32_t cp_rank, int32_t cp_size) {
    ForwardInput outputs = *this;
    if (cp_size <= 1 || !token_ids.defined() || token_ids.numel() == 0 ||
        !input_params.batch_forward_type.is_prefill()) {
      return outputs;
    }

    CHECK_GT(cp_size, 0);
    CHECK_GE(cp_rank, 0);
    CHECK_LT(cp_rank, cp_size);

    const int32_t num_sequences = input_params.num_sequences;
    CHECK_GT(num_sequences, 0);
    const int32_t num_chunks = cp_size * 2;

    std::vector<int32_t> cp_q_lens;
    cp_q_lens.reserve(num_sequences);
    std::vector<int64_t> gather_indices;
    gather_indices.reserve(token_ids.numel());
    int32_t cp_global_max_seq_len = 0;

    // Keep old/new per-sequence offsets for selected_token_idxes remapping.
    std::vector<int64_t> old_seq_offsets;
    old_seq_offsets.reserve(num_sequences + 1);
    old_seq_offsets.push_back(0);
    std::vector<int64_t> new_seq_offsets;
    new_seq_offsets.reserve(num_sequences + 1);
    new_seq_offsets.push_back(0);

    for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      const int32_t input_len = std::max(0, input_params.get_q_seq_len(seq_idx));
      const int64_t seq_start = old_seq_offsets.back();
      const int64_t chunk_len =
          (input_len + num_chunks - 1) / static_cast<int64_t>(num_chunks);

      auto range_len = [&](int64_t local_start, int64_t local_end) -> int64_t {
        local_start = std::max<int64_t>(0, local_start);
        local_end = std::max<int64_t>(0, local_end);
        local_start = std::min<int64_t>(local_start, input_len);
        local_end = std::min<int64_t>(local_end, input_len);
        return std::max<int64_t>(0, local_end - local_start);
      };

      int64_t local_len = 0;
      auto append_range = [&](int64_t local_start, int64_t local_end) {
        const int64_t valid_len = range_len(local_start, local_end);
        if (valid_len <= 0) {
          return;
        }
        const int64_t start = std::max<int64_t>(0, std::min<int64_t>(local_start, input_len));
        for (int64_t i = 0; i < valid_len; ++i) {
          gather_indices.push_back(seq_start + start + i);
        }
        local_len += valid_len;
      };

      append_range(chunk_len * cp_rank, chunk_len * (cp_rank + 1));
      append_range(chunk_len * (num_chunks - 1 - cp_rank),
                   chunk_len * (num_chunks - cp_rank));

      cp_q_lens.push_back(static_cast<int32_t>(local_len));
      old_seq_offsets.push_back(seq_start + input_len);
      new_seq_offsets.push_back(new_seq_offsets.back() + local_len);

      // Match MindIE semantics: max_seq_len uses cp_tokens.max() across CP ranks.
      int64_t seq_cp_max = 0;
      for (int32_t rank = 0; rank < cp_size; ++rank) {
        const int64_t former_len =
            range_len(chunk_len * rank, chunk_len * (rank + 1));
        const int64_t latter_len = range_len(
            chunk_len * (num_chunks - 1 - rank), chunk_len * (num_chunks - rank));
        seq_cp_max = std::max(seq_cp_max, former_len + latter_len);
      }
      cp_global_max_seq_len =
          std::max(cp_global_max_seq_len, static_cast<int32_t>(seq_cp_max));
    }
    CHECK_EQ(old_seq_offsets.back(), token_ids.numel());

    const auto gather_idx_options =
        torch::TensorOptions().device(token_ids.device()).dtype(torch::kLong);
    const torch::Tensor gather_idx =
        torch::tensor(gather_indices, gather_idx_options);

    outputs.token_ids = token_ids.index_select(/*dim=*/0, gather_idx);

    if (positions.defined()) {
      if (positions.dim() == 1) {
        outputs.positions = positions.index_select(/*dim=*/0, gather_idx);
      } else if (positions.dim() == 2) {
        // mRoPE positions are [3, num_tokens].
        outputs.positions = positions.index_select(/*dim=*/1, gather_idx);
      } else {
        CHECK(false) << "Unsupported positions dim for cp_partition: "
                     << positions.dim();
      }
    }

    auto& out_params = outputs.input_params;
    const auto build_seq_lens =
        [&](const std::vector<int>& original,
            const std::vector<int32_t>& lengths) -> std::vector<int> {
      const bool is_cumsum = original.size() == (num_sequences + 1) &&
                             !original.empty() && original.front() == 0;
      std::vector<int> result;
      if (is_cumsum) {
        result.reserve(num_sequences + 1);
        result.push_back(0);
        for (const int32_t len : lengths) {
          result.push_back(result.back() + len);
        }
      } else {
        result.assign(lengths.begin(), lengths.end());
      }
      return result;
    };

    out_params.q_seq_lens_vec =
        build_seq_lens(input_params.q_seq_lens_vec, cp_q_lens);
    out_params.kv_seq_lens_vec =
        build_seq_lens(input_params.kv_seq_lens_vec, cp_q_lens);

    auto q_seq_lens_options =
        out_params.q_seq_lens.defined()
            ? out_params.q_seq_lens.options()
            : torch::TensorOptions().device(token_ids.device()).dtype(torch::kInt);
    auto kv_seq_lens_options =
        out_params.kv_seq_lens.defined()
            ? out_params.kv_seq_lens.options()
            : torch::TensorOptions().device(token_ids.device()).dtype(torch::kInt);
    out_params.q_seq_lens = torch::tensor(out_params.q_seq_lens_vec,
                                          q_seq_lens_options);
    out_params.kv_seq_lens = torch::tensor(out_params.kv_seq_lens_vec,
                                           kv_seq_lens_options);

    std::vector<int32_t> q_cu_seq_lens(cp_q_lens.size());
    std::partial_sum(cp_q_lens.begin(), cp_q_lens.end(), q_cu_seq_lens.begin());
    auto q_cu_seq_lens_options =
        out_params.q_cu_seq_lens.defined()
            ? out_params.q_cu_seq_lens.options()
            : torch::TensorOptions().device(token_ids.device()).dtype(torch::kInt);
    out_params.q_cu_seq_lens =
        torch::tensor(q_cu_seq_lens, q_cu_seq_lens_options);

    out_params.q_max_seq_len = cp_global_max_seq_len;
    out_params.kv_max_seq_len = cp_global_max_seq_len;

    auto partition_token_level_tensor = [&](torch::Tensor& tensor) {
      if (tensor.defined() && tensor.dim() >= 1 &&
          tensor.size(0) == token_ids.size(0)) {
        tensor = tensor.index_select(/*dim=*/0, gather_idx);
      }
    };
    partition_token_level_tensor(out_params.new_cache_slots);
    partition_token_level_tensor(out_params.new_cache_slot_offsets);
    partition_token_level_tensor(out_params.input_embedding);

    std::unordered_map<int64_t, int64_t> old_to_new_idx;
    old_to_new_idx.reserve(gather_indices.size());
    for (int64_t new_idx = 0; new_idx < static_cast<int64_t>(gather_indices.size());
         ++new_idx) {
      old_to_new_idx[gather_indices[new_idx]] = new_idx;
    }

    auto remap_selected_token_idxes = [&](SamplingParameters& params) {
      if (!params.selected_token_idxes.defined()) {
        return;
      }
      if (gather_indices.empty()) {
        params.selected_token_idxes =
            torch::empty({0}, params.selected_token_idxes.options());
        if (params.sample_idxes.defined()) {
          params.sample_idxes = torch::empty({0}, params.sample_idxes.options());
        }
        return;
      }

      const auto selected_cpu = safe_to(params.selected_token_idxes,
                                        torch::kCPU,
                                        true)
                                    .to(torch::kLong)
                                    .contiguous();
      const int64_t* selected_ptr = selected_cpu.data_ptr<int64_t>();
      const int64_t selected_num = selected_cpu.numel();
      std::vector<int64_t> remapped_idxes;
      remapped_idxes.reserve(selected_num);

      for (int64_t i = 0; i < selected_num; ++i) {
        const int64_t old_idx = selected_ptr[i];
        const auto it = old_to_new_idx.find(old_idx);
        if (it != old_to_new_idx.end()) {
          remapped_idxes.push_back(it->second);
          continue;
        }

        auto upper =
            std::upper_bound(old_seq_offsets.begin(), old_seq_offsets.end(), old_idx);
        int64_t seq_idx = static_cast<int64_t>(upper - old_seq_offsets.begin()) - 1;
        seq_idx = std::max<int64_t>(
            0, std::min<int64_t>(seq_idx, static_cast<int64_t>(num_sequences) - 1));

        const int64_t new_start = new_seq_offsets[seq_idx];
        const int64_t new_end = new_seq_offsets[seq_idx + 1];
        remapped_idxes.push_back(new_end > new_start ? new_end - 1 : 0);
      }

      params.selected_token_idxes =
          torch::tensor(remapped_idxes, params.selected_token_idxes.options());
    };

    remap_selected_token_idxes(outputs.sampling_params);
    remap_selected_token_idxes(outputs.decoder_sampling_params);
    return outputs;
  }

  void print() const {
    LOG(INFO) << "  token_ids: " << token_ids << std::endl;
    LOG(INFO) << "  positions: " << positions << std::endl;
    input_params.print();
    LOG(INFO) << " params.selected_token_idxes "
              << sampling_params.selected_token_idxes;
    LOG(INFO) << " params.sample_idxes " << sampling_params.sample_idxes;
    LOG(INFO) << " params.do_sample " << sampling_params.do_sample;
  }

  const StepDecodeMeta* step_meta() const {
    return step_decode ? &(*step_decode) : nullptr;
  }

  bool has_step_meta() const { return step_decode.has_value(); }

  // flatten token ids
  torch::Tensor token_ids;
  // flatten positions
  torch::Tensor positions;
  ModelInputParams input_params;
  SamplingParameters sampling_params;
  SamplingParameters decoder_sampling_params;
  // beam search kernel input
  torch::Tensor acc_logprob;

  // step-level decode metadata
  std::optional<StepDecodeMeta> step_decode;
  // If true, skip sampler forward and only keep logits.
  bool skip_sampling_for_logits_only = false;

  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;
  EplbInfo eplb_info;

  // A tensor used to store all device-side input data, with other input tensors
  // constructed based on the address and offset of this tensor.
  torch::Tensor device_input_buffer;
};

// output after forward execution
struct ForwardOutput {
  // sample parameters for speculative decoding
  torch::Tensor do_sample;
  // whether to return logprobs
  bool logprobs = false;
  // max number of top logprobs in the batch
  int64_t max_top_logprobs = 0;
  SampleOutput sample_output;
  torch::Tensor logits;
  torch::Tensor embedding;

  // for eplb, collect the tokens load of experts on each worker.
  torch::Tensor expert_load_data;
  // for eplb, indicates that the specified layer on the worker
  // has completed the asynchronous loading of new weight.
  int32_t prepared_layer_id;

  BeamSearchOutput beam_search_output;
  torch::Tensor beam_sequence_group;
};

// Model input with raw data, which will be
// serielize to pb type before pass to remote worker.
struct RawForwardInput {
  std::vector<int32_t> flatten_tokens_vec;
  std::vector<int32_t> flatten_positions_vec;
  std::vector<std::vector<int32_t>> m_positions_vec;
  std::vector<const RequestSamplingParam*> sampling_params;
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  std::vector<std::vector<int64_t>> unique_token_ids_vec;
  std::vector<std::vector<int32_t>> unique_token_counts_vec;
  std::vector<int32_t> unique_token_lens_vec;
  BatchForwardType batch_forward_type;
  uint32_t max_seq_len;
  uint32_t q_max_seq_len;
  std::vector<int32_t> seq_lens;
  std::vector<int32_t> q_seq_lens;
  std::vector<int32_t> q_cu_seq_lens;
  std::vector<int32_t> kv_cache_tokens_nums;
  std::vector<int32_t> new_token_slot_ids;
  std::vector<std::vector<int32_t>> block_tables_vec;
  int32_t num_sequences;
  // num tokens of all workers，mainly used for dp case
  std::vector<int32_t> dp_global_token_nums;
  std::vector<int32_t> dp_is_decode;
  // kv info for disaggregated prefill/decode
  std::vector<TransferKVInfo> transfer_kv_infos;
  EplbInfo eplb_info;
  std::vector<std::vector<float>> embeddings;
  // chunked prefill case of speculative decoding
  // extra token ids for each sequence, and -1 for last chunk
  std::vector<int32_t> extra_token_ids;
  // embedding ids of each sequence
  std::vector<int> embedding_ids;
  // request ids of each sequence
  std::vector<std::string> request_ids;
  // swap
  std::vector<BlockTransferInfo> swap_blocks;
  uint64_t batch_id;
  // block copy kernel
  std::vector<int32_t> src_block_indices;
  std::vector<int32_t> dst_block_indices;
  std::vector<int32_t> cum_sum;
  // for continuous kvcache
  std::vector<int64_t> new_cache_slot_offsets;  //[n_tokens]
  std::vector<int64_t> kv_cache_start_offsets;  //[n_seq]
  // beam search kernel input
  std::vector<float> acc_logprob_vec;
  // for flashinfer
  std::vector<int32_t> paged_kv_indptr;         //[n_seq + 1]
  std::vector<int32_t> paged_kv_indices;        //[num_used_pages]
  std::vector<int32_t> paged_kv_last_page_len;  //[n_seq]
  // multimodal data
  MMBatchData mm_data;
};

struct RawSampleOutput {
  std::vector<RawToken> tokens;  // num tokens
};

struct RawForwardOutput {
  std::vector<RawSampleOutput> outputs;  // num seqs
  std::vector<int64_t> expert_load_data;
  int32_t prepared_layer_id;
  // beam search kernel output
  std::vector<int32_t> src_seq_idxes;
  std::vector<int32_t> out_tokens;
  std::vector<float> out_logprobs;

  // batch-level beam output for Rec multi-round mode
  std::vector<int32_t> beam_sequence_group;  // flattened 2D
  // multimodal embedding output
  std::vector<torch::Tensor> mm_embeddings;
};

struct BatchedForwardInputs {
  std::vector<ForwardInput> micro_inputs;
  SamplingParameters concated_sampling_params;
  // beam search kernel input
  torch::Tensor acc_logprob;
};

}  // namespace xllm
