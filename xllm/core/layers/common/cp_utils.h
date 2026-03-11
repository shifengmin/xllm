#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {

// Full CP prefill auxiliary tensors, aligned with MindIE
// sparse_attention.py::prepare_cp_prefill_inputs.
struct CPInputDict {
  torch::Tensor cp_load_balance_idx;
  torch::Tensor cp_o_recover_idx;
  torch::Tensor cp_kv_recover_idx;
  std::pair<torch::Tensor, torch::Tensor> k_gather_index;
  std::pair<torch::Tensor, torch::Tensor> actual_seq_lengths_key;
  std::pair<torch::Tensor, torch::Tensor> actual_seq_lengths_query;
};

// ATB deepseek-v3.2 decoder currently consumes these 5 tensors.
struct CPPrefillATBInputs {
  torch::Tensor seq_len_cp;
  torch::Tensor cp_load_balance_idx_first;
  torch::Tensor cp_load_balance_idx_last;
  torch::Tensor cp_o_recover_idx;
  torch::Tensor cp_kv_recover_idx;
};

torch::Tensor generate_cp_load_balance_idx(const torch::Tensor& input_lengths);

torch::Tensor generate_cp_o_recover_idx(const std::vector<int32_t>& chunk_lengths,
                                        const torch::Device& device);

torch::Tensor generate_cp_kv_recover_idx(int32_t cp_size,
                                         int64_t input_ids_size,
                                         const std::vector<int32_t>& chunk_lengths,
                                         const torch::Device& device);

std::pair<torch::Tensor, torch::Tensor> compute_input_lengths_cumsum_cp(
    const torch::Tensor& input_lengths_cumsum);

std::pair<torch::Tensor, torch::Tensor> generate_k_gather_index(
    const torch::Tensor& actual_seq_lengths_kv_cp_prev,
    const torch::Tensor& actual_seq_lengths_kv_cp_next,
    const torch::Tensor& input_lengths,
    int32_t cp_size);

CPInputDict prepare_cp_prefill_inputs(int32_t cp_size,
                                      const torch::Tensor& input_ids,
                                      const torch::Tensor& position_ids,
                                      const torch::Tensor& input_lengths_cumsum,
                                      const torch::Tensor& input_lengths);

// A minimal wrapper for ATB CP prefill inputs used by deepseek-v3.2 graph.
CPPrefillATBInputs prepare_cp_prefill_atb_inputs(int32_t cp_size,
                                                 const torch::Tensor& input_lengths);

}  // namespace layer
}  // namespace xllm
