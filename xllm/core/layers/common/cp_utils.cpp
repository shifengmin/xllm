#include "cp_utils.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace xllm {
namespace layer {
namespace {

constexpr int64_t kInt32Max = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
constexpr int64_t kInt32Min = static_cast<int64_t>(std::numeric_limits<int32_t>::min());

int32_t CheckedToInt32(int64_t value, const char* name) {
  TORCH_CHECK(value >= kInt32Min && value <= kInt32Max,
              name,
              " out of int32 range: ",
              value);
  return static_cast<int32_t>(value);
}

std::vector<int32_t> ToCPUInt32Vector(const torch::Tensor& tensor,
                                      const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D tensor");
  const auto dtype = tensor.scalar_type();
  TORCH_CHECK(dtype == torch::kInt32 || dtype == torch::kInt64 ||
                  dtype == torch::kLong,
              name,
              " must be int32/int64 tensor");

  auto cpu = tensor.to(torch::kCPU).to(torch::kInt32).contiguous();
  const auto* ptr = cpu.data_ptr<int32_t>();
  return std::vector<int32_t>(ptr, ptr + cpu.numel());
}

torch::Tensor MakeInt32Tensor(const std::vector<int32_t>& values,
                              const torch::Device& device) {
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  if (values.empty()) {
    return torch::empty({0}, options);
  }
  return torch::tensor(values, options);
}

}  // namespace

torch::Tensor generate_cp_load_balance_idx(const torch::Tensor& input_lengths) {
  std::vector<int32_t> lengths = ToCPUInt32Vector(input_lengths, "input_lengths");

  std::vector<int32_t> cp_load_balance_idx_first;
  std::vector<int32_t> cp_load_balance_idx_last;
  cp_load_balance_idx_first.reserve(input_lengths.numel());
  cp_load_balance_idx_last.reserve(input_lengths.numel());

  int64_t base = 0;
  for (int32_t length : lengths) {
    TORCH_CHECK(length >= 0, "input_lengths contains negative value: ", length);
    const int32_t divider = length / 2;
    for (int32_t i = 0; i < divider; ++i) {
      cp_load_balance_idx_first.push_back(CheckedToInt32(base + i,
                                                         "cp_load_balance_idx_first"));
    }
    for (int32_t i = divider; i < length; ++i) {
      cp_load_balance_idx_last.push_back(CheckedToInt32(base + i,
                                                        "cp_load_balance_idx_last"));
    }
    base += length;
  }

  cp_load_balance_idx_first.insert(cp_load_balance_idx_first.end(),
                                   cp_load_balance_idx_last.begin(),
                                   cp_load_balance_idx_last.end());
  return MakeInt32Tensor(cp_load_balance_idx_first, input_lengths.device());
}

torch::Tensor generate_cp_o_recover_idx(const std::vector<int32_t>& chunk_lengths,
                                        const torch::Device& device) {
  std::vector<int32_t> cp_o_recover_idx;

  int64_t base = 0;
  int64_t chunk_lengths_sum =
      std::accumulate(chunk_lengths.begin(), chunk_lengths.end(), int64_t{0});

  for (int32_t chunk_len : chunk_lengths) {
    TORCH_CHECK(chunk_len >= 0, "chunk_lengths contains negative value: ", chunk_len);
    for (int32_t idx = 0; idx < chunk_len; ++idx) {
      cp_o_recover_idx.push_back(CheckedToInt32(base + idx, "cp_o_recover_idx"));
    }
    for (int32_t idx = 0; idx < chunk_len; ++idx) {
      cp_o_recover_idx.push_back(
          CheckedToInt32(base + idx + chunk_lengths_sum, "cp_o_recover_idx"));
    }
    base += chunk_len;
  }

  return MakeInt32Tensor(cp_o_recover_idx, device);
}

torch::Tensor generate_cp_kv_recover_idx(int32_t cp_size,
                                         int64_t input_ids_size,
                                         const std::vector<int32_t>& chunk_lengths,
                                         const torch::Device& device) {
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");
  TORCH_CHECK(input_ids_size >= 0, "input_ids_size must be non-negative");

  std::vector<int32_t> cp_kv_recover_idx;
  int64_t req_offset = 0;

  for (int32_t req_chunk_len : chunk_lengths) {
    TORCH_CHECK(req_chunk_len >= 0,
                "chunk_lengths contains negative value: ",
                req_chunk_len);
    std::vector<std::vector<int32_t>> gather_idx_per_chunk(cp_size * 2);
    for (int32_t cp_rank_id = 0; cp_rank_id < cp_size; ++cp_rank_id) {
      const int64_t rank_offset = static_cast<int64_t>(cp_rank_id) * input_ids_size;
      for (int32_t idx = 0; idx < req_chunk_len; ++idx) {
        gather_idx_per_chunk[cp_rank_id].push_back(
            CheckedToInt32(rank_offset + req_offset + idx, "cp_kv_recover_idx"));
      }
      for (int32_t idx = req_chunk_len; idx < req_chunk_len * 2; ++idx) {
        gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id].push_back(
            CheckedToInt32(rank_offset + req_offset + idx, "cp_kv_recover_idx"));
      }
    }

    for (const auto& chunk : gather_idx_per_chunk) {
      cp_kv_recover_idx.insert(cp_kv_recover_idx.end(), chunk.begin(), chunk.end());
    }
    req_offset += static_cast<int64_t>(req_chunk_len) * 2;
  }

  return MakeInt32Tensor(cp_kv_recover_idx, device);
}

std::pair<torch::Tensor, torch::Tensor> compute_input_lengths_cumsum_cp(
    const torch::Tensor& input_lengths_cumsum) {
  std::vector<int32_t> cumsum =
      ToCPUInt32Vector(input_lengths_cumsum, "input_lengths_cumsum");

  std::vector<int32_t> prev(cumsum.size(), 0);
  std::vector<int32_t> next(cumsum.size(), 0);

  int32_t offset = 0;
  for (size_t i = 0; i < cumsum.size(); ++i) {
    TORCH_CHECK(cumsum[i] >= offset,
                "input_lengths_cumsum must be non-decreasing");
    prev[i] = offset + (cumsum[i] - offset) / 2;
    next[i] = cumsum[i];
    offset = cumsum[i];
  }

  return {MakeInt32Tensor(prev, input_lengths_cumsum.device()),
          MakeInt32Tensor(next, input_lengths_cumsum.device())};
}

std::pair<torch::Tensor, torch::Tensor> generate_k_gather_index(
    const torch::Tensor& actual_seq_lengths_kv_cp_prev,
    const torch::Tensor& actual_seq_lengths_kv_cp_next,
    const torch::Tensor& input_lengths,
    int32_t cp_size) {
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");

  std::vector<int32_t> prev_lens =
      ToCPUInt32Vector(actual_seq_lengths_kv_cp_prev, "actual_seq_lengths_kv_cp_prev");
  std::vector<int32_t> next_lens =
      ToCPUInt32Vector(actual_seq_lengths_kv_cp_next, "actual_seq_lengths_kv_cp_next");
  std::vector<int32_t> input_lens = ToCPUInt32Vector(input_lengths, "input_lengths");

  TORCH_CHECK(prev_lens.size() == next_lens.size() &&
                  prev_lens.size() == input_lens.size(),
              "length mismatch among kv lens and input lengths");

  std::vector<int32_t> k_gather_index_prev;
  std::vector<int32_t> k_gather_index_next;

  int64_t k_offset = 0;
  for (size_t i = 0; i < input_lens.size(); ++i) {
    TORCH_CHECK(prev_lens[i] >= 0 && next_lens[i] >= 0 && input_lens[i] >= 0,
                "negative sequence length is not allowed");

    for (int32_t idx = 0; idx < prev_lens[i]; ++idx) {
      k_gather_index_prev.push_back(
          CheckedToInt32(k_offset + idx, "k_gather_index_prev"));
    }
    for (int32_t idx = 0; idx < next_lens[i]; ++idx) {
      k_gather_index_next.push_back(
          CheckedToInt32(k_offset + idx, "k_gather_index_next"));
    }

    k_offset += static_cast<int64_t>(input_lens[i]) * cp_size;
  }

  return {MakeInt32Tensor(k_gather_index_prev, input_lengths.device()),
          MakeInt32Tensor(k_gather_index_next, input_lengths.device())};
}

CPInputDict prepare_cp_prefill_inputs(int32_t cp_size,
                                      const torch::Tensor& input_ids,
                                      const torch::Tensor& position_ids,
                                      const torch::Tensor& input_lengths_cumsum,
                                      const torch::Tensor& input_lengths) {
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");
  TORCH_CHECK(position_ids.defined(), "position_ids must be defined");
  TORCH_CHECK(position_ids.dim() == 1,
              "position_ids must be 1D in prepare_cp_prefill_inputs");

  CPInputDict cp_input_dict;

  std::vector<int32_t> lengths = ToCPUInt32Vector(input_lengths, "input_lengths");
  std::vector<int32_t> cumsum =
      ToCPUInt32Vector(input_lengths_cumsum, "input_lengths_cumsum");
  TORCH_CHECK(lengths.size() == cumsum.size(),
              "input_lengths and input_lengths_cumsum size mismatch");

  int64_t sum_lengths = std::accumulate(lengths.begin(), lengths.end(), int64_t{0});
  TORCH_CHECK(sum_lengths == input_ids.numel(),
              "sum(input_lengths) must equal input_ids.numel(), got ",
              sum_lengths,
              " vs ",
              input_ids.numel());

  std::vector<int32_t> chunk_lengths;
  chunk_lengths.reserve(lengths.size());
  for (int32_t len : lengths) {
    TORCH_CHECK(len >= 0, "input_lengths contains negative value: ", len);
    chunk_lengths.push_back(len / 2);
  }

  cp_input_dict.cp_load_balance_idx = generate_cp_load_balance_idx(input_lengths);
  cp_input_dict.cp_o_recover_idx =
      generate_cp_o_recover_idx(chunk_lengths, input_lengths.device());
  cp_input_dict.cp_kv_recover_idx = generate_cp_kv_recover_idx(
      cp_size, input_ids.numel(), chunk_lengths, input_lengths.device());

  auto [input_lengths_cumsum_cp_prev, input_lengths_cumsum_cp_next] =
      compute_input_lengths_cumsum_cp(input_lengths_cumsum);

  auto pos_cpu = position_ids.to(torch::kCPU).to(torch::kInt32).contiguous();
  const int32_t* pos_ptr = pos_cpu.data_ptr<int32_t>();
  const int64_t pos_numel = pos_cpu.numel();

  auto prev_idx_cpu =
      input_lengths_cumsum_cp_prev.to(torch::kCPU).to(torch::kInt32).contiguous();
  auto next_idx_cpu =
      input_lengths_cumsum_cp_next.to(torch::kCPU).to(torch::kInt32).contiguous();

  std::vector<int32_t> actual_prev;
  std::vector<int32_t> actual_next;
  actual_prev.reserve(lengths.size());
  actual_next.reserve(lengths.size());

  const int32_t* prev_ptr = prev_idx_cpu.data_ptr<int32_t>();
  const int32_t* next_ptr = next_idx_cpu.data_ptr<int32_t>();

  for (size_t i = 0; i < lengths.size(); ++i) {
    TORCH_CHECK(prev_ptr[i] > 0 && next_ptr[i] > 0,
                "input_lengths_cumsum values must be > 0");
    const int64_t prev_index = static_cast<int64_t>(prev_ptr[i]) - 1;
    const int64_t next_index = static_cast<int64_t>(next_ptr[i]) - 1;
    TORCH_CHECK(prev_index >= 0 && prev_index < pos_numel,
                "prev position index out of range: ",
                prev_index,
                ", pos_numel: ",
                pos_numel);
    TORCH_CHECK(next_index >= 0 && next_index < pos_numel,
                "next position index out of range: ",
                next_index,
                ", pos_numel: ",
                pos_numel);
    actual_prev.push_back(pos_ptr[prev_index] + 1);
    actual_next.push_back(pos_ptr[next_index] + 1);
  }

  auto actual_seq_lengths_kv_cp_prev =
      MakeInt32Tensor(actual_prev, input_lengths.device());
  auto actual_seq_lengths_kv_cp_next =
      MakeInt32Tensor(actual_next, input_lengths.device());

  cp_input_dict.k_gather_index = generate_k_gather_index(
      actual_seq_lengths_kv_cp_prev,
      actual_seq_lengths_kv_cp_next,
      input_lengths,
      cp_size);

  cp_input_dict.actual_seq_lengths_key = {
      torch::cumsum(actual_seq_lengths_kv_cp_prev, 0, torch::kInt32),
      torch::cumsum(actual_seq_lengths_kv_cp_next, 0, torch::kInt32)};

  auto input_lengths_cumsum_half =
      torch::floor_divide(input_lengths_cumsum.to(torch::kInt32), 2).contiguous();
  cp_input_dict.actual_seq_lengths_query = {input_lengths_cumsum_half,
                                            input_lengths_cumsum_half.clone()};

  return cp_input_dict;
}

CPPrefillATBInputs prepare_cp_prefill_atb_inputs(int32_t cp_size,
                                                 const torch::Tensor& input_lengths) {
  TORCH_CHECK(cp_size > 0, "cp_size must be positive");

  std::vector<int32_t> lengths = ToCPUInt32Vector(input_lengths, "input_lengths");
  std::vector<int32_t> chunk_lengths;
  chunk_lengths.reserve(lengths.size());

  int64_t local_token_num = 0;
  for (int32_t len : lengths) {
    TORCH_CHECK(len >= 0, "input_lengths contains negative value: ", len);
    chunk_lengths.push_back(len / 2);
    local_token_num += len;
  }

  CPPrefillATBInputs outputs;
  outputs.seq_len_cp = MakeInt32Tensor(chunk_lengths, input_lengths.device());

  auto cp_load_balance_idx = generate_cp_load_balance_idx(input_lengths);
  const int64_t first_numel =
      std::accumulate(chunk_lengths.begin(), chunk_lengths.end(), int64_t{0});
  TORCH_CHECK(first_numel >= 0 && first_numel <= cp_load_balance_idx.numel(),
              "invalid first_numel for cp_load_balance_idx split");
  outputs.cp_load_balance_idx_first =
      cp_load_balance_idx.narrow(/*dim=*/0, /*start=*/0, /*length=*/first_numel)
          .contiguous();
  outputs.cp_load_balance_idx_last =
      cp_load_balance_idx
          .narrow(/*dim=*/0,
                  /*start=*/first_numel,
                  /*length=*/cp_load_balance_idx.numel() - first_numel)
          .contiguous();

  outputs.cp_o_recover_idx =
      generate_cp_o_recover_idx(chunk_lengths, input_lengths.device());
  outputs.cp_kv_recover_idx = generate_cp_kv_recover_idx(
      cp_size, local_token_num, chunk_lengths, input_lengths.device());

  return outputs;
}

}  // namespace layer
}  // namespace xllm
