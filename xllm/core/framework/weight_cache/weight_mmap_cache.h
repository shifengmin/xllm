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

#include <torch/torch.h>

#include <cstdint>
#include <string>

namespace xllm {

class CausalLM;

struct CacheFileHeader {
  uint64_t magic;
  uint32_t version;
  uint32_t tensor_count;
  uint64_t data_offset;
  uint64_t total_data_size;
  uint8_t validation_hash[16];
  uint8_t reserved[76];
};

struct TensorCacheMeta {
  char name[256];
  int32_t ndim;
  int64_t shape[8];
  int32_t dtype;
  uint64_t data_offset;
  uint64_t byte_size;
};

class WeightMmapCache final {
 public:
  WeightMmapCache(const std::string& cache_dir,
                  const std::string& model_weights_path,
                  int32_t rank,
                  int32_t world_size,
                  const std::string& model_type,
                  const std::string& dtype,
                  const std::string& quant_method,
                  int64_t quant_bits,
                  int64_t group_size);

  ~WeightMmapCache();

  WeightMmapCache(const WeightMmapCache&) = delete;
  WeightMmapCache& operator=(const WeightMmapCache&) = delete;

  bool try_load(CausalLM* model, const torch::Device& device);

  bool save(const CausalLM* model);

  void invalidate();

 private:
  std::string compute_cache_path() const;
  void compute_validation_hash(uint8_t* out_hash) const;

  std::string cache_dir_;
  std::string model_weights_path_;
  int32_t rank_;
  int32_t world_size_;
  std::string model_type_;
  std::string dtype_;
  std::string quant_method_;
  int64_t quant_bits_;
  int64_t group_size_;

  void* mapped_addr_ = nullptr;
  size_t mapped_size_ = 0;
  int32_t fd_ = -1;
};

}  // namespace xllm
