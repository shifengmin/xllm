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

#include "core/framework/weight_cache/weight_mmap_cache.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xxHash/xxhash.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/framework/model/causal_lm.h"

namespace xllm {

namespace {

constexpr uint64_t kMagic = 0x584C4C4D57435631ULL;  // "XLLMWCV1"
constexpr uint32_t kVersion = 1;
constexpr size_t kDataAlignment = 64;
constexpr size_t kPageAlignment = 4096;

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

size_t tensor_byte_size(const torch::Tensor& tensor) {
  return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

}  // namespace

WeightMmapCache::WeightMmapCache(const std::string& cache_dir,
                                 const std::string& model_weights_path,
                                 int32_t rank,
                                 int32_t world_size,
                                 const std::string& model_type,
                                 const std::string& dtype,
                                 const std::string& quant_method,
                                 int64_t quant_bits,
                                 int64_t group_size)
    : cache_dir_(cache_dir),
      model_weights_path_(model_weights_path),
      rank_(rank),
      world_size_(world_size),
      model_type_(model_type),
      dtype_(dtype),
      quant_method_(quant_method),
      quant_bits_(quant_bits),
      group_size_(group_size) {}

WeightMmapCache::~WeightMmapCache() {
  if (mapped_addr_ != nullptr && mapped_addr_ != MAP_FAILED) {
    munmap(mapped_addr_, mapped_size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

std::string WeightMmapCache::compute_cache_path() const {
  std::string dir = cache_dir_;
  if (dir.empty()) {
    dir = model_weights_path_ + "/.xllm_cache";
  }

  XXH128_hash_t path_hash =
      XXH3_128bits(model_weights_path_.data(), model_weights_path_.size());
  char hash_prefix[17];
  snprintf(hash_prefix,
           sizeof(hash_prefix),
           "%016llx",
           static_cast<unsigned long long>(path_hash.low64));

  return dir + "/" + hash_prefix + "/rank" + std::to_string(rank_) + "_tp" +
         std::to_string(world_size_) + ".cache";
}

void WeightMmapCache::compute_validation_hash(uint8_t* out_hash) const {
  std::string fingerprint;
  fingerprint += model_weights_path_;
  fingerprint += "|model_type=" + model_type_;
  fingerprint += "|dtype=" + dtype_;
  fingerprint += "|world_size=" + std::to_string(world_size_);
  fingerprint += "|rank=" + std::to_string(rank_);
  fingerprint += "|quant_method=" + quant_method_;
  fingerprint += "|quant_bits=" + std::to_string(quant_bits_);
  fingerprint += "|group_size=" + std::to_string(group_size_);

  namespace fs = std::filesystem;
  std::error_code ec;
  if (fs::is_directory(model_weights_path_, ec)) {
    std::vector<std::string> entries;
    for (const auto& entry : fs::directory_iterator(model_weights_path_, ec)) {
      if (entry.path().extension() == ".safetensors" ||
          entry.path().extension() == ".bin") {
        auto mtime = fs::last_write_time(entry, ec);
        if (!ec) {
          entries.emplace_back(
              entry.path().filename().string() + "=" +
              std::to_string(mtime.time_since_epoch().count()));
        }
      }
    }
    std::sort(entries.begin(), entries.end());
    for (const auto& e : entries) {
      fingerprint += "|" + e;
    }
  }

  XXH128_hash_t hash = XXH3_128bits(fingerprint.data(), fingerprint.size());
  std::memcpy(out_hash, &hash, sizeof(hash));
}

bool WeightMmapCache::try_load(CausalLM* model, const torch::Device& device) {
  std::string cache_path = compute_cache_path();

  fd_ = open(cache_path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    return false;
  }

  struct stat sb;
  if (fstat(fd_, &sb) < 0) {
    close(fd_);
    fd_ = -1;
    return false;
  }
  mapped_size_ = static_cast<size_t>(sb.st_size);

  if (mapped_size_ < sizeof(CacheFileHeader)) {
    LOG(WARNING) << "Weight cache file too small, ignoring: " << cache_path;
    close(fd_);
    fd_ = -1;
    return false;
  }

  mapped_addr_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (mapped_addr_ == MAP_FAILED) {
    LOG(WARNING) << "Failed to mmap weight cache: " << cache_path;
    mapped_addr_ = nullptr;
    close(fd_);
    fd_ = -1;
    return false;
  }

  const auto* header = static_cast<const CacheFileHeader*>(mapped_addr_);
  if (header->magic != kMagic || header->version != kVersion) {
    LOG(INFO) << "Weight cache magic/version mismatch, rebuilding";
    munmap(mapped_addr_, mapped_size_);
    mapped_addr_ = nullptr;
    close(fd_);
    fd_ = -1;
    invalidate();
    return false;
  }

  uint8_t expected_hash[16];
  compute_validation_hash(expected_hash);
  if (std::memcmp(header->validation_hash, expected_hash, 16) != 0) {
    LOG(INFO) << "Weight cache validation hash mismatch, rebuilding";
    munmap(mapped_addr_, mapped_size_);
    mapped_addr_ = nullptr;
    close(fd_);
    fd_ = -1;
    invalidate();
    return false;
  }

  uint32_t tensor_count = header->tensor_count;
  uint64_t data_offset = header->data_offset;

  const auto* metas = reinterpret_cast<const TensorCacheMeta*>(
      static_cast<const char*>(mapped_addr_) + sizeof(CacheFileHeader));
  const char* data_base = static_cast<const char*>(mapped_addr_) + data_offset;

  auto params = model->named_parameters(/*recurse=*/true);
  std::unordered_map<std::string, torch::Tensor*> param_map;
  param_map.reserve(params.size());
  for (auto& pair : params) {
    param_map[pair.key()] = &pair.value();
  }

  for (uint32_t i = 0; i < tensor_count; ++i) {
    const auto& meta = metas[i];
    std::string name(meta.name);

    auto it = param_map.find(name);
    if (it == param_map.end()) {
      LOG(WARNING) << "Weight cache tensor not found in model: " << name;
      continue;
    }

    std::vector<int64_t> shape(meta.shape, meta.shape + meta.ndim);
    torch::ScalarType scalar_type = static_cast<torch::ScalarType>(meta.dtype);
    torch::Tensor cached_tensor =
        torch::from_blob(const_cast<char*>(data_base + meta.data_offset),
                         shape,
                         torch::TensorOptions().dtype(scalar_type));

    it->second->set_data(cached_tensor.to(device));
  }

  LOG(INFO) << "Loaded " << tensor_count
            << " preprocessed weight tensors from mmap cache";
  return true;
}

bool WeightMmapCache::save(const CausalLM* model) {
  auto params =
      const_cast<CausalLM*>(model)->named_parameters(/*recurse=*/true);
  uint32_t tensor_count = static_cast<uint32_t>(params.size());

  if (tensor_count == 0) {
    LOG(WARNING) << "No parameters to cache";
    return false;
  }

  size_t toc_size = sizeof(CacheFileHeader) +
                    static_cast<size_t>(tensor_count) * sizeof(TensorCacheMeta);
  size_t data_start = align_up(toc_size, kPageAlignment);

  size_t total_data_size = 0;
  std::vector<torch::Tensor> cpu_tensors;
  cpu_tensors.reserve(tensor_count);
  for (const auto& pair : params) {
    torch::Tensor t = pair.value().cpu().contiguous();
    total_data_size = align_up(total_data_size, kDataAlignment);
    total_data_size += tensor_byte_size(t);
    cpu_tensors.emplace_back(std::move(t));
  }

  size_t total_file_size = data_start + total_data_size;

  std::string cache_path = compute_cache_path();
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::create_directories(fs::path(cache_path).parent_path(), ec);
  if (ec) {
    LOG(WARNING) << "Failed to create cache directory: " << ec.message();
    return false;
  }

  std::string tmp_path = cache_path + ".tmp";
  int32_t save_fd = open(tmp_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (save_fd < 0) {
    LOG(WARNING) << "Failed to create weight cache file: " << tmp_path;
    return false;
  }

  if (ftruncate(save_fd, static_cast<off_t>(total_file_size)) != 0) {
    LOG(WARNING) << "Failed to allocate weight cache file (disk full?)";
    close(save_fd);
    unlink(tmp_path.c_str());
    return false;
  }

  void* write_addr = mmap(
      nullptr, total_file_size, PROT_READ | PROT_WRITE, MAP_SHARED, save_fd, 0);
  if (write_addr == MAP_FAILED) {
    LOG(WARNING) << "Failed to mmap weight cache for writing";
    close(save_fd);
    unlink(tmp_path.c_str());
    return false;
  }

  auto* header = static_cast<CacheFileHeader*>(write_addr);
  std::memset(header, 0, sizeof(CacheFileHeader));
  header->magic = kMagic;
  header->version = kVersion;
  header->tensor_count = tensor_count;
  header->data_offset = static_cast<uint64_t>(data_start);
  header->total_data_size = static_cast<uint64_t>(total_data_size);
  compute_validation_hash(header->validation_hash);

  auto* metas = reinterpret_cast<TensorCacheMeta*>(
      static_cast<char*>(write_addr) + sizeof(CacheFileHeader));
  char* data_base = static_cast<char*>(write_addr) + data_start;

  size_t current_offset = 0;
  uint32_t idx = 0;
  for (const auto& pair : params) {
    auto& meta = metas[idx];
    std::memset(&meta, 0, sizeof(TensorCacheMeta));

    const std::string& name = pair.key();
    CHECK(name.size() < sizeof(meta.name))
        << "Parameter name too long: " << name;
    std::strncpy(meta.name, name.c_str(), sizeof(meta.name) - 1);

    const torch::Tensor& t = cpu_tensors[idx];
    meta.ndim = static_cast<int32_t>(t.dim());
    for (int32_t d = 0; d < meta.ndim; ++d) {
      meta.shape[d] = t.size(d);
    }
    meta.dtype = static_cast<int32_t>(t.scalar_type());

    current_offset = align_up(current_offset, kDataAlignment);
    meta.data_offset = static_cast<uint64_t>(current_offset);
    meta.byte_size = static_cast<uint64_t>(tensor_byte_size(t));

    std::memcpy(data_base + current_offset, t.data_ptr(), meta.byte_size);
    current_offset += meta.byte_size;
    ++idx;
  }

  msync(write_addr, total_file_size, MS_SYNC);
  munmap(write_addr, total_file_size);
  close(save_fd);

  if (rename(tmp_path.c_str(), cache_path.c_str()) != 0) {
    LOG(WARNING) << "Failed to rename cache file from tmp";
    unlink(tmp_path.c_str());
    return false;
  }

  LOG(INFO) << "Saved " << tensor_count << " preprocessed weight tensors to "
            << cache_path << " (" << (total_file_size / (1024 * 1024))
            << " MB)";
  return true;
}

void WeightMmapCache::invalidate() {
  std::string cache_path = compute_cache_path();
  unlink(cache_path.c_str());
}

}  // namespace xllm
