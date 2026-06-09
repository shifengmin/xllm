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

#include <glog/logging.h>
#include <torch/torch.h>

#include <initializer_list>
#include <sstream>
#include <string>

#include "util/env_var.h"

namespace xllm {
namespace util {

struct NpuScatterTraceTensor {
  const char* name;
  torch::Tensor tensor;
};

inline bool npu_scatter_trace_enabled() {
  static const bool enabled = get_bool_env("XLLM_NPU_SCATTER_TRACE", false);
  return enabled;
}

void npu_scatter_trace_sync();

inline std::string npu_scatter_trace_describe_tensor(
    const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return "undefined";
  }
  std::ostringstream oss;
  oss << tensor.scalar_type() << "@" << tensor.device();
  oss << " shape=" << tensor.sizes();
  return oss.str();
}

inline void log_npu_scatter_before(
    const char* site,
    const char* op,
    std::initializer_list<NpuScatterTraceTensor> tensors) {
  if (!npu_scatter_trace_enabled()) {
    return;
  }
  npu_scatter_trace_sync();
  std::ostringstream oss;
  oss << "[NPU_SCATTER_TRACE] BEFORE " << site << " op=" << op;
  for (const auto& item : tensors) {
    oss << " " << item.name << "="
        << npu_scatter_trace_describe_tensor(item.tensor);
  }
  LOG(INFO) << oss.str();
}

inline void log_npu_scatter_after(const char* site, const char* op) {
  if (!npu_scatter_trace_enabled()) {
    return;
  }
  npu_scatter_trace_sync();
  LOG(INFO) << "[NPU_SCATTER_TRACE] AFTER " << site << " op=" << op;
}

}  // namespace util
}  // namespace xllm
