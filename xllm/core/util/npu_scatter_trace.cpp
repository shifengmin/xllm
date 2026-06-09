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

#include "util/npu_scatter_trace.h"

#if defined(USE_NPU)
#include <acl/acl.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

namespace xllm {
namespace util {

void npu_scatter_trace_sync() {
#if defined(USE_NPU)
  if (!npu_scatter_trace_enabled()) {
    return;
  }
  const auto stream = c10_npu::getCurrentNPUStream();
  const aclError ret = aclrtSynchronizeStream(stream.stream());
  if (ret != ACL_SUCCESS) {
    LOG(WARNING) << "[NPU_SCATTER_TRACE] aclrtSynchronizeStream failed, ret="
                 << ret;
  }
#endif
}

}  // namespace util
}  // namespace xllm
