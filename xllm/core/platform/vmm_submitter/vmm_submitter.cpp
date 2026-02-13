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

#include "vmm_submitter.h"

#include <glog/logging.h>

#include <chrono>
#include <thread>

#include "vmm_manager.h"
#include "vmm_worker.h"

namespace xllm {
namespace vmm {

VMMSubmitter::VMMSubmitter(int32_t device_id, std::shared_ptr<VMMWorker> worker)
    : device_id_(device_id), worker_(std::move(worker)) {}

VMMSubmitter::~VMMSubmitter() {}

uint64_t VMMSubmitter::map(VirPtr va, PhyMemHandle phy) {

  uint64_t request_id = next_request_id_++;
  VMMRequest req(OpType::MAP, va, phy, 0, request_id, this);

  if (!worker_->submit_request(req)) {
    LOG(ERROR) << "Failed to submit map request";
    return 0;
  }

  pending_map_++;
  return request_id;
}

uint64_t VMMSubmitter::unmap(VirPtr va, size_t aligned_size) {

  uint64_t request_id = next_request_id_++;
  VMMRequest req(OpType::UNMAP, va, 0, aligned_size, request_id, this);

  if (!worker_->submit_request(req)) {
    LOG(ERROR) << "Failed to submit unmap request";
    return 0;
  }

  pending_unmap_++;
  return request_id;
}

void VMMSubmitter::release_vaddr(VirPtr va, size_t aligned_size) {

  uint64_t request_id = next_request_id_++;
  VMMRequest req(OpType::RELEASE_VADDR, va, 0, aligned_size, request_id, this);
  if (!worker_->submit_request(req)) {
    LOG(ERROR) << "Failed to submit release_vaddr request";
  }
}

size_t VMMSubmitter::poll_completions(size_t max_completions) {
  size_t count = 0;
  VMMCompletion completion;

  while (count < max_completions && completion_queue_.try_dequeue(completion)) {
    if (completion.op_type == OpType::MAP) {
      if (pending_map_ > 0) pending_map_--;
    } else {
      if (pending_unmap_ > 0) pending_unmap_--;
    }
    if (!completion.success) {
      LOG(ERROR) << "Operation failed: request_id=" << completion.request_id
                 << ", type="
                 << (completion.op_type == OpType::MAP ? "MAP" : "UNMAP");
    }
    count++;
  }

  return count;
}

bool VMMSubmitter::all_map_done() const { return pending_map_ == 0; }

bool VMMSubmitter::all_unmap_done() const { return pending_unmap_ == 0; }

void VMMSubmitter::wait_all() {
  while (!all_map_done() || !all_unmap_done()) {
    poll_completions(32);
    std::this_thread::yield();
  }
}

bool VMMSubmitter::push_completion(const VMMCompletion& completion) {
  completion_queue_.enqueue(completion);
  return true;
}

}  // namespace vmm
}  // namespace xllm
