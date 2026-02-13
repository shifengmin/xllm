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

#include "vmm_manager.h"

#include <glog/logging.h>

#include "vmm_submitter.h"
#include "vmm_worker.h"

namespace xllm {
namespace vmm {

VMMManager::~VMMManager() { shutdown(); }

void VMMManager::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (shutdown_flag_.exchange(true)) {
    return;  // Already shutting down or done
  }

  LOG(INFO) << "Shutting down VMMManager...";

  for (auto& [device_id, worker] : workers_) {
    worker->stop();
  }

  workers_.clear();
  LOG(INFO) << "VMMManager shutdown complete";
}

VMMSubmitter* VMMManager::get_submitter(int32_t device_id) {
  thread_local std::unordered_map<int32_t, std::unique_ptr<VMMSubmitter>> maps;
  auto it = maps.find(device_id);
  if (it != maps.end()) {
    return it->second.get();
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    auto sub = vmm::VMMManager::get_instance().create_submitter(device_id);
    if (!sub) {
      return nullptr;
    }
    VMMSubmitter* p = sub.get();
    maps[device_id] = std::move(sub);
    return p;
  }
}

std::unique_ptr<VMMSubmitter> VMMManager::create_submitter(int32_t device_id) {
  std::shared_ptr<VMMWorker> worker = get_worker(device_id);
  if (worker == nullptr) {
    worker = create_worker(device_id);
    worker->start();
    workers_[device_id] = worker;
  }
  return std::unique_ptr<VMMSubmitter>(new VMMSubmitter(device_id, worker));
}

std::shared_ptr<VMMWorker> VMMManager::get_worker(int32_t device_id) {
  auto it = workers_.find(device_id);
  if (it != workers_.end()) {
    return it->second;  
  }
  return nullptr;
}

std::shared_ptr<VMMWorker> VMMManager::create_worker(int32_t device_id) {
  // Use private constructor to create worker
#ifdef XLLM_VMM_TEST
  worker_create_count_.fetch_add(1, std::memory_order_relaxed);
#endif
  return std::shared_ptr<VMMWorker>(new VMMWorker(device_id));
}

#ifdef XLLM_VMM_TEST
size_t VMMManager::test_worker_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return workers_.size();
}
#endif

}  // namespace vmm
}  // namespace xllm
