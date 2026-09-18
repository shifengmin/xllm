/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <string>

#include "core/common/macros.h"
#include "core/framework/config/option_category.h"

namespace xllm {

class JsonReader;
class KVCacheConfig;
class SchedulerConfig;

class DisaggPDConfig final {
 public:
  DisaggPDConfig() = default;
  ~DisaggPDConfig() = default;

  static DisaggPDConfig& get_instance();

  void from_flags();
  void from_json(const JsonReader& json);
  void append_config_json(nlohmann::ordered_json& config_json) const;
  void initialize();
  void normalize_mlu(KVCacheConfig& kv_cache_config,
                     SchedulerConfig& scheduler_config);
  void normalize_dcu(SchedulerConfig& scheduler_config);

  [[nodiscard]] static const OptionCategory& option_category() {
    static const OptionCategory kOptionCategory = {
        "DISAGGREGATED PREFILL-DECODE OPTIONS",
        {"enable_disagg_pd",
         "enable_pd_ooc",
         "disagg_pd_port",
         "instance_role",
         "kv_cache_transfer_mode",
         "transfer_listen_port",
         "pd_route"}};
    return kOptionCategory;
  }

  PROPERTY(bool, enable_disagg_pd) = false;

  PROPERTY(bool, enable_pd_ooc) = false;

  PROPERTY(int32_t, disagg_pd_port) = 7777;

  PROPERTY(std::string, instance_role) = "DEFAULT";

  PROPERTY(std::string, kv_cache_transfer_mode) = "PUSH";

  PROPERTY(int32_t, transfer_listen_port) = 26000;

  // Which data plane moves the KV cache: `legacy` (the rank-aligned strided
  // remap) or `canonical` (the route tables derived from both peer layouts).
  // The value is parsed where it is consumed, so a typo fails before any
  // transfer starts instead of silently selecting one of the two.
  PROPERTY(std::string, pd_route) = "legacy";

  PROPERTY(bool, kv_push_dst_rotate) = false;
};

}  // namespace xllm
