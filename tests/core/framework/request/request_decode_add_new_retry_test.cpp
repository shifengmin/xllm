/* Copyright 2025-2026 The xLLM Authors.
All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "framework/request/request.h"
#include "framework/request/request_state.h"

namespace xllm {
namespace {

std::shared_ptr<Request> make_request() {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     /*prompt_tokens=*/std::vector<int32_t>{1, 2, 3},
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     /*seq_capacity=*/16,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>("req-1", "x-req-1", "0", state, "svc-1");
}

TEST(RequestDecodeAddNewRetryTest, ImmediateByDefault) {
  auto request = make_request();
  EXPECT_EQ(request->decode_add_new_retries(), 0);
  EXPECT_TRUE(request->can_decode_add_new_retry(absl::Now()));
}

TEST(RequestDecodeAddNewRetryTest, IntervalGatesRetry) {
  auto request = make_request();
  request->bump_decode_add_new_retry(/*interval_ms=*/1000);
  EXPECT_EQ(request->decode_add_new_retries(), 1);
  EXPECT_FALSE(request->can_decode_add_new_retry(absl::Now()));
  EXPECT_TRUE(request->can_decode_add_new_retry(
      request->decode_add_new_next_retry_time()));
}

TEST(RequestDecodeAddNewRetryTest, ZeroIntervalAllowsImmediateRetry) {
  auto request = make_request();
  request->bump_decode_add_new_retry(/*interval_ms=*/0);
  EXPECT_EQ(request->decode_add_new_retries(), 1);
  EXPECT_TRUE(request->can_decode_add_new_retry(absl::Now()));
}

TEST(RequestDecodeAddNewRetryTest, RetriesAccumulateUntilMax) {
  auto request = make_request();
  constexpr int32_t kMaxRetries = 3;
  for (int32_t i = 0; i < kMaxRetries; ++i) {
    request->bump_decode_add_new_retry(/*interval_ms=*/0);
    EXPECT_EQ(request->decode_add_new_retries(), i + 1);
    EXPECT_LT(request->decode_add_new_retries(), kMaxRetries + 1);
  }
  EXPECT_GE(request->decode_add_new_retries(), kMaxRetries);
}

}  // namespace
}  // namespace xllm
