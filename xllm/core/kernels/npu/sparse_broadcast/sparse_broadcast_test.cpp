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

#include "sparse_broadcast.h"
#include "sparse_broadcast_gold.h"
#include "sparse_broadcast_layout.h"

#include <acl/acl.h>
#include <shmem.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using xllm::kernel::npu::kSparseBcastBlockDim;
using xllm::kernel::npu::kSparseBcastElemSize;
using xllm::kernel::npu::launch_sparse_broadcast;
using xllm::kernel::npu::sparse_bcast_dst_bytes;
using xllm::kernel::npu::sparse_bcast_gold_scatter;
using xllm::kernel::npu::sparse_bcast_packed_bytes;
using xllm::kernel::npu::sparse_bcast_row_bytes;
using xllm::kernel::npu::SparseBroadcastLaunchArgs;

#define CHECK_ACL(expr)                                                                 \
  do {                                                                                  \
    aclError acl_status = (expr);                                                       \
    if (acl_status != ACL_ERROR_NONE) {                                                 \
      std::fprintf(stderr, "%s:%d aclError=%d\n", __FILE__, __LINE__, acl_status);      \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

void fill_src(std::vector<uint8_t>* src, int32_t n_rows, int32_t d) {
  uint32_t row_bytes = sparse_bcast_row_bytes(static_cast<uint32_t>(d), kSparseBcastElemSize);
  src->assign(static_cast<size_t>(n_rows) * row_bytes, 0);
  for (int32_t r = 0; r < n_rows; ++r) {
    for (uint32_t b = 0; b < row_bytes; ++b) {
      (*src)[static_cast<uint32_t>(r) * row_bytes + b] =
          static_cast<uint8_t>((r * 31 + static_cast<int32_t>(b)) & 0xff);
    }
  }
}

int32_t compare_bytes(const uint8_t* got, const uint8_t* want, size_t n, const char* tag, int32_t pe) {
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr,
                   "pe=%d %s mismatch at %zu got=%u want=%u\n",
                   pe,
                   tag,
                   i,
                   static_cast<unsigned>(got[i]),
                   static_cast<unsigned>(want[i]));
      return 1;
    }
  }
  return 0;
}

int32_t run_case(int32_t pe,
                 aclrtStream stream,
                 uint64_t ffts_addr,
                 int32_t n_rows,
                 int32_t k,
                 int32_t d,
                 const std::vector<int32_t>& index_host,
                 int32_t seq,
                 const char* name) {
  uint32_t row_bytes = sparse_bcast_row_bytes(static_cast<uint32_t>(d), kSparseBcastElemSize);
  if (row_bytes % 32u != 0) {
    std::fprintf(stderr, "row_bytes must be 32B aligned, got %u\n", row_bytes);
    return 1;
  }

  size_t src_bytes = static_cast<size_t>(n_rows) * row_bytes;
  size_t dst_bytes = sparse_bcast_dst_bytes(static_cast<uint32_t>(n_rows),
                                            static_cast<uint32_t>(d),
                                            kSparseBcastElemSize);
  size_t idx_bytes = static_cast<size_t>(k) * sizeof(int32_t);
  size_t packed_bytes = sparse_bcast_packed_bytes(static_cast<uint32_t>(k),
                                                  static_cast<uint32_t>(d),
                                                  kSparseBcastElemSize);

  std::vector<uint8_t> src_host;
  fill_src(&src_host, n_rows, d);
  std::vector<uint8_t> gold(dst_bytes, 0x3c);
  sparse_bcast_gold_scatter(src_host.data(),
                            index_host.data(),
                            gold.data(),
                            n_rows,
                            k,
                            d,
                            static_cast<int32_t>(kSparseBcastElemSize));

  void* src_dev = nullptr;
  void* index_dev = nullptr;
  void* index_out_dev = nullptr;
  void* dst_dev = nullptr;
  CHECK_ACL(aclrtMalloc(&src_dev, src_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&index_dev, idx_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&index_out_dev, idx_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMalloc(&dst_dev, dst_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK_ACL(aclrtMemset(index_out_dev, idx_bytes, 0x3c, idx_bytes));
  CHECK_ACL(aclrtMemset(dst_dev, dst_bytes, 0x3c, dst_bytes));

  if (pe == 0) {
    CHECK_ACL(aclrtMemcpy(src_dev, src_bytes, src_host.data(), src_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(index_dev, idx_bytes, index_host.data(), idx_bytes, ACL_MEMCPY_HOST_TO_DEVICE));
  } else {
    CHECK_ACL(aclrtMemset(src_dev, src_bytes, 0xa5, src_bytes));
    CHECK_ACL(aclrtMemset(index_dev, idx_bytes, 0xa5, idx_bytes));
  }

  void* packed = aclshmem_malloc(packed_bytes);
  void* flag = aclshmem_malloc(64);
  if (packed == nullptr || flag == nullptr) {
    std::fprintf(stderr, "aclshmem_malloc failed\n");
    return 1;
  }
  CHECK_ACL(aclrtMemset(flag, 64, 0, 64));

  SparseBroadcastLaunchArgs args{};
  args.stream = stream;
  args.ffts_addr = ffts_addr;
  args.src = static_cast<uint8_t*>(src_dev);
  args.index = static_cast<int32_t*>(index_dev);
  args.index_out = static_cast<int32_t*>(index_out_dev);
  args.dst = static_cast<uint8_t*>(dst_dev);
  args.packed = static_cast<uint8_t*>(packed);
  args.flag = static_cast<int32_t*>(flag);
  args.n_rows = n_rows;
  args.k = k;
  args.d = d;
  args.elem_size = static_cast<int32_t>(kSparseBcastElemSize);
  args.root = 0;
  args.seq = seq;
  args.block_dim = kSparseBcastBlockDim;
  launch_sparse_broadcast(args);
  CHECK_ACL(aclrtSynchronizeStream(stream));

  std::vector<uint8_t> dst_host(dst_bytes);
  std::vector<int32_t> index_out_host(static_cast<size_t>(k));
  CHECK_ACL(aclrtMemcpy(dst_host.data(), dst_bytes, dst_dev, dst_bytes, ACL_MEMCPY_DEVICE_TO_HOST));
  CHECK_ACL(aclrtMemcpy(index_out_host.data(), idx_bytes, index_out_dev, idx_bytes, ACL_MEMCPY_DEVICE_TO_HOST));

  int32_t rc = compare_bytes(reinterpret_cast<uint8_t*>(index_out_host.data()),
                             reinterpret_cast<const uint8_t*>(index_host.data()),
                             idx_bytes,
                             "index_out",
                             pe);
  rc |= compare_bytes(dst_host.data(), gold.data(), dst_bytes, "dst", pe);
  if (rc == 0 && pe == 0) {
    std::printf("case %s k=%d d=%d n=%d passed\n", name, k, d, n_rows);
  }

  aclshmem_free(flag);
  aclshmem_free(packed);
  CHECK_ACL(aclrtFree(dst_dev));
  CHECK_ACL(aclrtFree(index_out_dev));
  CHECK_ACL(aclrtFree(index_dev));
  CHECK_ACL(aclrtFree(src_dev));
  return rc;
}

int32_t set_init_attr(int32_t pe,
                      int32_t n_pes,
                      const char* ip_port,
                      aclshmemx_uniqueid_t* uid,
                      aclshmemx_init_attr_t* attributes) {
  attributes->my_pe = pe;
  attributes->n_pes = n_pes;
  attributes->local_mem_size = 64ULL * 1024ULL * 1024ULL;
  attributes->comm_args = uid;
  if (ip_port == nullptr) {
    return ACLSHMEM_INVALID_VALUE;
  }
  size_t ip_len = std::strlen(ip_port);
  if (ip_len >= ACLSHMEM_MAX_IP_PORT_LEN) {
    ip_len = ACLSHMEM_MAX_IP_PORT_LEN - 1;
  }
  std::memcpy(attributes->ip_port, ip_port, ip_len);
  attributes->ip_port[ip_len] = '\0';
  return ACLSHMEM_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s n_pes pe_id ipport n_npus [first_npu]\n",
                 argv[0]);
    return 1;
  }
  int32_t n_pes = static_cast<int32_t>(std::atoi(argv[1]));
  int32_t pe = static_cast<int32_t>(std::atoi(argv[2]));
  const char* ip_port = argv[3];
  int32_t n_npus = static_cast<int32_t>(std::atoi(argv[4]));
  int32_t first_npu = 0;
  if (argc > 5) {
    first_npu = static_cast<int32_t>(std::atoi(argv[5]));
  }
  if (n_pes < 2 || pe < 0 || pe >= n_pes || n_npus <= 0) {
    std::fprintf(stderr, "invalid n_pes/pe_id/n_npus\n");
    return 1;
  }

  int32_t device_id = pe % n_npus + first_npu;
  CHECK_ACL(aclInit(nullptr));
  CHECK_ACL(aclrtSetDevice(device_id));

  aclshmemx_uniqueid_t uid{};
  aclshmemx_init_attr_t attributes{};
  if (set_init_attr(pe, n_pes, ip_port, &uid, &attributes) != ACLSHMEM_SUCCESS) {
    std::fprintf(stderr, "set_init_attr failed\n");
    return 1;
  }
  int32_t shmem_rc = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
  if (shmem_rc != ACLSHMEM_SUCCESS) {
    std::fprintf(stderr, "aclshmemx_init_attr failed: %d\n", shmem_rc);
    return 1;
  }

  aclrtStream stream = nullptr;
  CHECK_ACL(aclrtCreateStream(&stream));
  uint64_t ffts_addr = util_get_ffts_config();

  std::vector<int32_t> idx_small = {1, -1, 0, 3, 2, 7, -8, 1};
  int32_t rc = run_case(pe,
                        stream,
                        ffts_addr,
                        /*n_rows=*/8,
                        /*k=*/8,
                        /*d=*/16,
                        idx_small,
                        /*seq=*/1,
                        "small_pad");

  std::vector<int32_t> idx_p0(2048);
  for (int32_t i = 0; i < 2048; ++i) {
    if (i % 17 == 0) {
      idx_p0[static_cast<size_t>(i)] = -1;
    } else if (i % 29 == 0) {
      idx_p0[static_cast<size_t>(i)] = 4096;
    } else {
      idx_p0[static_cast<size_t>(i)] = (i * 13) % 512;
    }
  }
  if (rc == 0) {
    rc = run_case(pe,
                  stream,
                  ffts_addr,
                  /*n_rows=*/512,
                  /*k=*/2048,
                  /*d=*/576,
                  idx_p0,
                  /*seq=*/2,
                  "selected_kv_2048x576");
  }

  CHECK_ACL(aclrtDestroyStream(stream));
  aclshmem_finalize();
  CHECK_ACL(aclrtResetDevice(device_id));
  CHECK_ACL(aclFinalize());
  if (rc != 0) {
    std::fprintf(stderr, "pe=%d failed\n", pe);
    return 1;
  }
  if (pe == 0) {
    std::printf("sparse_broadcast_test passed\n");
  }
  return 0;
}
