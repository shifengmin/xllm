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

#include "kernel_operator.h"
#include "shmem.h"

using namespace xllm::kernel::npu;

namespace {

ACLSHMEM_DEVICE uint32_t min_u32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

ACLSHMEM_DEVICE uint16_t slot_event(uint32_t slot) {
  switch (slot) {
    case 1:
      return EVENT_ID1;
    case 2:
      return EVENT_ID2;
    case 3:
      return EVENT_ID3;
    default:
      return EVENT_ID0;
  }
}

ACLSHMEM_DEVICE __ubuf__ uint8_t* ub_ptr(uint32_t offset) {
  return reinterpret_cast<__ubuf__ uint8_t*>(static_cast<uint64_t>(offset));
}

ACLSHMEM_DEVICE void copy_gm2ub_bytes(__ubuf__ uint8_t* dst, __gm__ uint8_t* src, uint32_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  aclshmemi_copy_gm2ub(dst, src, nbytes, /*toL2Cache=*/false);
}

ACLSHMEM_DEVICE void copy_ub2gm_bytes(__gm__ uint8_t* dst, __ubuf__ uint8_t* src, uint32_t nbytes) {
  if (nbytes == 0) {
    return;
  }
  aclshmemi_copy_ub2gm(dst, src, nbytes, /*toL2Cache=*/false);
}

ACLSHMEM_DEVICE void drain_mte3(uint16_t event) {
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
}

ACLSHMEM_DEVICE void wait_mte2_then_store(uint16_t event) {
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
}

ACLSHMEM_DEVICE void fill_zero_row(__ubuf__ uint8_t* dst, uint32_t row_bytes) {
  AscendC::LocalTensor<uint16_t> zero;
  zero.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECOUT);
  zero.address_.bufferAddr = reinterpret_cast<uint64_t>(dst);
  zero.address_.dataLen = sparse_bcast_align32(row_bytes);
  uint32_t count = row_bytes / static_cast<uint32_t>(sizeof(uint16_t));
  if (count == 0) {
    return;
  }
  AscendC::Duplicate(zero, static_cast<uint16_t>(0), count);
  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID7);
}

ACLSHMEM_DEVICE void write_header(__gm__ uint8_t* packed, uint32_t k, uint32_t d, int32_t seq) {
  __ubuf__ uint32_t* hdr = reinterpret_cast<__ubuf__ uint32_t*>(ub_ptr(kSparseBcastUbBase));
  hdr[0] = kSparseBcastMagic;
  hdr[1] = k;
  hdr[2] = d;
  hdr[3] = static_cast<uint32_t>(seq);
  for (uint32_t i = 4; i < (kSparseBcastHeaderBytes / sizeof(uint32_t)); ++i) {
    hdr[i] = 0;
  }
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
  copy_ub2gm_bytes(packed, ub_ptr(kSparseBcastUbBase), kSparseBcastHeaderBytes);
  drain_mte3(EVENT_ID0);
}

ACLSHMEM_DEVICE void scatter_ub_rows(__gm__ uint8_t* dst,
                                     __ubuf__ uint8_t* ub_slot,
                                     __ubuf__ int32_t* idx_ub,
                                     uint32_t row_begin,
                                     uint32_t nrows,
                                     uint32_t row_bytes,
                                     uint32_t row_stride,
                                     int32_t n_rows) {
  for (uint32_t r = 0; r < nrows; ++r) {
    int32_t dst_idx = idx_ub[row_begin + r];
    if (dst_idx < 0 || dst_idx >= n_rows) {
      continue;
    }
    uint64_t dst_off = static_cast<uint64_t>(dst_idx) * static_cast<uint64_t>(row_bytes);
    copy_ub2gm_bytes(dst + dst_off, ub_slot + r * row_stride, row_bytes);
  }
}

ACLSHMEM_DEVICE void get_and_scatter(__gm__ uint8_t* dst,
                                     __gm__ uint8_t* remote_sel,
                                     __ubuf__ int32_t* idx_ub,
                                     __ubuf__ uint8_t* ub_base,
                                     uint32_t row_begin,
                                     uint32_t row_count,
                                     uint32_t row_bytes,
                                     uint32_t row_stride,
                                     int32_t n_rows) {
  if (row_count == 0) {
    return;
  }
  uint32_t chunk_rows = min_u32(row_count, sparse_bcast_chunk_rows(row_stride));
  uint32_t nchunk = (row_count + chunk_rows - 1u) / chunk_rows;
  uint32_t depth = min_u32(kSparseBcastDepth, nchunk);
  uint32_t issued = 0;
  for (; issued < depth; ++issued) {
    uint32_t nrows = min_u32(chunk_rows, row_count - issued * chunk_rows);
    uint32_t nbytes = nrows * row_stride;
    uint32_t packed_off = (row_begin + issued * chunk_rows) * row_stride;
    copy_gm2ub_bytes(ub_base + issued * kSparseBcastChunkBytes, remote_sel + packed_off, nbytes);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(slot_event(issued));
  }
  uint32_t done = 0;
  while (done < nchunk) {
    uint32_t slot = done % kSparseBcastDepth;
    uint16_t ev = slot_event(slot);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(ev);
    uint32_t nrows = min_u32(chunk_rows, row_count - done * chunk_rows);
    scatter_ub_rows(dst,
                    ub_base + slot * kSparseBcastChunkBytes,
                    idx_ub,
                    row_begin + done * chunk_rows,
                    nrows,
                    row_bytes,
                    row_stride,
                    n_rows);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
    ++done;
    if (issued < nchunk) {
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(ev);
      uint32_t nrows2 = min_u32(chunk_rows, row_count - issued * chunk_rows);
      uint32_t nbytes2 = nrows2 * row_stride;
      uint32_t packed_off2 = (row_begin + issued * chunk_rows) * row_stride;
      copy_gm2ub_bytes(ub_base + slot * kSparseBcastChunkBytes, remote_sel + packed_off2, nbytes2);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(ev);
      ++issued;
    }
  }
  for (uint32_t slot = 0; slot < depth; ++slot) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot_event(slot));
  }
}

ACLSHMEM_DEVICE void load_index(__gm__ uint8_t* index_src,
                                __gm__ int32_t* index_out,
                                __ubuf__ uint8_t* ub_index,
                                uint32_t index_bytes,
                                uint32_t core_idx) {
  copy_gm2ub_bytes(ub_index, index_src, index_bytes);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID4);
  if (core_idx == 0) {
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID4);
    copy_ub2gm_bytes(reinterpret_cast<__gm__ uint8_t*>(index_out), ub_index, index_bytes);
    drain_mte3(EVENT_ID4);
  }
}

ACLSHMEM_DEVICE void pack_rows(__gm__ uint8_t* src,
                               __gm__ int32_t* index,
                               __gm__ int32_t* index_out,
                               __gm__ uint8_t* dst,
                               __gm__ uint8_t* packed,
                               int32_t n_rows,
                               uint32_t k,
                               uint32_t d,
                               uint32_t elem_size,
                               uint32_t core_idx,
                               uint32_t ncores) {
  uint32_t row_bytes = sparse_bcast_row_bytes(d, elem_size);
  uint32_t row_stride = sparse_bcast_row_stride(d, elem_size);
  uint32_t index_bytes = k * static_cast<uint32_t>(sizeof(int32_t));
  uint32_t index_copy_bytes = sparse_bcast_index_bytes(k);
  __ubuf__ uint8_t* ub_index = ub_ptr(kSparseBcastUbBase);
  __ubuf__ uint8_t* ub_zero = ub_ptr(kSparseBcastUbBase + index_copy_bytes);
  __ubuf__ uint8_t* ub_row = ub_zero + sparse_bcast_align32(row_bytes);

  copy_gm2ub_bytes(ub_index, reinterpret_cast<__gm__ uint8_t*>(index), index_bytes);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

  fill_zero_row(ub_zero, row_bytes);

  if (core_idx == 0) {
    copy_ub2gm_bytes(packed + sparse_bcast_index_offset(), ub_index, index_copy_bytes);
    copy_ub2gm_bytes(reinterpret_cast<__gm__ uint8_t*>(index_out), ub_index, index_bytes);
    drain_mte3(EVENT_ID0);
  }

  uint32_t row_begin = 0;
  uint32_t row_count = 0;
  sparse_bcast_split_core_rows(k, ncores, core_idx, &row_begin, &row_count);
  __ubuf__ int32_t* idx_ub = reinterpret_cast<__ubuf__ int32_t*>(ub_index);
  __gm__ uint8_t* packed_sel = packed + sparse_bcast_selected_offset(k);

  for (uint32_t r = 0; r < row_count; ++r) {
    uint32_t row = row_begin + r;
    int32_t src_idx = idx_ub[row];
    if (src_idx < 0 || src_idx >= n_rows) {
      copy_ub2gm_bytes(packed_sel + row * row_stride, ub_zero, row_bytes);
    } else {
      uint64_t src_off = static_cast<uint64_t>(src_idx) * static_cast<uint64_t>(row_bytes);
      copy_gm2ub_bytes(ub_row, src + src_off, row_bytes);
      wait_mte2_then_store(EVENT_ID0);
      copy_ub2gm_bytes(packed_sel + row * row_stride, ub_row, row_bytes);
      copy_ub2gm_bytes(dst + src_off, ub_row, row_bytes);
    }
    drain_mte3(EVENT_ID0);
  }
}

ACLSHMEM_DEVICE void signal_packed(__gm__ int32_t* flag, int32_t seq, int32_t my_pe, int32_t n_pes) {
  for (int32_t pe = 0; pe < n_pes; ++pe) {
    if (pe == my_pe) {
      continue;
    }
    aclshmemx_signal_op(flag, seq, ACLSHMEM_SIGNAL_SET, pe);
  }
}

ACLSHMEM_DEVICE void sparse_broadcast_impl(uint64_t ffts_addr,
                                           __gm__ uint8_t* src,
                                           __gm__ int32_t* index,
                                           __gm__ int32_t* index_out,
                                           __gm__ uint8_t* dst,
                                           __gm__ uint8_t* packed,
                                           __gm__ int32_t* flag,
                                           int32_t n_rows,
                                           int32_t k,
                                           int32_t d,
                                           int32_t elem_size,
                                           int32_t root,
                                           int32_t seq) {
  util_set_ffts_config(ffts_addr);
  AscendC::TPipe pipe;
  (void)pipe;
  int32_t my_pe = aclshmem_my_pe();
  int32_t n_pes = aclshmem_n_pes();
  uint32_t core_idx = static_cast<uint32_t>(AscendC::GetBlockIdx());
  uint32_t ncores = static_cast<uint32_t>(AscendC::GetBlockNum());
  uint32_t ku = static_cast<uint32_t>(k);
  uint32_t du = static_cast<uint32_t>(d);
  uint32_t es = static_cast<uint32_t>(elem_size);

  if (my_pe == root) {
    pack_rows(src, index, index_out, dst, packed, n_rows, ku, du, es, core_idx, ncores);
    aclshmemi_barrier_core_soft();
    if (core_idx == 0) {
      write_header(packed, ku, du, seq);
      signal_packed(flag, seq, my_pe, n_pes);
    }
    return;
  }

  __gm__ uint8_t* remote_packed = reinterpret_cast<__gm__ uint8_t*>(aclshmem_ptr(packed, root));
  uint32_t index_bytes = ku * static_cast<uint32_t>(sizeof(int32_t));
  uint32_t index_copy_bytes = sparse_bcast_index_bytes(ku);
  __ubuf__ uint8_t* ub_index = ub_ptr(kSparseBcastUbBase);
  __ubuf__ uint8_t* ub_base = ub_ptr(kSparseBcastUbBase + index_copy_bytes);

  aclshmem_signal_wait_until(flag, ACLSHMEM_CMP_EQ, seq);
  AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID4);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID4);
  load_index(remote_packed + sparse_bcast_index_offset(), index_out, ub_index, index_bytes, core_idx);

  uint32_t row_begin = 0;
  uint32_t row_count = 0;
  sparse_bcast_split_core_rows(ku, ncores, core_idx, &row_begin, &row_count);
  uint32_t row_bytes = sparse_bcast_row_bytes(du, es);
  uint32_t row_stride = sparse_bcast_row_stride(du, es);
  get_and_scatter(dst,
                  remote_packed + sparse_bcast_selected_offset(ku),
                  reinterpret_cast<__ubuf__ int32_t*>(ub_index),
                  ub_base,
                  row_begin,
                  row_count,
                  row_bytes,
                  row_stride,
                  n_rows);
}

}  // namespace

extern "C" ACLSHMEM_GLOBAL_VECTOR void SparseBroadcastKernel(uint64_t ffts_addr,
                                                             GM_ADDR src,
                                                             GM_ADDR index,
                                                             GM_ADDR index_out,
                                                             GM_ADDR dst,
                                                             GM_ADDR packed,
                                                             GM_ADDR flag,
                                                             int32_t n_rows,
                                                             int32_t k,
                                                             int32_t d,
                                                             int32_t elem_size,
                                                             int32_t root,
                                                             int32_t seq) {
  sparse_broadcast_impl(ffts_addr,
                        reinterpret_cast<__gm__ uint8_t*>(src),
                        reinterpret_cast<__gm__ int32_t*>(index),
                        reinterpret_cast<__gm__ int32_t*>(index_out),
                        reinterpret_cast<__gm__ uint8_t*>(dst),
                        reinterpret_cast<__gm__ uint8_t*>(packed),
                        reinterpret_cast<__gm__ int32_t*>(flag),
                        n_rows,
                        k,
                        d,
                        elem_size,
                        root,
                        seq);
}

void xllm::kernel::npu::launch_sparse_broadcast(const SparseBroadcastLaunchArgs& args) {
  uint32_t block_dim = args.block_dim == 0 ? kSparseBcastBlockDim : args.block_dim;
  SparseBroadcastKernel<<<block_dim, nullptr, args.stream>>>(args.ffts_addr,
                                                             args.src,
                                                             reinterpret_cast<GM_ADDR>(args.index),
                                                             reinterpret_cast<GM_ADDR>(args.index_out),
                                                             args.dst,
                                                             args.packed,
                                                             reinterpret_cast<GM_ADDR>(args.flag),
                                                             args.n_rows,
                                                             args.k,
                                                             args.d,
                                                             args.elem_size,
                                                             args.root,
                                                             args.seq);
}
