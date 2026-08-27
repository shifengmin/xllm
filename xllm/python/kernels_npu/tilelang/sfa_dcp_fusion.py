#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TileLang-Ascend fused DCP SFA pre/post kernels.

Expert/mixed vector kernels. AllGather and AllToAll stay outside.
Production decode and ACL graph use the AOT wrappers ``sfa_dcp_remap`` /
``sfa_dcp_merge``. ``@tilelang.jit`` helpers here are only for local
Python precision checks.
"""

import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.kernels_npu.tilelang.utils import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    detect_vec_core_num,
    mte2_notify_v,
    mte2_wait_mte3,
    mte2_wait_v,
    mte3_notify_mte2,
    mte3_notify_v,
    mte3_wait_v,
    v_notify_mte2,
    v_notify_mte3,
    v_wait_mte2,
    v_wait_mte3,
)

VEC_NUM = 2
MAX_TOKENS = 256
MAX_HEADS = 1024
MERGE_MAX_HEADS = 256
MAX_TOPK = 2048
REMAP_VEC_LEN = 64
REMAP_COPY_LEN = 32
REMAP_EVT_SRC0 = 0
REMAP_EVT_SRC1 = 1
REMAP_EVT_IDX = 2
REMAP_EVT_OUT = 3
MERGE_EVT_LSE = 0
MERGE_EVT_SRC0 = 1
MERGE_EVT_SRC1 = 2
MERGE_EVT_STORE = 3
NOPE_DIM = 512
PE_DIM = 64
HEAD_DIM = 512
FUSED_Q_DIM = NOPE_DIM + PE_DIM
SUPPORTED_DTYPES = ("bf16", "float16")
TENSOR_DTYPES = {
    "bf16": "bfloat16",
    "float16": "float16",
}
FUSION_PASS_CONFIGS = {
    **DEFAULT_ASCEND_PASS_CONFIGS,
    "tl.disable_safe_memory_legalize": True,
}
# Vector-only remap / merge: manual MTE2/V/MTE3 flags, no AIC combine.
REMAP_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
    "tl.disable_safe_memory_legalize": True,
}
MERGE_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
    "tl.disable_safe_memory_legalize": True,
}
SUPPORTED_DCP_SIZES = (2, 4, 8)
SUPPORTED_REMAP_TOPK = (32, 2048)
SUPPORTED_SHARD_SIZES = (2, 4, 8)
DEFAULT_PHYSICAL_BLOCK_SIZE = 128
DEFAULT_MERGE_DTYPE = "bf16"


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "float16"
    raise RuntimeError(f"SFA DCP fusion kernels only support bf16/float16, got {dtype}")


def _require_npu(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.device.type != "npu":
            raise RuntimeError(f"SFA DCP fusion kernels require NPU tensors, got device={tensor.device}")
        if not tensor.is_contiguous():
            raise RuntimeError(f"SFA DCP fusion kernels require contiguous tensors, got shape={tuple(tensor.shape)}")


def _launch_vec_core_num(task_count: int) -> int:
    available = detect_vec_core_num()
    if task_count <= 0:
        return VEC_NUM
    needed = ((int(task_count) + VEC_NUM - 1) // VEC_NUM) * VEC_NUM
    return min(available, max(VEC_NUM, needed))


def remap_scratch_numel(num_tokens: int, index_topk: int) -> int:
    width = REMAP_VEC_LEN if index_topk <= REMAP_VEC_LEN else index_topk
    return int(num_tokens) * width


def _require_match(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    name: str,
) -> None:
    _require_npu(tensor)
    if tuple(tensor.shape) != shape or tensor.dtype != dtype:
        raise RuntimeError(
            f"{name} expects shape {shape} dtype={dtype}, got {tuple(tensor.shape)} dtype={tensor.dtype}"
        )


def _pow2_shift(value: int, name: str) -> int:
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"{name}({value}) must be a positive power of 2 for vector remap")
    return value.bit_length() - 1


def build_pack_query_kernel(*, dtype: str, vec_core_num: int):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"pack_query only supports {SUPPORTED_DTYPES}, got {dtype}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    tensor_dtype = TENSOR_DTYPES[dtype]
    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num
    max_nope = MAX_TOKENS * MAX_HEADS * NOPE_DIM
    max_pe = MAX_TOKENS * MAX_HEADS * PE_DIM
    max_send = MAX_HEADS * MAX_TOKENS * FUSED_Q_DIM

    @T.prim_func
    def pack_query_kernel(
        ql_nope: T.Tensor((1, max_nope), tensor_dtype),
        q_pe: T.Tensor((1, max_pe), tensor_dtype),
        send: T.Tensor((1, max_send), tensor_dtype),
        num_tokens: T.int32,
        num_heads: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total = num_tokens * num_heads
            per_task = (total + task_num - 1) // task_num
            row_start = task_id * per_task
            rows_left = T.if_then_else(total > row_start, total - row_start, 0)
            row_count = T.if_then_else(rows_left < per_task, rows_left, per_task)

            with T.Scope("V"):
                nope_ub = T.alloc_ub((NOPE_DIM,), tensor_dtype)
                pe_ub = T.alloc_ub((PE_DIM,), tensor_dtype)
                fused_ub = T.alloc_ub((FUSED_Q_DIM,), tensor_dtype)
                for local_row in T.serial(row_count):
                    flat = row_start + local_row
                    token = flat // num_heads
                    head = flat - token * num_heads
                    nope_off = (token * num_heads + head) * NOPE_DIM
                    pe_off = (token * num_heads + head) * PE_DIM
                    send_off = (head * num_tokens + token) * FUSED_Q_DIM
                    T.copy(ql_nope[0, nope_off], nope_ub)
                    T.copy(q_pe[0, pe_off], pe_ub)
                    T.copy(nope_ub, fused_ub[0:NOPE_DIM])
                    T.copy(pe_ub, fused_ub[NOPE_DIM:FUSED_Q_DIM])
                    T.copy(fused_ub, send[0, send_off])

    return pack_query_kernel


def build_unpack_query_kernel(*, dtype: str, vec_core_num: int):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"unpack_query only supports {SUPPORTED_DTYPES}, got {dtype}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    tensor_dtype = TENSOR_DTYPES[dtype]
    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num
    max_nope = MAX_TOKENS * MAX_HEADS * NOPE_DIM
    max_pe = MAX_TOKENS * MAX_HEADS * PE_DIM
    max_gathered = MAX_HEADS * MAX_TOKENS * FUSED_Q_DIM

    @T.prim_func
    def unpack_query_kernel(
        gathered: T.Tensor((1, max_gathered), tensor_dtype),
        ql_nope: T.Tensor((1, max_nope), tensor_dtype),
        q_pe: T.Tensor((1, max_pe), tensor_dtype),
        num_tokens: T.int32,
        num_heads: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total = num_tokens * num_heads
            per_task = (total + task_num - 1) // task_num
            row_start = task_id * per_task
            rows_left = T.if_then_else(total > row_start, total - row_start, 0)
            row_count = T.if_then_else(rows_left < per_task, rows_left, per_task)

            with T.Scope("V"):
                nope_ub = T.alloc_ub((NOPE_DIM,), tensor_dtype)
                pe_ub = T.alloc_ub((PE_DIM,), tensor_dtype)
                fused_ub = T.alloc_ub((FUSED_Q_DIM,), tensor_dtype)
                for local_row in T.serial(row_count):
                    flat = row_start + local_row
                    token = flat // num_heads
                    head = flat - token * num_heads
                    gathered_off = (head * num_tokens + token) * FUSED_Q_DIM
                    nope_off = (token * num_heads + head) * NOPE_DIM
                    pe_off = (token * num_heads + head) * PE_DIM
                    T.copy(gathered[0, gathered_off], fused_ub)
                    T.copy(fused_ub[0:NOPE_DIM], nope_ub)
                    T.copy(fused_ub[NOPE_DIM:FUSED_Q_DIM], pe_ub)
                    T.copy(nope_ub, ql_nope[0, nope_off])
                    T.copy(pe_ub, q_pe[0, pe_off])

    return unpack_query_kernel


def build_remap_topk_kernel(
    *,
    vec_core_num: int,
    topk: int = MAX_TOPK,
    physical_block_size: int,
    shard_size: int,
):
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")
    if topk <= 0 or topk > MAX_TOPK:
        raise ValueError(f"topk({topk}) must be in (0, {MAX_TOPK}]")
    if topk > REMAP_COPY_LEN and topk % 32 != 0:
        raise ValueError(f"topk({topk}) must be {REMAP_COPY_LEN} or a multiple of 32 for vector remap")

    phys_shift = _pow2_shift(physical_block_size, "physical_block_size")
    logical_block_size = physical_block_size * shard_size
    logical_shift = _pow2_shift(logical_block_size, "logical_block_size")
    logical_mask = logical_block_size - 1
    phys_mask = physical_block_size - 1
    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num
    max_elems = MAX_TOKENS * MAX_TOPK
    # 64-wide int vector is only correct on the low 32 lanes. topk=32 pads
    # to 64 and stays on that path. topk=2048 localizes in FP32 at full
    # width (int bitwise at 2048 drops the high half; float ALU does not).
    # Integer slots must fit in the 24-bit mantissa (physical/logical are
    # powers of two, so floor-div by reciprocal is exact in that range).
    vec_len = REMAP_VEC_LEN
    copy_len = REMAP_COPY_LEN
    small_topk = topk <= copy_len
    inv_logical = 1.0 / float(logical_block_size)
    inv_physical = 1.0 / float(physical_block_size)
    mask_len = 32
    full_mask_len = max(32, topk // 8)
    neg_inf = -1.0e30
    gather_elem_bytes = 4

    @T.macro
    def localize_chunk(
        src_fp_ub,
        local_ub,
        local_fp_ub,
        off_ub,
        owner_ub,
        block_ub,
        local_off_ub,
        logical_mask_ub,
        phys_mask_ub,
        rank_fp_ub,
        ge_mask,
        owner_mask,
    ):
        T.tile.compare(ge_mask, src_fp_ub, T.float32(0.0), "GE")
        T.tile.cast(local_ub, src_fp_ub, "CAST_RINT", vec_len)
        T.tile.bitwise_and(off_ub, local_ub, logical_mask_ub)
        T.tile.bitwise_rshift(owner_ub, off_ub, T.int32(phys_shift))
        T.tile.bitwise_rshift(block_ub, local_ub, T.int32(logical_shift))
        T.tile.bitwise_and(local_off_ub, off_ub, phys_mask_ub)
        T.tile.mul(local_ub, block_ub, T.int32(physical_block_size))
        T.tile.add(local_ub, local_ub, local_off_ub)
        T.tile.cast(local_fp_ub, local_ub, "CAST_NONE", vec_len)
        T.tile.cast(src_fp_ub, owner_ub, "CAST_NONE", vec_len)
        T.tile.compare(owner_mask, src_fp_ub, rank_fp_ub, "EQ")
        T.tile.select(
            local_fp_ub,
            ge_mask,
            local_fp_ub,
            T.float32(-1.0),
            "VSEL_TENSOR_SCALAR_MODE",
        )
        T.tile.select(
            local_fp_ub,
            owner_mask,
            local_fp_ub,
            T.float32(-1.0),
            "VSEL_TENSOR_SCALAR_MODE",
        )
        T.tile.cast(local_ub, local_fp_ub, "CAST_RINT", vec_len)

    @T.prim_func
    def remap_topk_kernel(
        topk_in: T.Tensor((max_elems,), "int32"),
        topk_out: T.Tensor((max_elems,), "int32"),
        idx_scratch: T.Tensor((max_elems,), "int32"),
        idx_scratch_u: T.Tensor((max_elems,), "uint32"),
        num_tokens: T.int32,
        shard_rank: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            per_task = (num_tokens + task_num - 1) // task_num
            token_start = task_id * per_task
            tokens_left = T.if_then_else(num_tokens > token_start, num_tokens - token_start, 0)
            token_count = T.if_then_else(tokens_left < per_task, tokens_left, per_task)

            with T.Scope("V"):
                src0_ub = T.alloc_ub((vec_len,), "int32")
                src1_ub = T.alloc_ub((vec_len,), "int32")
                local_ub = T.alloc_ub((vec_len,), "int32")
                logical_mask_ub = T.alloc_ub((vec_len,), "int32")
                phys_mask_ub = T.alloc_ub((vec_len,), "int32")
                off_ub = T.alloc_ub((vec_len,), "int32")
                owner_ub = T.alloc_ub((vec_len,), "int32")
                rank_ub = T.alloc_ub((vec_len,), "int32")
                block_ub = T.alloc_ub((vec_len,), "int32")
                local_off_ub = T.alloc_ub((vec_len,), "int32")
                idx_u_ub = T.alloc_ub((vec_len,), "uint32")
                src_fp_ub = T.alloc_ub((vec_len,), "float32")
                local_fp_ub = T.alloc_ub((vec_len,), "float32")
                rank_fp_ub = T.alloc_ub((vec_len,), "float32")
                keys_ub = T.alloc_ub((vec_len,), "float32")
                idx_fp_ub = T.alloc_ub((vec_len,), "float32")
                out_fp_ub = T.alloc_ub((vec_len,), "float32")
                sort_ub = T.alloc_ub((vec_len * 2,), "float32")
                ge_mask = T.alloc_ub((mask_len,), "uint8")
                owner_mask = T.alloc_ub((mask_len,), "uint8")
                full_src_ub = T.alloc_ub((topk,), "int32")
                full_off_ub = T.alloc_ub((topk,), "int32")
                full_idx_u_ub = T.alloc_ub((topk,), "uint32")
                full_fp_ub = T.alloc_ub((topk,), "float32")
                full_keys_ub = T.alloc_ub((topk,), "float32")
                full_idx_fp_ub = T.alloc_ub((topk,), "float32")
                full_out_fp_ub = T.alloc_ub((topk,), "float32")
                full_sort_ub = T.alloc_ub((topk * 2,), "float32")
                full_ge_mask = T.alloc_ub((full_mask_len,), "uint8")
                full_owner_mask = T.alloc_ub((full_mask_len,), "uint8")
                work0_fp_ub = T.alloc_ub((topk,), "float32")
                T.tile.fill(logical_mask_ub, T.int32(logical_mask))
                T.tile.fill(phys_mask_ub, T.int32(phys_mask))
                T.tile.fill(rank_ub, shard_rank)
                T.tile.cast(rank_fp_ub, rank_ub, "CAST_NONE", vec_len)
                T.tile.fill(src0_ub, -1)
                T.tile.fill(src1_ub, -1)
                v_notify_mte2(REMAP_EVT_SRC0)
                mte2_wait_v(REMAP_EVT_SRC0)
                for local_token in T.serial(token_count):
                    token = token_start + local_token
                    src_off = token * topk
                    with T.If(local_token > 0), T.Then():
                        v_wait_mte3(REMAP_EVT_OUT)
                        mte2_wait_mte3(REMAP_EVT_OUT)
                    if small_topk:
                        T.copy(topk_in[src_off], src0_ub[0:copy_len])
                        mte2_notify_v(REMAP_EVT_SRC0)
                        v_wait_mte2(REMAP_EVT_SRC0)
                        T.tile.cast(src_fp_ub, src0_ub, "CAST_NONE", vec_len)
                        localize_chunk(
                            src_fp_ub,
                            local_ub,
                            local_fp_ub,
                            off_ub,
                            owner_ub,
                            block_ub,
                            local_off_ub,
                            logical_mask_ub,
                            phys_mask_ub,
                            rank_fp_ub,
                            ge_mask,
                            owner_mask,
                        )
                        T.tile.arith_progression(
                            keys_ub,
                            T.float32(vec_len),
                            T.float32(-1.0),
                            vec_len,
                        )
                        T.tile.select(
                            keys_ub,
                            ge_mask,
                            keys_ub,
                            T.float32(neg_inf),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.select(
                            keys_ub,
                            owner_mask,
                            keys_ub,
                            T.float32(neg_inf),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.sort(sort_ub, keys_ub, vec_len)
                        T.tile.gather_mask(idx_fp_ub, sort_ub, "P1010")
                        T.tile.cast(off_ub, idx_fp_ub, "CAST_RINT", vec_len)
                        T.tile.mul(off_ub, off_ub, T.int32(gather_elem_bytes))
                        idx_off = token * vec_len
                        v_notify_mte3(REMAP_EVT_IDX)
                        mte3_wait_v(REMAP_EVT_IDX)
                        T.copy(off_ub, idx_scratch[idx_off])
                        mte3_notify_mte2(REMAP_EVT_IDX)
                        mte2_wait_mte3(REMAP_EVT_IDX)
                        T.copy(idx_scratch_u[idx_off], idx_u_ub)
                        mte2_notify_v(REMAP_EVT_IDX)
                        v_wait_mte2(REMAP_EVT_IDX)
                        T.tile.gather(out_fp_ub, local_fp_ub, idx_u_ub, 0)
                        T.tile.cast(local_ub, out_fp_ub, "CAST_RINT", vec_len)
                        v_notify_mte3(REMAP_EVT_OUT)
                        mte3_wait_v(REMAP_EVT_OUT)
                        T.copy(local_ub[0:copy_len], topk_out[src_off])
                        mte3_notify_mte2(REMAP_EVT_OUT)
                        mte3_notify_v(REMAP_EVT_OUT)
                    else:
                        T.copy(topk_in[src_off], full_src_ub)
                        mte2_notify_v(REMAP_EVT_SRC0)
                        v_wait_mte2(REMAP_EVT_SRC0)
                        T.tile.cast(full_fp_ub, full_src_ub, "CAST_NONE", topk)
                        T.tile.compare(full_ge_mask, full_fp_ub, T.float32(0.0), "GE")
                        T.tile.select(
                            full_keys_ub,
                            full_ge_mask,
                            full_fp_ub,
                            T.float32(0.0),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.mul(
                            full_idx_fp_ub,
                            full_keys_ub,
                            T.float32(inv_logical),
                        )
                        T.tile.cast(full_off_ub, full_idx_fp_ub, "CAST_FLOOR", topk)
                        T.tile.cast(work0_fp_ub, full_off_ub, "CAST_NONE", topk)
                        T.tile.mul(
                            full_idx_fp_ub,
                            work0_fp_ub,
                            T.float32(logical_block_size),
                        )
                        T.tile.sub(full_keys_ub, full_keys_ub, full_idx_fp_ub)
                        T.tile.mul(
                            full_idx_fp_ub,
                            full_keys_ub,
                            T.float32(inv_physical),
                        )
                        T.tile.cast(full_off_ub, full_idx_fp_ub, "CAST_FLOOR", topk)
                        T.tile.cast(full_idx_fp_ub, full_off_ub, "CAST_NONE", topk)
                        T.tile.compare(
                            full_owner_mask,
                            full_idx_fp_ub,
                            T.Cast("float32", shard_rank),
                            "EQ",
                        )
                        T.tile.mul(
                            full_out_fp_ub,
                            full_idx_fp_ub,
                            T.float32(physical_block_size),
                        )
                        T.tile.sub(full_keys_ub, full_keys_ub, full_out_fp_ub)
                        T.tile.mul(
                            full_out_fp_ub,
                            work0_fp_ub,
                            T.float32(physical_block_size),
                        )
                        T.tile.add(full_fp_ub, full_out_fp_ub, full_keys_ub)
                        T.tile.select(
                            full_fp_ub,
                            full_ge_mask,
                            full_fp_ub,
                            T.float32(-1.0),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.select(
                            full_fp_ub,
                            full_owner_mask,
                            full_fp_ub,
                            T.float32(-1.0),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(full_ge_mask, full_fp_ub, T.float32(0.0), "GE")
                        T.tile.arith_progression(
                            full_keys_ub,
                            T.float32(topk),
                            T.float32(-1.0),
                            topk,
                        )
                        T.tile.select(
                            full_keys_ub,
                            full_ge_mask,
                            full_keys_ub,
                            T.float32(neg_inf),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.sort(full_sort_ub, full_keys_ub, topk)
                        T.tile.gather_mask(full_idx_fp_ub, full_sort_ub, "P1010")
                        T.tile.cast(full_off_ub, full_idx_fp_ub, "CAST_RINT", topk)
                        T.tile.mul(full_off_ub, full_off_ub, T.int32(gather_elem_bytes))
                        idx_base = token * topk
                        v_notify_mte3(REMAP_EVT_IDX)
                        mte3_wait_v(REMAP_EVT_IDX)
                        T.copy(full_off_ub, idx_scratch[idx_base])
                        mte3_notify_mte2(REMAP_EVT_IDX)
                        mte2_wait_mte3(REMAP_EVT_IDX)
                        T.copy(idx_scratch_u[idx_base], full_idx_u_ub)
                        mte2_notify_v(REMAP_EVT_IDX)
                        v_wait_mte2(REMAP_EVT_IDX)
                        T.tile.gather(full_out_fp_ub, full_fp_ub, full_idx_u_ub, 0)
                        T.tile.cast(full_src_ub, full_out_fp_ub, "CAST_RINT", topk)
                        v_notify_mte3(REMAP_EVT_OUT)
                        mte3_wait_v(REMAP_EVT_OUT)
                        T.copy(full_src_ub, topk_out[src_off])
                        mte3_notify_mte2(REMAP_EVT_OUT)
                        mte3_notify_v(REMAP_EVT_OUT)
                with T.If(token_count > 0), T.Then():
                    v_wait_mte3(REMAP_EVT_OUT)
                    mte2_wait_mte3(REMAP_EVT_OUT)

    return remap_topk_kernel


def build_pack_a2a_kernel(*, dtype: str, vec_core_num: int, head_dim: int = HEAD_DIM):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"pack_a2a only supports {SUPPORTED_DTYPES}, got {dtype}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")
    if head_dim != HEAD_DIM:
        raise ValueError(f"pack_a2a head_dim must be {HEAD_DIM}, got {head_dim}")

    tensor_dtype = TENSOR_DTYPES[dtype]
    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num
    max_out = MAX_TOKENS * MAX_HEADS * HEAD_DIM
    max_lse = MAX_HEADS * MAX_TOKENS
    max_softmax = MAX_HEADS * MAX_TOKENS

    @T.prim_func
    def pack_a2a_kernel(
        sfa_output: T.Tensor((1, max_out), tensor_dtype),
        softmax_max: T.Tensor((1, max_softmax), "float32"),
        softmax_sum: T.Tensor((1, max_softmax), "float32"),
        out_send: T.Tensor((1, max_out), tensor_dtype),
        lse_send: T.Tensor((1, max_lse), "float32"),
        num_tokens: T.int32,
        num_heads: T.int32,
        n2_size: T.int32,
        g_size: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total = num_tokens * num_heads
            per_task = (total + task_num - 1) // task_num
            row_start = task_id * per_task
            rows_left = T.if_then_else(total > row_start, total - row_start, 0)
            row_count = T.if_then_else(rows_left < per_task, rows_left, per_task)

            with T.Scope("V"):
                out_ub = T.alloc_ub((HEAD_DIM,), tensor_dtype)
                max_ub = T.alloc_ub((8,), "float32")
                sum_ub = T.alloc_ub((8,), "float32")
                lse_ub = T.alloc_ub((8,), "float32")
                for local_row in T.serial(row_count):
                    flat = row_start + local_row
                    token = flat // num_heads
                    head = flat - token * num_heads
                    out_src_off = (token * num_heads + head) * HEAD_DIM
                    out_dst_off = (head * num_tokens + token) * HEAD_DIM
                    T.copy(sfa_output[0, out_src_off], out_ub)
                    T.copy(out_ub, out_send[0, out_dst_off])
                T.barrier_all()
                for local_row in T.serial(row_count):
                    flat = row_start + local_row
                    token = flat // num_heads
                    head = flat - token * num_heads
                    n2 = head // g_size
                    group = head - n2 * g_size
                    softmax_off = (n2 * num_tokens + token) * g_size + group
                    lse_dst_off = head * num_tokens + token
                    T.copy(softmax_max[0, softmax_off : softmax_off + 1], max_ub[0:1])
                    T.copy(softmax_sum[0, softmax_off : softmax_off + 1], sum_ub[0:1])
                    T.tile.ln(lse_ub, sum_ub)
                    T.tile.add(lse_ub, lse_ub, max_ub)
                    T.copy(lse_ub[0:1], lse_send[0, lse_dst_off : lse_dst_off + 1])

    return pack_a2a_kernel


def build_merge_lse_kernel(
    *,
    dtype: str,
    vec_core_num: int,
    dcp_size: int,
    head_dim: int = HEAD_DIM,
):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"merge_lse only supports {SUPPORTED_DTYPES}, got {dtype}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")
    if dcp_size not in SUPPORTED_DCP_SIZES:
        raise ValueError(f"merge_lse dcp_size must be one of {SUPPORTED_DCP_SIZES}, got {dcp_size}")
    if head_dim != HEAD_DIM:
        raise ValueError(f"merge_lse head_dim must be {HEAD_DIM}, got {head_dim}")

    tensor_dtype = TENSOR_DTYPES[dtype]
    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num
    max_out = dcp_size * MERGE_MAX_HEADS * MAX_TOKENS * HEAD_DIM
    max_lse = dcp_size * MERGE_MAX_HEADS * MAX_TOKENS
    max_merged = MAX_TOKENS * MERGE_MAX_HEADS * HEAD_DIM
    finite_hi = 1.0e30
    finite_lo = -1.0e30
    lse_align = 64
    mask_bytes = 32

    @T.prim_func
    def merge_lse_kernel(
        output_recv: T.Tensor((1, max_out), tensor_dtype),
        lse_recv: T.Tensor((1, max_lse), "float32"),
        merged: T.Tensor((1, max_merged), tensor_dtype),
        num_tokens: T.int32,
        num_heads: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total = num_tokens * num_heads
            per_task = (total + task_num - 1) // task_num
            row_start = task_id * per_task
            rows_left = T.if_then_else(total > row_start, total - row_start, 0)
            row_count = T.if_then_else(rows_left < per_task, rows_left, per_task)

            with T.Scope("V"):
                half0_ub = T.alloc_ub((HEAD_DIM,), tensor_dtype)
                half1_ub = T.alloc_ub((HEAD_DIM,), tensor_dtype)
                store_ub = T.alloc_ub((HEAD_DIM,), tensor_dtype)
                fp_ub = T.alloc_ub((HEAD_DIM,), "float32")
                acc_ub = T.alloc_ub((HEAD_DIM,), "float32")
                lse0_ub = T.alloc_ub((lse_align,), "float32")
                lse1_ub = T.alloc_ub((lse_align,), "float32")
                lse2_ub = T.alloc_ub((lse_align,), "float32")
                lse3_ub = T.alloc_ub((lse_align,), "float32")
                lse4_ub = T.alloc_ub((lse_align,), "float32")
                lse5_ub = T.alloc_ub((lse_align,), "float32")
                lse6_ub = T.alloc_ub((lse_align,), "float32")
                lse7_ub = T.alloc_ub((lse_align,), "float32")
                w0_ub = T.alloc_ub((lse_align,), "float32")
                w1_ub = T.alloc_ub((lse_align,), "float32")
                w2_ub = T.alloc_ub((lse_align,), "float32")
                w3_ub = T.alloc_ub((lse_align,), "float32")
                w4_ub = T.alloc_ub((lse_align,), "float32")
                w5_ub = T.alloc_ub((lse_align,), "float32")
                w6_ub = T.alloc_ub((lse_align,), "float32")
                w7_ub = T.alloc_ub((lse_align,), "float32")
                max_ub = T.alloc_ub((lse_align,), "float32")
                sum_ub = T.alloc_ub((lse_align,), "float32")
                inv_ub = T.alloc_ub((lse_align,), "float32")
                mask_ub = T.alloc_ub((mask_bytes,), "uint8")

                v_notify_mte2(MERGE_EVT_LSE)
                mte2_wait_v(MERGE_EVT_LSE)
                v_notify_mte2(MERGE_EVT_SRC0)
                mte2_wait_v(MERGE_EVT_SRC0)
                v_notify_mte2(MERGE_EVT_SRC1)
                mte2_wait_v(MERGE_EVT_SRC1)
                for local_row in T.serial(row_count):
                    flat = row_start + local_row
                    token = flat // num_heads
                    head = flat - token * num_heads
                    T.tile.fill(lse0_ub, finite_lo)
                    if dcp_size > 1:
                        T.tile.fill(lse1_ub, finite_lo)
                    if dcp_size > 2:
                        T.tile.fill(lse2_ub, finite_lo)
                        T.tile.fill(lse3_ub, finite_lo)
                    if dcp_size > 4:
                        T.tile.fill(lse4_ub, finite_lo)
                        T.tile.fill(lse5_ub, finite_lo)
                        T.tile.fill(lse6_ub, finite_lo)
                        T.tile.fill(lse7_ub, finite_lo)
                    v_notify_mte2(MERGE_EVT_LSE)
                    mte2_wait_v(MERGE_EVT_LSE)
                    T.copy(
                        lse_recv[
                            0,
                            (0 * num_heads + head) * num_tokens + token : (0 * num_heads + head) * num_tokens
                            + token
                            + 1,
                        ],
                        lse0_ub[0:1],
                    )
                    if dcp_size > 1:
                        T.copy(
                            lse_recv[
                                0,
                                (1 * num_heads + head) * num_tokens + token : (1 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse1_ub[0:1],
                        )
                    if dcp_size > 2:
                        T.copy(
                            lse_recv[
                                0,
                                (2 * num_heads + head) * num_tokens + token : (2 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse2_ub[0:1],
                        )
                        T.copy(
                            lse_recv[
                                0,
                                (3 * num_heads + head) * num_tokens + token : (3 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse3_ub[0:1],
                        )
                    if dcp_size > 4:
                        T.copy(
                            lse_recv[
                                0,
                                (4 * num_heads + head) * num_tokens + token : (4 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse4_ub[0:1],
                        )
                        T.copy(
                            lse_recv[
                                0,
                                (5 * num_heads + head) * num_tokens + token : (5 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse5_ub[0:1],
                        )
                        T.copy(
                            lse_recv[
                                0,
                                (6 * num_heads + head) * num_tokens + token : (6 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse6_ub[0:1],
                        )
                        T.copy(
                            lse_recv[
                                0,
                                (7 * num_heads + head) * num_tokens + token : (7 * num_heads + head) * num_tokens
                                + token
                                + 1,
                            ],
                            lse7_ub[0:1],
                        )
                    T.copy(
                        output_recv[
                            0,
                            ((0 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                        ],
                        half0_ub,
                    )
                    mte2_notify_v(MERGE_EVT_LSE)
                    mte2_notify_v(MERGE_EVT_SRC0)
                    v_wait_mte2(MERGE_EVT_LSE)
                    T.tile.compare(mask_ub, lse0_ub, finite_hi, "LT")
                    T.tile.select(
                        lse0_ub,
                        mask_ub,
                        lse0_ub,
                        T.float32(finite_lo),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.tile.compare(mask_ub, lse0_ub, finite_lo, "GT")
                    T.tile.select(
                        lse0_ub,
                        mask_ub,
                        lse0_ub,
                        T.float32(finite_lo),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    if dcp_size > 1:
                        T.tile.compare(mask_ub, lse1_ub, finite_hi, "LT")
                        T.tile.select(
                            lse1_ub,
                            mask_ub,
                            lse1_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse1_ub, finite_lo, "GT")
                        T.tile.select(
                            lse1_ub,
                            mask_ub,
                            lse1_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                    if dcp_size > 2:
                        T.tile.compare(mask_ub, lse2_ub, finite_hi, "LT")
                        T.tile.select(
                            lse2_ub,
                            mask_ub,
                            lse2_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse2_ub, finite_lo, "GT")
                        T.tile.select(
                            lse2_ub,
                            mask_ub,
                            lse2_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse3_ub, finite_hi, "LT")
                        T.tile.select(
                            lse3_ub,
                            mask_ub,
                            lse3_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse3_ub, finite_lo, "GT")
                        T.tile.select(
                            lse3_ub,
                            mask_ub,
                            lse3_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                    if dcp_size > 4:
                        T.tile.compare(mask_ub, lse4_ub, finite_hi, "LT")
                        T.tile.select(
                            lse4_ub,
                            mask_ub,
                            lse4_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse4_ub, finite_lo, "GT")
                        T.tile.select(
                            lse4_ub,
                            mask_ub,
                            lse4_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse5_ub, finite_hi, "LT")
                        T.tile.select(
                            lse5_ub,
                            mask_ub,
                            lse5_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse5_ub, finite_lo, "GT")
                        T.tile.select(
                            lse5_ub,
                            mask_ub,
                            lse5_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse6_ub, finite_hi, "LT")
                        T.tile.select(
                            lse6_ub,
                            mask_ub,
                            lse6_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse6_ub, finite_lo, "GT")
                        T.tile.select(
                            lse6_ub,
                            mask_ub,
                            lse6_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse7_ub, finite_hi, "LT")
                        T.tile.select(
                            lse7_ub,
                            mask_ub,
                            lse7_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                        T.tile.compare(mask_ub, lse7_ub, finite_lo, "GT")
                        T.tile.select(
                            lse7_ub,
                            mask_ub,
                            lse7_ub,
                            T.float32(finite_lo),
                            "VSEL_TENSOR_SCALAR_MODE",
                        )
                    T.copy(lse0_ub, max_ub)
                    if dcp_size > 1:
                        T.tile.max(max_ub, max_ub, lse1_ub)
                    if dcp_size > 2:
                        T.tile.max(max_ub, max_ub, lse2_ub)
                        T.tile.max(max_ub, max_ub, lse3_ub)
                    if dcp_size > 4:
                        T.tile.max(max_ub, max_ub, lse4_ub)
                        T.tile.max(max_ub, max_ub, lse5_ub)
                        T.tile.max(max_ub, max_ub, lse6_ub)
                        T.tile.max(max_ub, max_ub, lse7_ub)
                    T.tile.sub(w0_ub, lse0_ub, max_ub[0])
                    T.tile.exp(w0_ub, w0_ub)
                    T.copy(w0_ub, sum_ub)
                    if dcp_size > 1:
                        T.tile.sub(w1_ub, lse1_ub, max_ub[0])
                        T.tile.exp(w1_ub, w1_ub)
                        T.tile.add(sum_ub, sum_ub, w1_ub)
                    if dcp_size > 2:
                        T.tile.sub(w2_ub, lse2_ub, max_ub[0])
                        T.tile.exp(w2_ub, w2_ub)
                        T.tile.add(sum_ub, sum_ub, w2_ub)
                        T.tile.sub(w3_ub, lse3_ub, max_ub[0])
                        T.tile.exp(w3_ub, w3_ub)
                        T.tile.add(sum_ub, sum_ub, w3_ub)
                    if dcp_size > 4:
                        T.tile.sub(w4_ub, lse4_ub, max_ub[0])
                        T.tile.exp(w4_ub, w4_ub)
                        T.tile.add(sum_ub, sum_ub, w4_ub)
                        T.tile.sub(w5_ub, lse5_ub, max_ub[0])
                        T.tile.exp(w5_ub, w5_ub)
                        T.tile.add(sum_ub, sum_ub, w5_ub)
                        T.tile.sub(w6_ub, lse6_ub, max_ub[0])
                        T.tile.exp(w6_ub, w6_ub)
                        T.tile.add(sum_ub, sum_ub, w6_ub)
                        T.tile.sub(w7_ub, lse7_ub, max_ub[0])
                        T.tile.exp(w7_ub, w7_ub)
                        T.tile.add(sum_ub, sum_ub, w7_ub)
                    T.tile.add(sum_ub, sum_ub, 1.0e-20)
                    T.tile.reciprocal(inv_ub, sum_ub)
                    T.tile.mul(w0_ub, w0_ub, inv_ub[0])
                    if dcp_size > 1:
                        T.tile.mul(w1_ub, w1_ub, inv_ub[0])
                    if dcp_size > 2:
                        T.tile.mul(w2_ub, w2_ub, inv_ub[0])
                        T.tile.mul(w3_ub, w3_ub, inv_ub[0])
                    if dcp_size > 4:
                        T.tile.mul(w4_ub, w4_ub, inv_ub[0])
                        T.tile.mul(w5_ub, w5_ub, inv_ub[0])
                        T.tile.mul(w6_ub, w6_ub, inv_ub[0])
                        T.tile.mul(w7_ub, w7_ub, inv_ub[0])
                    v_wait_mte2(MERGE_EVT_SRC0)
                    if dcp_size > 1:
                        T.copy(
                            output_recv[
                                0,
                                ((1 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half1_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC1)
                    T.tile.cast(acc_ub, half0_ub, "CAST_NONE", HEAD_DIM)
                    T.tile.mul(acc_ub, acc_ub, w0_ub[0])
                    if dcp_size == 2:
                        v_wait_mte2(MERGE_EVT_SRC1)
                        T.tile.cast(fp_ub, half1_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w1_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                    if dcp_size > 2:
                        v_notify_mte2(MERGE_EVT_SRC0)
                        mte2_wait_v(MERGE_EVT_SRC0)
                        v_wait_mte2(MERGE_EVT_SRC1)
                        T.copy(
                            output_recv[
                                0,
                                ((2 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half0_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC0)
                        T.tile.cast(fp_ub, half1_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w1_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                        v_notify_mte2(MERGE_EVT_SRC1)
                        mte2_wait_v(MERGE_EVT_SRC1)
                        v_wait_mte2(MERGE_EVT_SRC0)
                        T.copy(
                            output_recv[
                                0,
                                ((3 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half1_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC1)
                        T.tile.cast(fp_ub, half0_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w2_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                    if dcp_size == 4:
                        v_wait_mte2(MERGE_EVT_SRC1)
                        T.tile.cast(fp_ub, half1_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w3_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                    if dcp_size > 4:
                        v_notify_mte2(MERGE_EVT_SRC0)
                        mte2_wait_v(MERGE_EVT_SRC0)
                        v_wait_mte2(MERGE_EVT_SRC1)
                        T.copy(
                            output_recv[
                                0,
                                ((4 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half0_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC0)
                        T.tile.cast(fp_ub, half1_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w3_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                        v_notify_mte2(MERGE_EVT_SRC1)
                        mte2_wait_v(MERGE_EVT_SRC1)
                        v_wait_mte2(MERGE_EVT_SRC0)
                        T.copy(
                            output_recv[
                                0,
                                ((5 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half1_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC1)
                        T.tile.cast(fp_ub, half0_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w4_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                        v_notify_mte2(MERGE_EVT_SRC0)
                        mte2_wait_v(MERGE_EVT_SRC0)
                        v_wait_mte2(MERGE_EVT_SRC1)
                        T.copy(
                            output_recv[
                                0,
                                ((6 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half0_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC0)
                        T.tile.cast(fp_ub, half1_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w5_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                        v_notify_mte2(MERGE_EVT_SRC1)
                        mte2_wait_v(MERGE_EVT_SRC1)
                        v_wait_mte2(MERGE_EVT_SRC0)
                        T.copy(
                            output_recv[
                                0,
                                ((7 * num_heads + head) * num_tokens + token) * HEAD_DIM,
                            ],
                            half1_ub,
                        )
                        mte2_notify_v(MERGE_EVT_SRC1)
                        T.tile.cast(fp_ub, half0_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w6_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                        v_wait_mte2(MERGE_EVT_SRC1)
                        T.tile.cast(fp_ub, half1_ub, "CAST_NONE", HEAD_DIM)
                        T.tile.mul(fp_ub, fp_ub, w7_ub[0])
                        T.tile.add(acc_ub, acc_ub, fp_ub)
                    if dcp_size > 1:
                        v_notify_mte2(MERGE_EVT_SRC1)
                        mte2_wait_v(MERGE_EVT_SRC1)
                    v_notify_mte2(MERGE_EVT_SRC0)
                    mte2_wait_v(MERGE_EVT_SRC0)
                    T.tile.cast(store_ub, acc_ub, "CAST_RINT", HEAD_DIM)
                    v_notify_mte3(MERGE_EVT_STORE)
                    mte3_wait_v(MERGE_EVT_STORE)
                    dst_off = (token * num_heads + head) * HEAD_DIM
                    T.copy(store_ub, merged[0, dst_off])
                    mte3_notify_v(MERGE_EVT_STORE)
                    v_wait_mte3(MERGE_EVT_STORE)

    return merge_lse_kernel


@tilelang.jit(pass_configs=FUSION_PASS_CONFIGS)
def pack_query_kernel_jit(dtype: str, vec_core_num: int):
    return build_pack_query_kernel(dtype=dtype, vec_core_num=vec_core_num)


@tilelang.jit(pass_configs=FUSION_PASS_CONFIGS)
def unpack_query_kernel_jit(dtype: str, vec_core_num: int):
    return build_unpack_query_kernel(dtype=dtype, vec_core_num=vec_core_num)


@tilelang.jit(pass_configs=REMAP_PASS_CONFIGS)
def remap_topk_kernel_jit(
    vec_core_num: int,
    topk: int,
    physical_block_size: int,
    shard_size: int,
):
    return build_remap_topk_kernel(
        vec_core_num=vec_core_num,
        topk=topk,
        physical_block_size=physical_block_size,
        shard_size=shard_size,
    )


@tilelang.jit(pass_configs=FUSION_PASS_CONFIGS)
def pack_a2a_kernel_jit(dtype: str, vec_core_num: int):
    return build_pack_a2a_kernel(dtype=dtype, vec_core_num=vec_core_num)


@tilelang.jit(pass_configs=MERGE_PASS_CONFIGS)
def merge_lse_kernel_jit(dtype: str, vec_core_num: int, dcp_size: int):
    return build_merge_lse_kernel(
        dtype=dtype,
        vec_core_num=vec_core_num,
        dcp_size=dcp_size,
    )


def fused_pack_query_for_allgather(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
) -> torch.Tensor:
    _require_npu(ql_nope, q_pe)
    if (
        ql_nope.dim() != 3
        or q_pe.dim() != 3
        or ql_nope.shape[:-1] != q_pe.shape[:-1]
        or ql_nope.dtype != q_pe.dtype
        or ql_nope.shape[-1] != NOPE_DIM
        or q_pe.shape[-1] != PE_DIM
        or ql_nope.shape[0] <= 0
        or ql_nope.shape[1] <= 0
        or ql_nope.shape[0] > MAX_TOKENS
        or ql_nope.shape[1] > MAX_HEADS
    ):
        raise RuntimeError(
            "pack_query kernel expects TND ql_nope/q_pe with "
            f"D=({NOPE_DIM},{PE_DIM}), T<={MAX_TOKENS}, H<={MAX_HEADS}, "
            f"got {tuple(ql_nope.shape)} / {tuple(q_pe.shape)}"
        )

    dtype_name = _dtype_name(ql_nope.dtype)
    num_tokens = int(ql_nope.shape[0])
    num_heads = int(ql_nope.shape[1])
    send = torch.empty(
        (num_heads, num_tokens, FUSED_Q_DIM),
        dtype=ql_nope.dtype,
        device=ql_nope.device,
    )
    kernel = pack_query_kernel_jit(dtype_name, _launch_vec_core_num(num_tokens * num_heads))
    kernel(
        ql_nope.view(1, -1),
        q_pe.view(1, -1),
        send.view(1, -1),
        num_tokens,
        num_heads,
    )
    return send


def fused_unpack_query_after_allgather(
    gathered: torch.Tensor,
    nope_dim: int,
    pe_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_npu(gathered)
    if (
        gathered.dim() != 3
        or nope_dim != NOPE_DIM
        or pe_dim != PE_DIM
        or gathered.shape[-1] != FUSED_Q_DIM
        or gathered.shape[0] <= 0
        or gathered.shape[1] <= 0
        or gathered.shape[0] > MAX_HEADS
        or gathered.shape[1] > MAX_TOKENS
    ):
        raise RuntimeError(
            "unpack_query kernel expects gathered [H, T, "
            f"{FUSED_Q_DIM}] with H<={MAX_HEADS}, T<={MAX_TOKENS}, "
            f"got {tuple(gathered.shape)}, nope_dim={nope_dim}, pe_dim={pe_dim}"
        )

    dtype_name = _dtype_name(gathered.dtype)
    num_heads = int(gathered.shape[0])
    num_tokens = int(gathered.shape[1])
    ql_nope = torch.empty(
        (num_tokens, num_heads, NOPE_DIM),
        dtype=gathered.dtype,
        device=gathered.device,
    )
    q_pe = torch.empty(
        (num_tokens, num_heads, PE_DIM),
        dtype=gathered.dtype,
        device=gathered.device,
    )
    kernel = unpack_query_kernel_jit(dtype_name, _launch_vec_core_num(num_tokens * num_heads))
    kernel(
        gathered.view(1, -1),
        ql_nope.view(1, -1),
        q_pe.view(1, -1),
        num_tokens,
        num_heads,
    )
    return ql_nope, q_pe


def fused_remap_sparse_indices(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
    *,
    out: torch.Tensor | None = None,
    idx_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if layout.shard_size <= 1:
        return topk_indices
    _require_npu(topk_indices)
    if (
        topk_indices.dtype != torch.int32
        or topk_indices.dim() != 2
        or index_topk <= 0
        or index_topk > MAX_TOPK
        or topk_indices.shape[-1] != index_topk
        or topk_indices.shape[0] <= 0
        or topk_indices.shape[0] > MAX_TOKENS
    ):
        raise RuntimeError(
            "remap_topk kernel expects int32 [T, index_topk] with "
            f"T<={MAX_TOKENS}, index_topk<={MAX_TOPK}, "
            f"got {tuple(topk_indices.shape)} dtype={topk_indices.dtype} "
            f"index_topk={index_topk}"
        )

    num_tokens = int(topk_indices.shape[0])
    if out is None:
        out = torch.empty_like(topk_indices)
    else:
        _require_match(
            out,
            shape=tuple(topk_indices.shape),
            dtype=topk_indices.dtype,
            name="remap out",
        )
    scratch_n = remap_scratch_numel(num_tokens, index_topk)
    if idx_scratch is None:
        idx_scratch = torch.empty(
            scratch_n,
            dtype=torch.int32,
            device=topk_indices.device,
        )
    else:
        _require_npu(idx_scratch)
        if idx_scratch.dtype != torch.int32 or idx_scratch.numel() < scratch_n:
            raise RuntimeError(
                "remap idx_scratch expects int32 numel>="
                f"{scratch_n}, got dtype={idx_scratch.dtype} numel={idx_scratch.numel()}"
            )
        idx_scratch = idx_scratch.reshape(-1)[:scratch_n]
    kernel = remap_topk_kernel_jit(
        _launch_vec_core_num(num_tokens),
        index_topk,
        int(layout.physical_block_size),
        int(layout.shard_size),
    )
    kernel(
        topk_indices.view(-1),
        out.view(-1),
        idx_scratch,
        idx_scratch.view(torch.uint32),
        num_tokens,
        int(layout.shard_rank),
    )
    return out


def fused_pack_a2a_payloads(
    sfa_output: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_npu(sfa_output, softmax_max, softmax_sum)
    if (
        sfa_output.dim() != 3
        or sfa_output.shape[-1] != HEAD_DIM
        or softmax_max.shape != softmax_sum.shape
        or softmax_max.dim() != 3
        or softmax_max.dtype != torch.float32
        or softmax_sum.dtype != torch.float32
        or sfa_output.shape[0] <= 0
        or sfa_output.shape[1] <= 0
        or sfa_output.shape[0] > MAX_TOKENS
        or sfa_output.shape[1] > MAX_HEADS
        or softmax_max.shape[0] * softmax_max.shape[2] != sfa_output.shape[1]
        or softmax_max.shape[1] != sfa_output.shape[0]
    ):
        raise RuntimeError(
            "pack_a2a kernel expects sfa_output [T, H, "
            f"{HEAD_DIM}] and softmax_max/sum [N2, T, G] with N2*G=H, "
            f"got out={tuple(sfa_output.shape)} max={tuple(softmax_max.shape)} "
            f"sum={tuple(softmax_sum.shape)}"
        )

    dtype_name = _dtype_name(sfa_output.dtype)
    num_tokens = int(sfa_output.shape[0])
    num_heads = int(sfa_output.shape[1])
    n2_size = int(softmax_max.shape[0])
    g_size = int(softmax_max.shape[2])
    out_send = torch.empty(
        (num_heads, num_tokens, HEAD_DIM),
        dtype=sfa_output.dtype,
        device=sfa_output.device,
    )
    lse_send = torch.empty(
        (num_heads, num_tokens, 1),
        dtype=torch.float32,
        device=sfa_output.device,
    )
    kernel = pack_a2a_kernel_jit(dtype_name, _launch_vec_core_num(num_tokens * num_heads))
    kernel(
        sfa_output.view(1, -1),
        softmax_max.view(1, -1),
        softmax_sum.view(1, -1),
        out_send.view(1, -1),
        lse_send.view(1, -1),
        num_tokens,
        num_heads,
        n2_size,
        g_size,
    )
    return out_send, lse_send


def fused_merge_dcp_outputs(
    output_recv: torch.Tensor,
    lse_recv: torch.Tensor,
) -> torch.Tensor:
    _require_npu(output_recv, lse_recv)
    dcp_size = int(output_recv.shape[0]) if output_recv.dim() == 4 else -1
    if (
        output_recv.dim() != 4
        or lse_recv.dim() != 3
        or output_recv.shape[:3] != lse_recv.shape
        or output_recv.shape[-1] != HEAD_DIM
        or dcp_size not in SUPPORTED_DCP_SIZES
        or lse_recv.dtype != torch.float32
        or output_recv.shape[1] <= 0
        or output_recv.shape[2] <= 0
        or output_recv.shape[1] > MERGE_MAX_HEADS
        or output_recv.shape[2] > MAX_TOKENS
    ):
        raise RuntimeError(
            "merge_lse kernel expects output_recv [dcp, H, T, "
            f"{HEAD_DIM}] and lse_recv [dcp, H, T] with dcp in "
            f"{SUPPORTED_DCP_SIZES}, H<={MERGE_MAX_HEADS}, T<={MAX_TOKENS}, "
            f"got out={tuple(output_recv.shape)} lse={tuple(lse_recv.shape)}"
        )

    dtype_name = _dtype_name(output_recv.dtype)
    num_heads = int(output_recv.shape[1])
    num_tokens = int(output_recv.shape[2])
    merged = torch.empty(
        (num_tokens, num_heads, HEAD_DIM),
        dtype=output_recv.dtype,
        device=output_recv.device,
    )
    kernel = merge_lse_kernel_jit(dtype_name, _launch_vec_core_num(num_tokens * num_heads), dcp_size)
    kernel(
        output_recv.view(1, -1),
        lse_recv.view(1, -1),
        merged.view(1, -1),
        num_tokens,
        num_heads,
    )
    return merged
