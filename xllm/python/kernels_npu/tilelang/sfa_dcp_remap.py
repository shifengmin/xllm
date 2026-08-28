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
"""TileLang-Ascend DCP sparse-index remap kernel.

Production decode and ACL graph use the AOT wrapper ``sfa_dcp_remap``.
``@tilelang.jit`` helpers here are only for local Python precision checks.
"""

import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

from xllm.python.kernels_npu.tilelang.utils import (
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
MAX_TOPK = 2048
REMAP_VEC_LEN = 64
REMAP_COPY_LEN = 32
REMAP_EVT_SRC0 = 0
REMAP_EVT_IDX = 2
REMAP_EVT_OUT = 3
REMAP_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
    "tl.disable_safe_memory_legalize": True,
}
DEFAULT_REMAP_TOPK = 2048
SUPPORTED_REMAP_TOPK = (DEFAULT_REMAP_TOPK,)
SUPPORTED_SHARD_SIZES = (2, 4, 8, 16)
DEFAULT_PHYSICAL_BLOCK_SIZE = 128


def _require_npu(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.device.type != "npu":
            raise RuntimeError(f"sfa_dcp_remap requires NPU tensors, got device={tensor.device}")
        if not tensor.is_contiguous():
            raise RuntimeError(f"sfa_dcp_remap requires contiguous tensors, got shape={tuple(tensor.shape)}")


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
    if shard_size not in SUPPORTED_SHARD_SIZES:
        raise ValueError(f"remap_topk shard_size must be one of {SUPPORTED_SHARD_SIZES}, got {shard_size}")
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


def fused_remap_sparse_indices(
    topk_indices: torch.Tensor,
    physical_block_size: int,
    shard_size: int,
    shard_rank: int,
    *,
    out: torch.Tensor | None = None,
    idx_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    if shard_size <= 1:
        return topk_indices
    if int(shard_size) not in SUPPORTED_SHARD_SIZES:
        raise RuntimeError(f"remap_topk kernel expects shard_size in {SUPPORTED_SHARD_SIZES}, got {shard_size}")
    if shard_rank < 0 or shard_rank >= shard_size:
        raise RuntimeError(f"remap_topk kernel expects 0 <= shard_rank < shard_size, got shard_rank={shard_rank}")
    _require_npu(topk_indices)
    if topk_indices.dtype != torch.int32 or topk_indices.dim() != 2:
        raise RuntimeError(
            "remap_topk kernel expects int32 [T, index_topk] with "
            f"T<={MAX_TOKENS}, index_topk<={MAX_TOPK}, "
            f"got {tuple(topk_indices.shape)} dtype={topk_indices.dtype}"
        )
    num_tokens = int(topk_indices.shape[0])
    index_topk = int(topk_indices.shape[-1])
    if num_tokens <= 0 or num_tokens > MAX_TOKENS or index_topk <= 0 or index_topk > MAX_TOPK:
        raise RuntimeError(
            "remap_topk kernel expects int32 [T, index_topk] with "
            f"T<={MAX_TOKENS}, index_topk<={MAX_TOPK}, "
            f"got {tuple(topk_indices.shape)} dtype={topk_indices.dtype}"
        )

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
        int(physical_block_size),
        int(shard_size),
    )
    kernel(
        topk_indices.view(-1),
        out.view(-1),
        idx_scratch,
        idx_scratch.view(torch.uint32),
        num_tokens,
        int(shard_rank),
    )
    return out
