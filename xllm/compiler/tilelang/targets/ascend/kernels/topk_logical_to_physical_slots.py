#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.

import tilelang
import tilelang.language as T

from ....common.spec import DispatchField, TilelangKernel, register_kernel

AIC_CORE_NUM = 16
BLOCK_SIZE_SPECIALIZATIONS = (16, 32, 64, 128, 256)

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


def build_topk_logical_to_physical_slots_kernel(block_size: int):
    if block_size <= 0:
        raise ValueError(f"block_size({block_size}) must be > 0")

    topk_numel = T.symbolic("topk_numel")
    block_table_rows = T.symbolic("block_table_rows")
    block_table_cols = T.symbolic("block_table_cols")
    packed_count = T.symbolic("packed_count")

    @T.prim_func
    def topk_logical_to_physical_slots(
        topk_positions: T.Tensor((topk_numel,), "int32"),
        block_tables: T.Tensor(
            (block_table_rows, block_table_cols), "int32"
        ),
        packed_gather_indices: T.Tensor((packed_count,), "int32"),
        packed_query_block_rows: T.Tensor((packed_count,), "int32"),
        physical_slots: T.Tensor((packed_count,), "int32"),
    ):
        with T.Kernel(AIC_CORE_NUM, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                physical_slot_ub = T.alloc_ub((1,), "int32")
                vector_task_count = AIC_CORE_NUM * 2
                task_id = cid * 2 + vid
                work_per_task = (
                    packed_count + vector_task_count - 1
                ) // vector_task_count
                entry_start = task_id * work_per_task
                entries_left = T.if_then_else(
                    packed_count > entry_start,
                    packed_count - entry_start,
                    0,
                )
                entry_count = T.if_then_else(
                    entries_left < work_per_task,
                    entries_left,
                    work_per_task,
                )

                for entry_local in T.serial(entry_count):
                    entry_idx = entry_start + entry_local
                    topk_idx = packed_gather_indices[entry_idx]
                    query_row = packed_query_block_rows[entry_idx]
                    logical_pos = topk_positions[topk_idx]
                    logical_block = logical_pos // block_size
                    block_offset = logical_pos % block_size
                    physical_block = block_tables[query_row, logical_block]
                    physical_slot_ub[0] = (
                        physical_block * block_size + block_offset
                    )
                    T.copy(physical_slot_ub, physical_slots[entry_idx])

    return topk_logical_to_physical_slots


@register_kernel
class TopkLogicalToPhysicalSlotsKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("block_size", "int32")]
    SPECIALIZATIONS = [
        {
            "variant_key": f"bs{block_size}",
            "block_size": block_size,
        }
        for block_size in BLOCK_SIZE_SPECIALIZATIONS
    ]

    @staticmethod
    def generate_source(block_size: int) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_topk_logical_to_physical_slots_kernel(
            block_size=block_size
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
