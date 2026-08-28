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

import tilelang

from xllm.python.kernels_npu.tilelang import (
    sfa_dcp_remap as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.sfa_dcp_remap import (
    DEFAULT_PHYSICAL_BLOCK_SIZE,
    DEFAULT_REMAP_TOPK,
    REMAP_PASS_CONFIGS,
    SUPPORTED_REMAP_TOPK,
    SUPPORTED_SHARD_SIZES,
    build_remap_topk_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class SfaDcpRemapKernel(TilelangKernel):
    KERNEL_NAME = "sfa_dcp_remap"
    DISPATCH_SCHEMA = [
        DispatchField("topk", "int32"),
        DispatchField("physical_block_size", "int32"),
        DispatchField("shard_size", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"tk{topk}_pb{DEFAULT_PHYSICAL_BLOCK_SIZE}_ss{shard_size}",
            "topk": topk,
            "physical_block_size": DEFAULT_PHYSICAL_BLOCK_SIZE,
            "shard_size": shard_size,
        }
        for topk in SUPPORTED_REMAP_TOPK
        for shard_size in SUPPORTED_SHARD_SIZES
    ]

    @staticmethod
    def generate_source(topk: int, physical_block_size: int, shard_size: int) -> str:
        if topk not in SUPPORTED_REMAP_TOPK:
            raise ValueError(f"sfa_dcp_remap only supports topk={DEFAULT_REMAP_TOPK}, got {topk}")
        if physical_block_size != DEFAULT_PHYSICAL_BLOCK_SIZE:
            raise ValueError(
                "sfa_dcp_remap only supports physical_block_size="
                f"{DEFAULT_PHYSICAL_BLOCK_SIZE}, got {physical_block_size}"
            )
        if shard_size not in SUPPORTED_SHARD_SIZES:
            raise ValueError(f"sfa_dcp_remap only supports shard_size in {SUPPORTED_SHARD_SIZES}, got {shard_size}")
        tilelang.disable_cache()
        tilelang_kernel = build_remap_topk_kernel(
            vec_core_num=detect_vec_core_num(),
            topk=topk,
            physical_block_size=physical_block_size,
            shard_size=shard_size,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=REMAP_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
