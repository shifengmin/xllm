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
    sfa_dcp_fusion as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.sfa_dcp_fusion import (
    DEFAULT_MERGE_DTYPE,
    MERGE_PASS_CONFIGS,
    SUPPORTED_DCP_SIZES,
    build_merge_lse_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class SfaDcpMergeKernel(TilelangKernel):
    KERNEL_NAME = "sfa_dcp_merge"
    DISPATCH_SCHEMA = [
        DispatchField("dcp_size", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"dcp{dcp_size}_{DEFAULT_MERGE_DTYPE}",
            "dcp_size": dcp_size,
            "dtype": DEFAULT_MERGE_DTYPE,
        }
        for dcp_size in SUPPORTED_DCP_SIZES
    ]

    @staticmethod
    def generate_source(dcp_size: int, dtype: str) -> str:
        if dcp_size not in SUPPORTED_DCP_SIZES:
            raise ValueError(f"sfa_dcp_merge only supports dcp_size in {SUPPORTED_DCP_SIZES}, got {dcp_size}")
        if dtype != DEFAULT_MERGE_DTYPE:
            raise ValueError(f"sfa_dcp_merge only supports dtype={DEFAULT_MERGE_DTYPE}, got {dtype}")
        tilelang.disable_cache()
        tilelang_kernel = build_merge_lse_kernel(
            dtype=dtype,
            vec_core_num=detect_vec_core_num(),
            dcp_size=dcp_size,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=MERGE_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
