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

"""Precision checks for DCP SFA fusion kernels vs naive torch goldens."""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp_ref import (
    merge_dcp_outputs,
    pack_query_for_allgather,
    remap_sparse_indices,
    remap_sparse_indices_compact,
    unpack_query_after_allgather,
)

BF16_RTOL = 1.56e-2
BF16_ATOL = 9.77e-4


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _require_npu() -> torch.device:
    if not _npu_available():
        pytest.skip("NPU is not available")
    return torch.device("npu")


def _make_logical_slots(
    num_tokens: int,
    topk: int,
    layout: KVShardLayout,
    *,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(0)
    logical_block = layout.logical_block_size
    num_blocks = 8
    slots = torch.randint(
        0,
        num_blocks * logical_block,
        (num_tokens, topk),
        device=device,
        dtype=torch.int32,
    )
    # Mix in invalid / unowned entries.
    mask = torch.rand((num_tokens, topk), device=device) < 0.25
    slots = torch.where(mask, torch.full_like(slots, KVShardLayout.INVALID_SLOT), slots)
    return slots


def test_remap_compact_matches_naive_sort() -> None:
    for dcp_size, dcp_rank in ((4, 1), (16, 7)):
        layout = KVShardLayout.from_dcp(physical_block_size=128, dcp_size=dcp_size, dcp_rank=dcp_rank)
        slots = _make_logical_slots(3, 2048, layout, device=torch.device("cpu"))
        naive = remap_sparse_indices(slots, layout, index_topk=2048)
        compact = remap_sparse_indices_compact(slots, layout, index_topk=2048)
        assert torch.equal(naive, compact)


def test_pack_unpack_naive_roundtrip() -> None:
    torch.manual_seed(1)
    ql_nope = torch.randn(2, 8, 512, dtype=torch.bfloat16)
    q_pe = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    send = pack_query_for_allgather(ql_nope, q_pe)
    out_nope, out_pe = unpack_query_after_allgather(send, 512, 64)
    assert torch.equal(out_nope, ql_nope)
    assert torch.equal(out_pe, q_pe)


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_fused_remap_matches_naive() -> None:
    from xllm.python.kernels_npu.tilelang.sfa_dcp_remap import fused_remap_sparse_indices

    device = _require_npu()
    layout = KVShardLayout.from_dcp(physical_block_size=128, dcp_size=4, dcp_rank=2)
    for num_tokens, topk in (
        (1, 2048),
        (2, 2048),
        (3, 2048),
        (5, 2048),
        (7, 2048),
        (49, 2048),
        (63, 2048),
        (64, 2048),
    ):
        slots = _make_logical_slots(num_tokens, topk, layout, device=device)
        naive = remap_sparse_indices(slots, layout, index_topk=topk)
        fused = fused_remap_sparse_indices(
            slots,
            layout.physical_block_size,
            layout.dcp_size,
            layout.dcp_rank,
        )
        torch.npu.synchronize()
        assert torch.equal(fused, naive), f"remap mismatch T={num_tokens} topk={topk}"


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_fused_remap_dcp16_matches_naive() -> None:
    from xllm.python.kernels_npu.tilelang.sfa_dcp_remap import fused_remap_sparse_indices

    device = _require_npu()
    layout = KVShardLayout.from_dcp(physical_block_size=128, dcp_size=16, dcp_rank=7)
    for num_tokens in (1, 8):
        slots = _make_logical_slots(num_tokens, 2048, layout, device=device)
        naive = remap_sparse_indices(slots, layout, index_topk=2048)
        fused = fused_remap_sparse_indices(
            slots,
            layout.physical_block_size,
            layout.dcp_size,
            layout.dcp_rank,
        )
        torch.npu.synchronize()
        assert torch.equal(fused, naive), f"remap dcp=16 mismatch T={num_tokens}"


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_attention_update_merge_matches_naive() -> None:
    from xllm.python.layers.sfa_dcp import merge_dcp_outputs_with_attention_update

    device = _require_npu()
    dcp_size = 4
    num_heads = 8
    for num_tokens in (1, 2, 3, 7, 8, 16, 48):
        torch.manual_seed(4 + num_tokens)
        output_recv = torch.randn(
            dcp_size,
            num_heads,
            num_tokens,
            512,
            device=device,
            dtype=torch.bfloat16,
        )
        lse_recv = torch.randn(
            dcp_size,
            num_heads,
            num_tokens,
            device=device,
            dtype=torch.float32,
        )
        lse_recv[0, 0, 0] = float("-inf")
        lse_recv[1, 0, 0] = float("inf")
        if num_tokens > 1:
            lse_recv[2, 1, -1] = float("-inf")
            lse_recv[3, 1, -1] = float("inf")
        naive = merge_dcp_outputs(output_recv, lse_recv)
        merged = torch.empty(
            (num_tokens, num_heads, 512),
            device=device,
            dtype=torch.bfloat16,
        )
        fused = merge_dcp_outputs_with_attention_update(output_recv, lse_recv, merged)
        torch.npu.synchronize()
        torch.testing.assert_close(
            fused,
            naive,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
            msg=f"attention_update merge mismatch T={num_tokens}",
        )


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_attention_update_merge_runtime_dcp_matches_naive() -> None:
    from xllm.python.layers.sfa_dcp import merge_dcp_outputs_with_attention_update

    device = _require_npu()
    num_heads = 8
    for dcp_size in (2, 8, 16):
        for num_tokens in (1, 8):
            torch.manual_seed(16 + dcp_size + num_tokens)
            output_recv = torch.randn(
                dcp_size,
                num_heads,
                num_tokens,
                512,
                device=device,
                dtype=torch.bfloat16,
            )
            lse_recv = torch.randn(
                dcp_size,
                num_heads,
                num_tokens,
                device=device,
                dtype=torch.float32,
            )
            lse_recv[0, 0, 0] = float("-inf")
            if dcp_size > 2:
                lse_recv[dcp_size - 1, 0, 0] = float("inf")
            naive = merge_dcp_outputs(output_recv, lse_recv)
            merged = torch.empty(
                (num_tokens, num_heads, 512),
                device=device,
                dtype=torch.bfloat16,
            )
            fused = merge_dcp_outputs_with_attention_update(output_recv, lse_recv, merged)
            torch.npu.synchronize()
            torch.testing.assert_close(
                fused,
                naive,
                rtol=BF16_RTOL,
                atol=BF16_ATOL,
                msg=f"attention_update merge dcp={dcp_size} mismatch T={num_tokens}",
            )


def _aot_ops_available() -> bool:
    ops = getattr(torch.ops, "xllm_ops", None)
    return ops is not None and hasattr(ops, "sfa_dcp_remap_out")


def _remap_scratch_numel(num_tokens: int, topk: int) -> int:
    return num_tokens * (64 if topk <= 64 else topk)


def _capture_replay(run_fn, mutate_fn):
    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        run_fn()
    torch.npu.synchronize()
    with torch.npu.stream(stream), torch.npu.graph(graph, stream=stream):
        captured = run_fn()
    torch.npu.synchronize()
    mutate_fn()
    with torch.npu.stream(stream):
        graph.replay()
    torch.npu.synchronize()
    return captured


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
@pytest.mark.skipif(not _aot_ops_available(), reason="sfa_dcp AOT ops are not registered")
def test_aot_remap_acl_graph_replay() -> None:
    device = _require_npu()
    layout = KVShardLayout.from_dcp(physical_block_size=128, dcp_size=4, dcp_rank=2)
    for num_tokens, topk in ((1, 2048), (8, 2048)):
        slots = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        out = torch.empty_like(slots)
        scratch = torch.empty(
            _remap_scratch_numel(num_tokens, topk),
            dtype=torch.int32,
            device=device,
        )
        out_ptr = out.data_ptr()
        slots.copy_(_make_logical_slots(num_tokens, topk, layout, device=device))

        def run(
            slots: torch.Tensor = slots,
            out: torch.Tensor = out,
            scratch: torch.Tensor = scratch,
        ) -> torch.Tensor:
            return torch.ops.xllm_ops.sfa_dcp_remap_out(
                slots,
                layout.physical_block_size,
                layout.dcp_size,
                layout.dcp_rank,
                out,
                scratch,
            )

        replay_slots = _make_logical_slots(num_tokens, topk, layout, device=device)

        def mutate(
            slots: torch.Tensor = slots,
            replay_slots: torch.Tensor = replay_slots,
        ) -> None:
            slots.copy_(replay_slots)

        graph_out = _capture_replay(run, mutate)
        naive = remap_sparse_indices(replay_slots, layout, index_topk=topk)
        assert graph_out.data_ptr() == out_ptr
        assert torch.equal(graph_out, naive), f"AOT remap graph mismatch T={num_tokens}"


@pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
def test_attention_update_merge_acl_graph_replay() -> None:
    from xllm.python.layers.sfa_dcp import merge_dcp_outputs_with_attention_update

    device = _require_npu()
    dcp_size = 4
    num_heads = 8
    for num_tokens in (1, 8):
        torch.manual_seed(20 + num_tokens)
        output_recv = torch.empty(
            dcp_size,
            num_heads,
            num_tokens,
            512,
            device=device,
            dtype=torch.bfloat16,
        )
        lse_recv = torch.empty(
            dcp_size,
            num_heads,
            num_tokens,
            device=device,
            dtype=torch.float32,
        )
        merged = torch.empty(
            (num_tokens, num_heads, 512),
            device=device,
            dtype=torch.bfloat16,
        )
        merged_ptr = merged.data_ptr()
        output_recv.copy_(torch.randn_like(output_recv))
        lse_recv.copy_(torch.randn_like(lse_recv))

        def run(
            output_recv: torch.Tensor = output_recv,
            lse_recv: torch.Tensor = lse_recv,
            merged: torch.Tensor = merged,
        ) -> torch.Tensor:
            return merge_dcp_outputs_with_attention_update(output_recv, lse_recv, merged)

        replay_out = torch.randn_like(output_recv)
        replay_lse = torch.randn_like(lse_recv)
        replay_lse[0, 0, 0] = float("-inf")

        def mutate(
            output_recv: torch.Tensor = output_recv,
            lse_recv: torch.Tensor = lse_recv,
            replay_out: torch.Tensor = replay_out,
            replay_lse: torch.Tensor = replay_lse,
        ) -> None:
            output_recv.copy_(replay_out)
            lse_recv.copy_(replay_lse)

        graph_out = _capture_replay(run, mutate)
        naive = merge_dcp_outputs(replay_out, replay_lse)
        assert graph_out.data_ptr() == merged_ptr
        torch.testing.assert_close(
            graph_out,
            naive,
            rtol=BF16_RTOL,
            atol=BF16_ATOL,
            msg=f"attention_update merge graph mismatch T={num_tokens}",
        )
