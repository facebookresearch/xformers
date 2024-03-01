# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2

import xformers.ops as xops

min_run_time = 0.5
device = torch.device("cuda")


CASES = [
    dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=1, K=128)
    for i in range(8, 18)
] + [
    dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=2, K=128)
    for i in range(8, 18)
]


def quantize_kv_int4(k: torch.Tensor, num_groups: int = 1) -> torch.Tensor:
    """
    Auxiliary int4 row quantization function used for benchmarking and tests.
    Matches the behaviour of torch.ops.llama_cpp.dequantize_int4_cache -
    quantization parameters (scale and offset) of each row along the last
    dimension of the tensor are assumed to be packed into two float16 values
    at the beginning of the row.
    """
    # Scale and shift are such that quantization linearly maps int4 values range [0..15]
    # to input values range min(k)..max(k) individually for every row
    k = k.reshape(*k.shape[:-1], num_groups, k.shape[-1] // num_groups)
    # print(f"k_reshape = {k.shape}")
    max_vals = torch.max(k, dim=-1, keepdim=True).values
    min_vals = torch.min(k, dim=-1, keepdim=True).values
    scale_k: torch.Tensor = (max_vals - min_vals) / 15
    # print(f"scale_k_shape = {scale_k.shape}")

    shift_k = torch.min(k, dim=-1, keepdim=True).values
    scale_k = scale_k.to(torch.float16)
    shift_k = shift_k.to(torch.float16)
    in_bytes = ((k - shift_k.expand(k.shape)) / scale_k.expand(k.shape)) + 0.5
    in_bytes = in_bytes.to(torch.uint8)
    in_int4 = in_bytes & 0xF
    in_int4_packed = in_int4[..., ::2] + (in_int4[..., 1::2] << 4)
    scale_shift = torch.concat(
        [scale_k.view(torch.uint8), shift_k.view(torch.uint8)], dim=-1
    )
    k_quant = torch.concat(
        [
            scale_shift.flatten(start_dim=-2),
            in_int4_packed.flatten(start_dim=-2),
        ],
        dim=-1,
    ).view(torch.int16)
    return k_quant


def _setup_test(
    functions, fw: bool = False, bw: bool = False, cuda_graph: bool = True, **kwargs
):
    for k, benchmark_cls in functions.items():
        benchmark_object = benchmark_cls(**kwargs, bw=bw)
        label = benchmark_object.label
        label += "fw" if fw else ""
        label += "bw" if bw else ""

        def run_one():
            if fw:
                benchmark_object.fw()
            if bw:
                benchmark_object.bw()

        if cuda_graph:
            run_one()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run_one()

            def run_one():
                g.replay()

        yield benchmark.Timer(
            stmt="fn()",
            globals={
                "fn": run_one,
            },
            label=label,
            description=k,
            sub_label=benchmark_object.sub_label,
        )


class AttentionDecodingFlashDecoding:
    OP: Any = xops.fmha.flash.FwOp

    def __init__(
        self, B: int, Mq: int, Mkv: int, Hq: int, Hkv: int, K: int, bw: bool
    ) -> None:
        dtype = torch.float16
        self.sub_label = (
            f"B={B} Mq={Mq} Mkv={Mkv} Hq={Hq} Hkv={Hkv} K={K} TotalBytes="
            f"{((B * Mkv * Hkv * K * 2) + (B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2}"
        )
        self.label = "attn_decoding"
        self.shapes = (B, Mq, Mkv, Hq, Hkv, K)

        assert Hkv <= Hq
        assert Hq % Hkv == 0
        self.q = torch.randn(
            [B, Mq, Hkv, Hq // Hkv, K], device="cuda", dtype=dtype, requires_grad=bw
        )
        self.k = torch.randn(
            [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=bw
        )
        self.v = torch.randn(
            [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=bw
        )

        # self.k = torch.randn(
        #     [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=bw
        # ).expand(-1, -1, -1, Hq // Hkv, -1)
        # self.v = torch.randn(
        #     [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=bw
        # ).expand(-1, -1, -1, Hq // Hkv, -1)

        if Hq == Hkv:
            self.q = self.q[:, :, :, 0]
            self.k = self.k[:, :, :, 0]
            self.v = self.v[:, :, :, 0]
        if Hkv == 1:
            self.q = self.q[:, :, 0]
            self.k = self.k[:, :, 0]
            self.v = self.v[:, :, 0]

    def fw(self) -> None:
        xops.memory_efficient_attention_forward(self.q, self.k, self.v, op=self.OP)


class AttentionDecodingSplitKV(AttentionDecodingFlashDecoding):
    OP = xops.fmha.triton_splitk.FwOp
    def __init__(self, B: int, Mq: int, Mkv: int, Hq: int, Hkv: int, K: int, bw: bool
    ) -> None:
        super(AttentionDecodingSplitKV, self).__init__(B, Mq, Mkv, Hq, Hkv, K, bw)
        if Hkv == 1:
            self.k = self.k.expand(-1, -1, Hq // Hkv, -1)
            self.v = self.v.expand(-1, -1, Hq // Hkv, -1)
        else:
            self.k = self.k.expand(-1, -1, -1, Hq // Hkv, -1)
            self.v = self.v.expand(-1, -1, -1, Hq // Hkv, -1)


class AttentionDecodingPyTorchRepeat(AttentionDecodingFlashDecoding):
    def __init__(self, B: int, Mq: int, Mkv: int, Hq: int, Hkv: int, K: int, bw: bool
    ) -> None:
        super(AttentionDecodingPyTorchRepeat, self).__init__(B, Mq, Mkv, Hq, Hkv, K, bw)
        if Hkv == 1:
            self.k = self.k.expand(-1, -1, Hq // Hkv, -1)
            self.v = self.v.expand(-1, -1, Hq // Hkv, -1)
        else:
            self.k = self.k.expand(-1, -1, -1, Hq // Hkv, -1)
            self.v = self.v.expand(-1, -1, -1, Hq // Hkv, -1)

    def fw(self) -> None:
        B, Mq, Mkv, Hq, Hkv, K = self.shapes
        scale = 1 / K**0.5
        q = self.q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
        k = self.k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        v = self.v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-1, -2)).softmax(-1) * scale
        return attn @ v


class AttentionDecodingSplitInt4KV(AttentionDecodingFlashDecoding):
    # OP = xops.fmha.triton_splitk.FwOp_S1
    OP = xops.fmha.triton_splitk.FwOp
    def __init__(self, B: int, Mq: int, Mkv: int, Hq: int, Hkv: int, K: int, bw: bool
    ) -> None:
        super(AttentionDecodingSplitInt4KV, self).__init__(B, Mq, Mkv, Hq, Hkv, K, bw)
        # quantize to int data type
        num_groups = 1
        self.k = (
            quantize_kv_int4(self.k, num_groups=num_groups)
            .contiguous()
            .view(torch.int32)
        )
        self.v = (
            quantize_kv_int4(self.v, num_groups=num_groups)
            .contiguous()
            .view(torch.int32)
        )
        # print(f"shape1, k = {self.k.shape}, v = {self.v.shape}")
        if Hkv == 1:
            self.k = self.k.expand(-1, -1, Hq // Hkv, -1)
            self.v = self.v.expand(-1, -1, Hq // Hkv, -1)
        else:
            self.k = self.k.expand(-1, -1, -1, Hq // Hkv, -1)
            self.v = self.v.expand(-1, -1, -1, Hq // Hkv, -1)
        # self.k = self.k.view(1, B * Mkv, Hq, -1)
        # self.v = self.v.view(1, B * Mkv, Hq, -1)
        # print(f"shape2, k = {self.k.shape}, v = {self.v.shape}")


BENCHMARKS = {
    "pytorch": AttentionDecodingPyTorchRepeat,
    "flash-decoding": AttentionDecodingFlashDecoding,
    "triton_splitK": AttentionDecodingSplitKV,
    "triton_int4KV" : AttentionDecodingSplitInt4KV,
}


try:
    import flash_attn

    class AttentionDecodingFlashAttention(AttentionDecodingFlashDecoding):
        def fw(self) -> None:
            q, k, v = self.q, self.k, self.v
            if q.ndim == 5:
                B, Mq, H1, H2, K = q.shape
                B, Mkv, H1, H2, K = k.shape
                q = q.reshape([B, Mq, H1 * H2, K])
                k = k[:, :, :, 0]
                v = v[:, :, :, 0]
            return flash_attn.flash_attn_func(q, k, v)

    BENCHMARKS[
        f"flash-attention@{flash_attn.__version__}"
    ] = AttentionDecodingFlashAttention
except ImportError:
    pass


benchmark_main_helper2(
    "attn_decoding",
    fw=True,
    cases=CASES,
    functions=BENCHMARKS,
    min_run_time=min_run_time,
)
