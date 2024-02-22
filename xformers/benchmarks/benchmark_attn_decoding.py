# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import Any, Dict, Type

import torch
from torch.utils import benchmark
from utils import benchmark_main_helper2

import xformers.ops as xops

min_run_time = 0.5
device = torch.device("cuda")


CASES = [
    dict(B=max(1, 2 ** (16 - i)), Mq=1, Mkv=2**i, Hq=16, Hkv=hkv, K=128)
    for i in range(8, 18)
    for hkv in (1, 2)
]


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
        ).expand(-1, -1, -1, Hq // Hkv, -1)
        self.v = torch.randn(
            [B, Mkv, Hkv, 1, K], device="cuda", dtype=dtype, requires_grad=bw
        ).expand(-1, -1, -1, Hq // Hkv, -1)

        if Hq == Hkv:
            self.q = self.q[:, :, :, 0]
            self.k = self.k[:, :, :, 0]
            self.v = self.v[:, :, :, 0]
        if Hkv == 1:
            self.q = self.q[:, :, 0]
            self.k = self.k[:, :, 0]
            self.v = self.v[:, :, 0]

    def fw(self) -> None:
        try:
            xops.memory_efficient_attention_forward(self.q, self.k, self.v, op=self.OP)
        except (RuntimeError, ValueError) as e:
            print(f"Runtime error: {e}")


class AttentionDecodingCK(AttentionDecodingFlashDecoding):
    OP = xops.fmha.ck.FwOp


class AttentionDecodingCKDecoder(AttentionDecodingFlashDecoding):
    OP = xops.fmha.ck_decoder.FwOp


class AttentionDecodingSplitKV(AttentionDecodingFlashDecoding):
    OP = xops.fmha.triton_splitk.FwOp


class AttentionDecodingCKSplitKV(AttentionDecodingFlashDecoding):
    OP = xops.fmha.ck_splitk.FwOp


class AttentionDecodingPyTorchRepeat(AttentionDecodingFlashDecoding):
    def fw(self) -> None:
        B, Mq, Mkv, Hq, Hkv, K = self.shapes
        scale = 1 / K**0.5
        q = self.q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
        k = self.k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        v = self.v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-1, -2)).softmax(-1) * scale
        return attn @ v


BENCHMARKS: Dict[str, Type[AttentionDecodingFlashDecoding]] = {
    "pytorch": AttentionDecodingPyTorchRepeat,
}

if torch.version.cuda:
    BENCHMARKS["flash-decoding"] = AttentionDecodingFlashDecoding

if torch.version.hip:
    BENCHMARKS.update(
        {
            "ck": AttentionDecodingCK,
            "ck-decoder": AttentionDecodingCKDecoder,
            "ck_splitK": AttentionDecodingCKSplitKV,
        }
    )


if (sys.version_info.major, sys.version_info.minor) >= (3, 9):
    BENCHMARKS["triton_splitK"] = AttentionDecodingSplitKV

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
