# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Run with --omit-baselines to skip slow baselines.
# See other CLI arguments in benchmark_main_helper in utils.py.

import sys
from typing import Any, Dict, Type

import pytest
import torch

import xformers.ops as xops
from xformers.attn_bias_utils import create_attn_bias
from xformers.benchmarks.utils import benchmark_main_helper2, NotSupportedInputError

min_run_time = 0.5
device = torch.device("cuda")


CASES = [
    dict(
        B=max(1, 2 ** (16 - i)),
        Mq=1,
        Mkv=2**i,
        Hq=16,
        Hkv=hkv,
        K=128,
        attn_bias_type=xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    for i in range(8, 18)
    for hkv in (1, 2)
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


class AttentionDecodingBase:
    OP: Any = None

    def __init__(
        self,
        B: int,
        Mq: int,
        Mkv: int,
        Hq: int,
        Hkv: int,
        K: int,
        bw: bool,
        attn_bias_type,
    ) -> None:
        dtype = torch.float16
        torch.manual_seed(10)
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

        self.attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=Hq,
            num_heads_groups=Hq // Hkv,
            q_len=Mq,
            kv_len=Mkv,
            dtype=dtype,
            device=device,
            requires_grad=False,
            fmt="BMHK",
            op=self.OP,
        )

        if isinstance(
            self.attn_bias,
            xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ):
            self.q = self.q.view(1, -1, *self.q.shape[2:])
            self.k = self.k.view(1, -1, *self.k.shape[2:])
            self.v = self.v.view(1, -1, *self.v.shape[2:])

        if hasattr(self.OP, "not_supported_reasons"):
            inp = xops.fmha.Inputs(
                query=self.q, key=self.k, value=self.v, attn_bias=self.attn_bias
            )
            not_supported_reasons = self.OP.not_supported_reasons(inp)
            if not_supported_reasons:
                raise NotSupportedInputError(not_supported_reasons)

    def get_inputs(self):
        inp = xops.fmha.Inputs(
            query=self.q, key=self.k, value=self.v, attn_bias=self.attn_bias
        )
        return inp

    def fw(self) -> None:
        try:
            xops.memory_efficient_attention_forward(
                self.q, self.k, self.v, op=self.OP, attn_bias=self.attn_bias
            )
        except (RuntimeError, ValueError) as e:
            print(f"Runtime error: {e}")


class AttentionDecodingCUTLASS(AttentionDecodingBase):
    OP = xops.fmha.cutlass.FwOp


class AttentionDecodingCK(AttentionDecodingBase):
    OP = xops.fmha.ck.FwOp

    def __init__(
        self,
        B: int,
        Mq: int,
        Mkv: int,
        Hq: int,
        Hkv: int,
        K: int,
        bw: bool,
        attn_bias_type,
    ) -> None:
        dtype = torch.float16
        torch.manual_seed(10)
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

        self.attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=Hq,
            num_heads_groups=Hq // Hkv,
            q_len=Mq,
            kv_len=Mkv,
            dtype=dtype,
            device=device,
            requires_grad=False,
            fmt="BMHK",
            op=self.OP,
        )

        if isinstance(
            self.attn_bias,
            xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ):
            self.q = self.q.view(1, -1, *self.q.shape[2:])
            self.k = self.k.view(1, -1, *self.k.shape[2:])
            self.v = self.v.view(1, -1, *self.v.shape[2:])

        if hasattr(self.OP, "not_supported_reasons"):
            inp = xops.fmha.Inputs(
                query=self.q, key=self.k, value=self.v, attn_bias=self.attn_bias
            )
            not_supported_reasons = self.OP.not_supported_reasons(inp)
            if not_supported_reasons:
                raise NotSupportedInputError(not_supported_reasons)


class AttentionDecodingCKDecoder(AttentionDecodingBase):
    OP = xops.fmha.ck_decoder.FwOp


class AttentionDecodingSplitKV(AttentionDecodingBase):
    OP = xops.fmha.triton_splitk.FwOp


class AttentionDecodingCKSplitKV(AttentionDecodingBase):
    OP = xops.fmha.ck_splitk.FwOp


class AttentionDecodingSplitInt4KV(AttentionDecodingBase):
    OP = xops.fmha.triton_splitk.FwOp

    def __init__(
        self,
        B: int,
        Mq: int,
        Mkv: int,
        Hq: int,
        Hkv: int,
        K: int,
        bw: bool,
        attn_bias_type,
    ) -> None:
        # super(AttentionDecodingSplitInt4KV, self).__init__(B, Mq, Mkv, Hq, Hkv, K, bw, attn_bias_type)
        dtype = torch.float16
        torch.manual_seed(10)
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

        num_groups = 1
        self.k = (
            quantize_kv_int4(self.k, num_groups=num_groups)
            .contiguous()
            .view(torch.int32)
        ).expand(-1, -1, -1, Hq // Hkv, -1)
        self.v = (
            quantize_kv_int4(self.v, num_groups=num_groups)
            .contiguous()
            .view(torch.int32)
        ).expand(-1, -1, -1, Hq // Hkv, -1)

        if Hq == Hkv:
            self.q = self.q[:, :, :, 0]
            self.k = self.k[:, :, :, 0]
            self.v = self.v[:, :, :, 0]
        if Hkv == 1:
            self.q = self.q[:, :, 0]
            self.k = self.k[:, :, 0]
            self.v = self.v[:, :, 0]

        self.attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=Hq,
            num_heads_groups=Hq // Hkv,
            q_len=Mq,
            kv_len=Mkv,
            dtype=dtype,
            device=device,
            requires_grad=False,
            fmt="BMHK",
            op=self.OP,
        )

        if isinstance(
            self.attn_bias,
            xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        ):
            self.q = self.q.view(1, -1, *self.q.shape[2:])
            self.k = self.k.view(1, -1, *self.k.shape[2:])
            self.v = self.v.view(1, -1, *self.v.shape[2:])

        if hasattr(self.OP, "not_supported_reasons"):
            inp = xops.fmha.Inputs(
                query=self.q, key=self.k, value=self.v, attn_bias=self.attn_bias
            )
            not_supported_reasons = self.OP.not_supported_reasons(inp)
            if not_supported_reasons:
                raise NotSupportedInputError(not_supported_reasons)


class AttentionDecodingPyTorchRepeat(AttentionDecodingBase):
    def fw(self) -> None:
        B, Mq, Mkv, Hq, Hkv, K = self.shapes
        scale = 1 / K**0.5
        q = self.q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3)
        k = self.k.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        v = self.v.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-1, -2) * scale).softmax(-1)
        return attn @ v


BENCHMARKS: Dict[str, Type[AttentionDecodingBase]] = {
    "pytorch": AttentionDecodingPyTorchRepeat,
}

if torch.version.cuda:
    BENCHMARKS["cutlass"] = AttentionDecodingCUTLASS

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
    BENCHMARKS["triton_int4KV"] = AttentionDecodingSplitInt4KV

try:
    import flash_attn

    class AttentionDecodingFlashAttention(AttentionDecodingBase):
        def fw(self) -> None:
            q, k, v = self.q, self.k, self.v
            if q.ndim == 5:
                B, Mq, H1, H2, K = q.shape
                B, Mkv, H1, H2, K = k.shape
                q = q.reshape([B, Mq, H1 * H2, K])
                k = k[:, :, :, 0]
                v = v[:, :, :, 0]
            return flash_attn.flash_attn_func(q, k, v)

    BENCHMARKS[f"flash-attention@{flash_attn.__version__}"] = (
        AttentionDecodingFlashAttention
    )
except ImportError:
    pass


TEST_CASES = [
    dict(
        B=max(1, 2 ** (16 - i)),
        Mq=1,
        Mkv=2**i,
        Hq=16,
        Hkv=hkv,
        K=128,
        attn_bias_type=None,
    )
    for i in range(8, 18)
    for hkv in range(1, 3)
] + [
    dict(B=i, Mq=1, Mkv=4097, Hq=8, Hkv=1, K=128, attn_bias_type=None)
    for i in [2, 4, 8, 16, 32, 64, 128]
]


def get_benchmark_names():
    decoder_names = list(BENCHMARKS.keys())
    decoder_names.remove("pytorch")
    return decoder_names


# tests to verify correctness of each decoder implementation
@pytest.mark.parametrize(
    "name, case",
    [(name, case) for name in get_benchmark_names() for case in TEST_CASES],
)
def test_flash_attention_decoder(name, case):
    baseline = AttentionDecodingPyTorchRepeat(
        case["B"],
        case["Mq"],
        case["Mkv"],
        case["Hq"],
        case["Hkv"],
        case["K"],
        False,
        case["attn_bias_type"],
    )
    if name == "ck-decoder" and case["Mkv"] >= 2**14:
        pytest.skip("ck-decoder does not support Mkv >= 16K")

    baseline_out = baseline.fw()
    inputs = baseline.get_inputs()
    decoder = BENCHMARKS[name]

    assert name in ["ck-decoder", "ck_splitK", "ck", "triton_splitK", "triton_int4KV"]
    decoder_output, ctx = decoder.OP.apply(inputs, False)

    q, k, v = inputs.get_qkv_in_bmghk()
    B, M, G, H, Kq = q.shape
    mqa_swap_seqlen_head = False
    if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
        mqa_swap_seqlen_head = True
    if mqa_swap_seqlen_head:
        decoder_output = (
            decoder_output.reshape(B, -1, M, Kq).transpose(1, 2).contiguous()
        )
    else:
        decoder_output = decoder_output.reshape(B, H * G, -1, Kq).contiguous()

    decoder_output = decoder_output.transpose(2, 1).contiguous()
    torch.testing.assert_close(decoder_output, baseline_out, atol=1e-2, rtol=0)


def main() -> None:
    """
    run performance benchmark
    """
    benchmark_main_helper2(
        "attn_decoding",
        fw=True,
        cases=CASES,
        functions=BENCHMARKS,
        min_run_time=min_run_time,
    )


if __name__ == "__main__":
    main()  # pragma: no cover
