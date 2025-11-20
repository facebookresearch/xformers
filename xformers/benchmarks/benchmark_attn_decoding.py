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
from xformers.ops.fmha.triton_splitk import InputsFp8
from xformers.benchmarks.utils import benchmark_main_helper2, NotSupportedInputError, is_ocp_fp8, quantize_kv_int4, quantize_fp8_asymmetric


min_run_time = 0.5
device = torch.device("cuda")
pt_fp8_dtype = torch.float8_e4m3fn if is_ocp_fp8() else torch.float8_e4m3fnuz


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

CASES = [
    dict(
        B=128,
        Mq=1,
        Mkv=32769,
        Hq=8,
        Hkv=1,
        K=128,
        attn_bias_type=xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        # attn_bias_type=None,
    ),
    dict(
        B=128,
        Mq=1,
        Mkv=8193,
        Hq=8,
        Hkv=1,
        K=128,
        attn_bias_type=xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        # attn_bias_type=None,
    ),
]

MOE_CASES = [
    dict(
        B=b,
        Mq=1,
        Mkv=mkv,
        Hq=15,
        Hkv=1,
        K=128,
        attn_bias_type=xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    for b in [32, 128]
    for mkv in [8193, 32769]
]

CASES += MOE_CASES

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
        dtype = torch.bfloat16
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


# triton attention decoder using fp8 as input
class AttentionDecodingSplitFp8KV(AttentionDecodingBase):
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
        dtype = torch.bfloat16
        torch.manual_seed(10)
        self.sub_label = (
            f"B={B} Mq={Mq} Mkv={Mkv} Hq={Hq} Hkv={Hkv} K={K} TotalBytes="
            f"{((B * Mkv * Hkv * K * 2) + ((B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2)}"
        )
        self.label = "attn_decoding"
        self.shapes = (B, Mq, Mkv, Hq, Hkv, K)

        G = Hq // Hkv
        max_context_length = Mkv

        assert Hkv <= Hq
        assert Hq % Hkv == 0

        self.q = torch.randn(1, B * Mq, Hkv, G, K, dtype=dtype, device=device)
        self.k = torch.randn(1, B * max_context_length, Hkv, 1, K, dtype=dtype, device=device)
        self.v = torch.randn(1, B * max_context_length, Hkv, 1, K, dtype=dtype, device=device)

        k_fp8, k_fp8_scales, k_fp8_shifts = quantize_fp8_asymmetric(
            self.k.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
        )
        v_fp8, v_fp8_scales, v_fp8_shifts = quantize_fp8_asymmetric(
            self.v.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
        )

        k_fp8_scales = k_fp8_scales.to(torch.float16)
        v_fp8_scales = v_fp8_scales.to(torch.float16)
        k_fp8_shifts = k_fp8_shifts.to(torch.float16)
        v_fp8_shifts = v_fp8_shifts.to(torch.float16)

        def _combine_scale_shift(scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return (
                torch.concat([scale.unsqueeze(-1), shift.unsqueeze(-1)], dim=-1)
                .flatten(-2)
                .to(torch.float16)
            )

        # for fp8 direct input. kv are of the fp8 data type, while scale and shift are of
        # fp16 data type, and concated as one input
        k_fp8_scales_shifts = _combine_scale_shift(k_fp8_scales, k_fp8_shifts)
        v_fp8_scales_shifts = _combine_scale_shift(v_fp8_scales, v_fp8_shifts)

        def _to_expanded_shape(x):
            return x.view(1, B * max_context_length, Hkv, 1, -1).expand(
                1, B * max_context_length, Hkv, G, -1
            )

        self.k_fp8_scales_shifts = (
            _to_expanded_shape(k_fp8_scales_shifts).squeeze(-1)
        )
        self.v_fp8_scales_shifts = (
            _to_expanded_shape(v_fp8_scales_shifts).squeeze(-1)
        )
        self.k_fp8 = _to_expanded_shape(k_fp8)
        self.v_fp8 = _to_expanded_shape(v_fp8)

        self.attn_bias = None
        if attn_bias_type is not None:
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

            seq_len = torch.full((B, ), Mkv, dtype=torch.int32, device='cuda')
            self.attn_bias.k_seqinfo.seqlen = seq_len
            self.attn_bias.k_seqinfo.max_seqlen=Mkv

    def get_inputs(self):
        inp = InputsFp8(
            query=self.q,
            key=self.k_fp8,
            value=self.v_fp8,
            k_fp8_scale_shift=self.k_fp8_scales_shifts,
            v_fp8_scale_shift=self.v_fp8_scales_shifts,
            attn_bias=self.attn_bias,
        )
        return inp

    def fw(self) -> None:
        try:
            xops.fmha._memory_efficient_attention_forward(
                self.get_inputs(), op=xops.fmha.triton_splitk.FwOp
            )
        except (RuntimeError, ValueError) as e:
            print(f"Runtime error: {e}")


# triton attention decoder using packed fp8 as input
class AttentionDecodingSplitPackedFp8KV(AttentionDecodingBase):
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
        dtype = torch.bfloat16
        torch.manual_seed(10)
        self.sub_label = (
            f"B={B} Mq={Mq} Mkv={Mkv} Hq={Hq} Hkv={Hkv} K={K} TotalBytes="
            f"{((B * Mkv * Hkv * K * 2) + ((B * Mq * Hq * K) + (B * Mq * Hq * K)) * 2)}"
        )
        self.label = "attn_decoding"
        self.shapes = (B, Mq, Mkv, Hq, Hkv, K)

        G = Hq // Hkv
        max_context_length = Mkv

        assert Hkv <= Hq
        assert Hq % Hkv == 0

        self.q = torch.randn(1, B * Mq, Hkv, G, K, dtype=dtype, device=device)
        self.k = torch.randn(1, B * max_context_length, Hkv, 1, K, dtype=dtype, device=device)
        self.v = torch.randn(1, B * max_context_length, Hkv, 1, K, dtype=dtype, device=device)

        k_fp8, k_fp8_scales, k_fp8_shifts = quantize_fp8_asymmetric(
            self.k.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
        )
        v_fp8, v_fp8_scales, v_fp8_shifts = quantize_fp8_asymmetric(
            self.v.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
        )

        k_fp8_packed, v_fp8_packed = k_fp8.view(torch.int32), v_fp8.view(torch.int32)

        def _to_expanded_shape(x):
            return x.view(1, B * max_context_length, Hkv, 1, -1).expand(
                1, B * max_context_length, Hkv, G, -1
            )

        def _combine_scale_shift_packed(scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return (
                torch.concat([scale.unsqueeze(-1), shift.unsqueeze(-1)], dim=-1)
                .flatten(-2)
                .to(torch.float16)
                .view(torch.int32)
            )

        self.k_fp8_packed = _to_expanded_shape(k_fp8_packed)
        self.v_fp8_packed = _to_expanded_shape(v_fp8_packed)

        k_fp8_scales_shifts_packed = _combine_scale_shift_packed(k_fp8_scales, k_fp8_shifts)
        v_fp8_scales_shifts_packed = _combine_scale_shift_packed(v_fp8_scales, v_fp8_shifts)

        self.k_fp8_scales_shifts_packed = (
            _to_expanded_shape(k_fp8_scales_shifts_packed).squeeze(-1).contiguous()
        )
        self.v_fp8_scales_shifts_packed = (
            _to_expanded_shape(v_fp8_scales_shifts_packed).squeeze(-1).contiguous()
        )

        self.attn_bias = None
        if attn_bias_type is not None:
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

            #hard code sequence len to be the same as the
            seq_len = torch.full((B, ), Mkv, dtype=torch.int32, device='cuda')
            self.attn_bias.k_seqinfo.seqlen = seq_len
            self.attn_bias.k_seqinfo.max_seqlen=Mkv

    def get_inputs(self):
        inp = InputsFp8(
            query=self.q,
            key=self.k_fp8_packed,
            value=self.v_fp8_packed,
            k_fp8_scale_shift=self.k_fp8_scales_shifts_packed,
            v_fp8_scale_shift=self.v_fp8_scales_shifts_packed,
            attn_bias=self.attn_bias,
        )
        return inp

    def fw(self) -> None:
        try:
            xops.fmha._memory_efficient_attention_forward(
                self.get_inputs(), op=xops.fmha.triton_splitk.FwOp
            )
        except (RuntimeError, ValueError) as e:
            print(f"Runtime error: {e}")


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



if (sys.version_info.major, sys.version_info.minor) >= (3, 9):
    BENCHMARKS["triton_splitK"] = AttentionDecodingSplitKV
    BENCHMARKS["packed_fp8"] = AttentionDecodingSplitPackedFp8KV
    BENCHMARKS["fp8"] = AttentionDecodingSplitFp8KV
    # BENCHMARKS["triton_int4KV"] = AttentionDecodingSplitInt4KV

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


def dequantization(inp, B, Mq, Mkv, Hq, Hkv, K):
    q = inp.query
    k_fp8 = inp.key
    v_fp8 = inp.value
    q = q.reshape([B, Mq, -1, K]).permute(0, 2, 1, 3).contiguous()

    k_fp8 = k_fp8.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    k_f32 = k_fp8.to(torch.float32).contiguous()

    k_fp8_scale_shift = inp.k_fp8_scale_shift
    k_scale_f32 = k_fp8_scale_shift[:,:,:,:,0].to(torch.float32)
    k_scale_f32 = k_scale_f32.reshape([B, Mkv, -1, 1]).permute(0, 2, 1, 3).contiguous()
    k_shift_f32 = k_fp8_scale_shift[:,:,:,:,1].to(torch.float32)
    k_shift_f32 = k_shift_f32.reshape([B, Mkv, -1, 1]).permute(0, 2, 1, 3).contiguous()
    k = k_f32 * k_scale_f32 + k_shift_f32

    v_fp8_scale_shift = inp.v_fp8_scale_shift
    v_scale_f32 = v_fp8_scale_shift[:,:,:,:,0].to(torch.float32)
    v_scale_f32 = v_scale_f32.reshape([B, Mkv, -1, 1]).permute(0, 2, 1, 3)

    v_shift_f32 = v_fp8_scale_shift[:,:,:,:,1].to(torch.float32)
    v_shift_f32 = v_shift_f32.reshape([B, Mkv, -1, 1]).permute(0, 2, 1, 3)

    v_fp8 = v_fp8.reshape([B, Mkv, -1, K]).permute(0, 2, 1, 3)
    v_f32 = v_fp8.to(torch.float32).contiguous()

    v = v_f32 * v_scale_f32 + v_shift_f32

    return q, k, v


def attention_naive(inp, B, Mq, Mkv, Hq, Hkv, K):

    q, k, v = dequantization(inp, B, Mq, Mkv, Hq, Hkv, K)

    scale = 1 / K**0.5
    attn = (q.to(torch.float32) @ k.transpose(-1, -2) * scale).softmax(-1)

    return (attn @ v).to(q.dtype)


TEST_CASES = [
    dict(
        B=128,
        Mq=1,
        Mkv=32769,
        Hq=8,
        Hkv=1,
        K=128,
        attn_bias_type=xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        # attn_bias_type=None,
    ),
    dict(
        B=128,
        Mq=1,
        Mkv=8193,
        Hq=8,
        Hkv=1,
        K=128,
        attn_bias_type=xops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
        # attn_bias_type=None,
    ),
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
    if name == "ck-decoder" and case["Mkv"] >= 2**14:
        pytest.skip("ck-decoder does not support Mkv >= 16K")
    decoder = BENCHMARKS[name](
        case["B"],
        case["Mq"],
        case["Mkv"],
        case["Hq"],
        case["Hkv"],
        case["K"],
        False,
        case["attn_bias_type"],
    )
    inputs = decoder.get_inputs()

    assert name in ["ck_splitK", "ck", "triton_splitK", "triton_int4KV", "packed_fp8", "fp8"]
    decoder_output, ctx = decoder.OP.apply(inputs, False)

    # compute baseline using fp8 inputs to avoid the quanti/dequatnization error
    naive_output = attention_naive(inputs, case["B"], case["Mq"], case["Mkv"], case["Hq"], case["Hkv"], case["K"])
    k = inputs.key
    v = inputs.value
    q = inputs.query
    M, B, G, H, Kq = q.shape

    mqa_swap_seqlen_head = False
    if k.shape[3] > 1 and k.stride(3) == 0 and v.stride(3) == 0:
        mqa_swap_seqlen_head = True
    if mqa_swap_seqlen_head:
        decoder_output = (
            decoder_output.reshape(B, -1, M * G, Kq).transpose(1, 2).contiguous()
        )
    else:
        decoder_output = decoder_output.reshape(B, H * G, -1, Kq).contiguous()
    decoder_output = decoder_output.transpose(2, 1).contiguous()
    torch.testing.assert_close(decoder_output, naive_output, atol=5e-4, rtol=0.000)


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


