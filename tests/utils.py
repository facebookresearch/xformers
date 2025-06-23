# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps
from typing import Optional, Tuple

import numpy as np
import pytest
import torch

from xformers.attn_bias_utils import pack_kv_cache, ref_attention, ref_attention_bmhk
from xformers.ops.fmha import Inputs
from xformers.ops.fmha.attn_bias import (
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
)
from xformers.ops.fmha.triton_splitk import InputsFp8

cuda_or_mtia_only = pytest.mark.skipif(
    not torch.cuda.is_available() and not torch.mtia.is_available(),
    reason="requires CUDA or MTIA",
)
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
rocm_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.version.hip, reason="requires ROCM"
)
disable_on_rocm = pytest.mark.skipif(
    not not torch.version.hip, reason="could not be done on ROCM"
)
disable_on_mtia = pytest.mark.skipif(
    torch.mtia.is_available(), reason="Not supported yet on MTIA"
)


# We don't want to compare MTIA output against another MTIA output yet for 2 reasons:
#   1. Some kernels may have bugs, and the reference implementation may share some of
#      the same kernels as the mem_eff implementation. We would then end up comparing
#      a faulty output against another faulty output, which could lead to tests passing
#      when they shouldn't.
#   2. We may run on some emulated devices that can be slower than the CPU implementation,
#      and therefore increase the time it takes to run the tests by a lot.
def use_cpu_ref(device: str):
    return device.startswith("mtia")


def maybe_use_cpu_ref(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        new_args = list(args)
        new_kwargs = kwargs.copy()
        original_device = None

        for key in kwargs:
            if isinstance(kwargs[key], torch.Tensor) and use_cpu_ref(
                kwargs[key].device.type
            ):
                assert original_device is None or kwargs[key].device == original_device
                original_device = kwargs[key].device
                new_kwargs[key] = kwargs[key].cpu()

        for index, arg in enumerate(new_args):
            if isinstance(arg, torch.Tensor) and use_cpu_ref(arg.device.type):
                assert original_device is None or arg.device == original_device
                original_device = arg.device
                new_args[index] = arg.cpu()

        output = fn(*new_args, **new_kwargs)

        if isinstance(output, torch.Tensor) and original_device is not None:
            output = output.to(original_device)

        return output

    return wrapped


def disable_tf32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        cuda, cudnn = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            False,
            False,
        )
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                cuda,
                cudnn,
            )

    return wrapped


ref_attention_for_test = disable_tf32(maybe_use_cpu_ref(ref_attention))
ref_attention_bmhk_for_test = disable_tf32(maybe_use_cpu_ref(ref_attention_bmhk))


def assert_allclose(
    out: Optional[torch.Tensor],
    ref: Optional[torch.Tensor],
    msg: str = "failed",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> None:
    assert out is not None, f"{msg}: output Tensor is None"
    assert ref is not None, f"{msg}: reference Tensor is None"
    assert out.shape == ref.shape, f"Shape: {out.shape} (expected: {ref.shape})"
    if out.dtype != ref.dtype:
        assert False, f"out dtype: {out.dtype}, ref dtype: {ref.dtype}"
    if out.numel() == 0:
        return
    flatten_diff = ((out - ref).abs() - atol - ref.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()
    max_location = np.unravel_index(int(max_pos), out.shape)
    max_diff = flatten_diff[max_pos]
    num_different = flatten_diff.numel() - torch.count_nonzero(flatten_diff <= 0)
    percentage = num_different / flatten_diff.numel()
    del flatten_diff
    assert torch.allclose(out, ref, rtol=rtol, atol=atol), (
        f"{msg}: "
        f"out={out.flatten()[max_pos]} and ref={ref.flatten()[max_pos]} (diff={max_diff} > 0)"
        f" at {max_location} of shape {tuple(out.shape)} / atol={atol}, rtol={rtol}"
        f"/ total failing elements: {num_different} ({percentage*100:.3}%)"
    )


def construct_fp8_attention_inputs(
    B: int,
    Mkv: int,
    Mq: int,
    Hkv: int,
    Hq: int,
    K: int,
    page_size: int,
    device: torch.device,
    dtype: torch.dtype,
    randomize_lengths: bool = True,
) -> Tuple[InputsFp8, Inputs, InputsFp8, Inputs]:
    """
    Construct inputs for benchmarks and tests of Triton Split-k attention
    with fused row-wise FP8 dequantization.
    Quantization coefficients are packed as int32 tensors where each
    element contains two fp16 numbers - scales and shifts.
    """
    G = Hq // Hkv
    q = torch.randn(1, B * Mq, Hkv, G, K, dtype=dtype, device=device)
    k = torch.randn(1, B * Mkv, Hkv, 1, K, dtype=dtype, device=device)
    v = torch.randn(1, B * Mkv, Hkv, 1, K, dtype=dtype, device=device)

    pt_fp8_dtype = (
        torch.float8_e4m3fnuz if torch.version.hip is not None else torch.float8_e4m3fn
    )

    k_fp8, k_fp8_scales, k_fp8_shifts = quantize_fp8_asymmetric(
        k.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
    )
    v_fp8, v_fp8_scales, v_fp8_shifts = quantize_fp8_asymmetric(
        v.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
    )

    k_fp8, v_fp8 = k_fp8.view(torch.int32), v_fp8.view(torch.int32)

    k_fp8_scales = k_fp8_scales.to(torch.float16)
    v_fp8_scales = v_fp8_scales.to(torch.float16)
    k_fp8_shifts = k_fp8_shifts.to(torch.float16)
    v_fp8_shifts = v_fp8_shifts.to(torch.float16)

    def _to_expanded_shape(x):
        return x.view(1, B * Mkv, Hkv, 1, -1).expand(1, B * Mkv, Hkv, G, -1)

    k_fp8 = _to_expanded_shape(k_fp8)
    v_fp8 = _to_expanded_shape(v_fp8)

    k_fp8_scales_shifts = _combine_scale_shift(k_fp8_scales, k_fp8_shifts)
    v_fp8_scales_shifts = _combine_scale_shift(v_fp8_scales, v_fp8_shifts)

    k_fp8_scales_shifts = (
        _to_expanded_shape(k_fp8_scales_shifts).squeeze(-1).contiguous()
    )
    v_fp8_scales_shifts = (
        _to_expanded_shape(v_fp8_scales_shifts).squeeze(-1).contiguous()
    )

    k_fp8_scales = _to_expanded_shape(k_fp8_scales).squeeze(-1).to(torch.float16)
    v_fp8_scales = _to_expanded_shape(v_fp8_scales).squeeze(-1).to(torch.float16)
    k_fp8_shifts = _to_expanded_shape(k_fp8_shifts).squeeze(-1).to(torch.float16)
    v_fp8_shifts = _to_expanded_shape(v_fp8_shifts).squeeze(-1).to(torch.float16)

    kv_seqlens = (
        torch.randint(0, Mkv, size=(B,)).tolist() if randomize_lengths else [Mkv] * B
    )

    k_deq = dequantize_fp8_asymmetric(
        k_fp8.view(pt_fp8_dtype), k_fp8_scales, k_fp8_shifts
    ).to(dtype)
    v_deq = dequantize_fp8_asymmetric(
        v_fp8.view(pt_fp8_dtype), v_fp8_scales, v_fp8_shifts
    ).to(dtype)

    k_deq = k_deq[:, :, :, :1, :].contiguous()
    v_deq = v_deq[:, :, :, :1, :].contiguous()

    attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1 for _ in range(B)],
        kv_seqlen=kv_seqlens,
        kv_padding=Mkv,
    )
    inp = InputsFp8(
        query=q,
        key=k_fp8,
        value=v_fp8,
        k_fp8_scale_shift=k_fp8_scales_shifts,
        v_fp8_scale_shift=v_fp8_scales_shifts,
        attn_bias=attn_bias,
    )
    inp_ref = Inputs(
        query=q,
        key=_to_expanded_shape(k_deq),
        value=_to_expanded_shape(v_deq),
        attn_bias=attn_bias,
    )

    # Paged K/V cache
    block_tables, packed_cache_k, packed_cache_v = pack_kv_cache(
        k_deq.view(B, Mkv, Hkv, -1),
        v_deq.view(B, Mkv, Hkv, -1),
        kv_seqlens,
        page_size,
    )
    block_tables_orig, packed_cache_k_orig, packed_cache_v_orig = pack_kv_cache(
        k.view(B, Mkv, Hkv, -1),
        v.view(B, Mkv, Hkv, -1),
        kv_seqlens,
        page_size,
    )
    assert (block_tables_orig == block_tables).all()

    (
        packed_cache_k_fp8,
        packed_k_fp8_scales,
        packed_k_fp8_shifts,
    ) = quantize_fp8_asymmetric(
        packed_cache_k_orig.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
    )
    (
        packed_cache_v_fp8,
        packed_v_fp8_scales,
        packed_v_fp8_shifts,
    ) = quantize_fp8_asymmetric(
        packed_cache_v_orig.view(-1, K), pt_fp8_dtype=pt_fp8_dtype
    )

    total_len_rounded = packed_cache_k_fp8.shape[0] // Hkv

    packed_cache_k_fp8 = packed_cache_k_fp8.view(torch.int32).view(
        1, total_len_rounded, Hkv, -1
    )
    packed_cache_v_fp8 = packed_cache_v_fp8.view(torch.int32).view(
        1, total_len_rounded, Hkv, -1
    )

    packed_k_fp8_scales_shifts = _combine_scale_shift(
        packed_k_fp8_scales, packed_k_fp8_shifts
    )
    packed_v_fp8_scales_shifts = _combine_scale_shift(
        packed_v_fp8_scales, packed_v_fp8_shifts
    )

    def _to_packed_expanded_shape(x):
        return x.reshape(1, total_len_rounded, Hkv, 1, -1).expand(
            1, total_len_rounded, Hkv, Hq // Hkv, -1
        )

    packed_k_fp8_scales_shifts = (
        _to_packed_expanded_shape(packed_k_fp8_scales_shifts).squeeze(-1).contiguous()
    )
    packed_v_fp8_scales_shifts = (
        _to_packed_expanded_shape(packed_v_fp8_scales_shifts).squeeze(-1).contiguous()
    )

    attn_bias_paged = attn_bias.make_paged(
        block_tables=block_tables,
        page_size=page_size,
        paged_type=PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    inp_bf16_paged = Inputs(
        query=q,
        key=_to_packed_expanded_shape(packed_cache_k),
        value=_to_packed_expanded_shape(packed_cache_v),
        attn_bias=attn_bias_paged,
    )
    inp_fp8_paged = InputsFp8(
        query=q,
        key=_to_packed_expanded_shape(packed_cache_k_fp8),
        value=_to_packed_expanded_shape(packed_cache_v_fp8),
        attn_bias=attn_bias_paged,
        k_fp8_scale_shift=packed_k_fp8_scales_shifts,
        v_fp8_scale_shift=packed_v_fp8_scales_shifts,
    )

    return inp, inp_ref, inp_fp8_paged, inp_bf16_paged


def _combine_scale_shift(scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return (
        torch.concat([scale.unsqueeze(-1), shift.unsqueeze(-1)], dim=-1)
        .flatten(-2)
        .to(torch.float16)
        .view(torch.int32)
    )


def quantize_fp8_asymmetric(
    x: torch.Tensor,
    pt_fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_fp8 = torch.finfo(pt_fp8_dtype).max

    shift = x.mean(dim=1)
    x_centered = x - shift[..., None]

    row_max: torch.Tensor = x_centered.abs().max(dim=-1)[0]
    scale = max_fp8 * row_max.to(torch.float32).pow(-1)
    scale = torch.nan_to_num(scale, posinf=1)

    x_quant = (x_centered / scale[..., None]).to(pt_fp8_dtype)
    return x_quant, scale, shift


def dequantize_fp8_asymmetric(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return x.to(scale.dtype) * scale[..., None] + shift[..., None]
