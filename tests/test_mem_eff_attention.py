# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from functools import partial
from typing import List, Optional, Sequence, Tuple, Type, TypeVar

import pytest
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch.utils.checkpoint import checkpoint

import xformers.ops
from xformers.attn_bias_utils import create_attn_bias
from xformers.ops import fmha
from xformers.ops.fmha import ALL_BW_OPS, ALL_FW_OPS
from xformers.ops.fmha.common import AttentionOpBase
from xformers.ops.fmha.dispatch import _dispatch_fw_priority_list

from .utils import assert_allclose

torch.backends.cuda.matmul.allow_tf32 = False
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm70_or_better_only = pytest.mark.skipif(
    compute_capability < (7, 0), reason="requires sm70+"
)
sm75_or_better_only = pytest.mark.skipif(
    compute_capability < (7, 5), reason="requires sm75+"
)
sm80_or_better_only = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm80+"
)
_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

T = TypeVar(
    "T", Type[fmha.common.AttentionFwOpBase], Type[fmha.common.AttentionBwOpBase]
)


def _filter_unsupported_ops(ops: Sequence[T]) -> Sequence[T]:
    return [
        op
        for op in ops
        if (
            "cpu" in op.SUPPORTED_DEVICES
            or op.CUDA_MINIMUM_COMPUTE_CAPABILITY <= compute_capability
        )
        and op.is_available()
    ]


ALL_FW_OPS = _filter_unsupported_ops(ALL_FW_OPS)
ALL_BW_OPS = _filter_unsupported_ops(ALL_BW_OPS)


def sample_random_supported_fw(
    inp: fmha.Inputs, seed: int
) -> Type[fmha.common.AttentionFwOpBase]:
    r = random.Random(seed)
    fw_ops = list(ALL_FW_OPS)
    r.shuffle(fw_ops)
    for op in fw_ops:
        if op.supports(inp):
            return op
    raise NotImplementedError(f"Could not find a FW operator for: {inp}")


def generate_test_shapes_B_Mq_Mkv_H_K_Kv(op):
    shapes = []
    for B in op._TEST_BATCH_SIZES:
        for Mq in [32, 256]:
            for Mkv in [32, 64, 256, 1024]:
                for K in op._TEST_K:
                    shapes.append((B, Mq, Mkv, 1, K, K))
        Mq = 256
        Mkv = 128
        K = 32
        H = 1
        # Weird values of parameters
        for M in [2, 3, 15, 31, 32, 34, 68, 72, 90, 132, 136]:
            shapes.append((B, M, Mkv, H, K, K))
            shapes.append((B, Mq, M, H, K, K))
        for _K in [1, 2, 3, 31, 34, 36, 38, 40, 64, 80, 160, 256 + 2, 256 + 8, 512]:
            if _K <= op.SUPPORTED_MAX_K:
                shapes.append((B, Mq, Mkv, H, _K, _K))
        # Different value for K / Kv
        if op.SUPPORTS_DIFFERENT_VALUE_EMBED:
            for _K in [32, 36, 64, 256 + 8]:
                shapes.append((B, Mq, Mkv, H, K, _K))
                shapes.append((B, Mq, Mkv, H, _K, K))
        # Exotic sizes
        for K in op._TEST_K:
            shapes.append((B, 16, 1024, H, K, K))
            shapes.append((B, 1024, 16, H, K, K))
        # Some number of heads
        for H in [3, 5, 12]:
            shapes.append((max(1, B // H), Mq, Mkv, H, K, K))
    # Filter-out not supported shapes
    shapes = [
        shape
        for shape in shapes
        if len(
            op.shape_not_supported_reasons(
                Mq=shape[1], Mkv=shape[2], K=shape[4], Kv=shape[5]
            )
        )
        == 0
    ]
    # Add some random shapes
    if op in [
        fmha.cutlass.FwOp,
        fmha.cutlass.BwOp,
        fmha.flash.BwOp,
    ]:
        K_CHOICES = [8 * i for i in range(1, 256 // 8)]
        r = random.Random(0)
        found_count = 0
        while found_count < 200:
            B = r.randint(1, 400)
            Mq = r.randint(1, 500)
            Mkv = r.randint(1, 500)
            H = r.randint(2, 11)
            B = max(B // H, 1)
            K = r.choice(K_CHOICES)
            Kv = r.choice(K_CHOICES)
            if not op.SUPPORTS_DIFFERENT_VALUE_EMBED:
                Kv = K
            if len(op.shape_not_supported_reasons(Mq, Mkv, K, Kv)):
                continue
            found_count += 1
            shapes.append((B, Mq, Mkv, H, K, Kv))
    return shapes


def make_id(op, device, dtype, bias_type, *shape):
    return (
        f"{op.NAME}-{device}-{str(dtype)}-{bias_type.__name__}"
        f"-{'-'.join([str(s) for s in shape])}"
    )


def _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(
    ops_list: Sequence[Type[fmha.AttentionOpBase]], max_shapes_per_op: int = 65000
):
    r = random.Random(0)
    combination = []
    ids = []
    for op in ops_list:
        op_count = 0
        # Sort list of masks, so it's deterministic across runs
        LIST_MASKS = list(sorted(op.SUPPORTED_ATTN_BIAS_TYPES, key=lambda x: str(x)))
        for shape in generate_test_shapes_B_Mq_Mkv_H_K_Kv(op):
            has_one = False
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                for dtype in op.SUPPORTED_DTYPES:
                    bias_type = r.choice(LIST_MASKS)
                    # Avoid using too much memory
                    if bias_type not in [
                        type(None),
                        fmha.attn_bias.LowerTriangularMask,
                    ]:
                        B, Mq, Mkv, H, K, Kv = shape
                        B = min(B, 12)

                        if bias_type in {
                            fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
                            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
                        }:
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq) + 2
                        elif (
                            bias_type
                            is fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
                        ):
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq)
                        shape = (B, Mq, Mkv, H, K, Kv)
                    combination.append((op, device, dtype, bias_type, *shape))
                    ids.append(
                        f"{op.NAME}-{device}-{str(dtype)}-{bias_type.__name__}"
                        f"-{'-'.join([str(s) for s in shape])}"
                    )
                    has_one = True
            if has_one:
                op_count += 1
            if op_count > max_shapes_per_op:
                break
        # Some specific shapes for which we want to run without any mask
        bias_type = type(None)
        for shape in (
            # Some strides/dims don't fit on an uint16
            (1, 128, 128, 300, 128, 128),
            (13, 1, 67, 200, 8, 8),
            (1, 1 + 2**16, 4, 1, 8, 8),
            (1, 4, 1 + 2**16, 1, 8, 8),
            # TODO: Some strides don't fit on an uint32
            # Crashes on Flash, Errors on Cutlass
            # (1, 1, 64000, 300, 128, 128)
        ):
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                for dtype in op.SUPPORTED_DTYPES:
                    combination.append((op, device, dtype, bias_type, *shape))
    return {
        "argvalues": combination,
        "ids": [make_id(*c) for c in combination],
    }


parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_FW_OPS),
)
parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs = pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_FW_OPS, max_shapes_per_op=1),
)
parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = pytest.mark.parametrize(
    "opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_BW_OPS),
)
parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs = pytest.mark.parametrize(
    "opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    **_generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(ALL_BW_OPS, max_shapes_per_op=1),
)


def ref_attention(q, k, v, attn_bias=None, drop_mask=None, p=0.0, scale=None):
    if q.ndim == 5:

        def attn_bias_group(group: int):
            if isinstance(attn_bias, torch.Tensor):
                return attn_bias[:, group]
            if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
                return fmha.attn_bias.LowerTriangularMaskWithTensorBias(
                    attn_bias._bias[:, group]
                )
            return attn_bias

        return torch.stack(
            [
                ref_attention_bmhk(
                    q[:, :, g],
                    k[:, :, g],
                    v[:, :, g],
                    scale=scale,
                    attn_bias=attn_bias_group(g),
                )
                for g in range(q.shape[2])
            ],
            dim=2,
        )
    if q.ndim == 4:
        assert p == 0.0
        return ref_attention_bmhk(q, k, v, scale=scale, attn_bias=attn_bias)
    q = q.float()
    k = k.float()
    v = v.float()

    scale = scale if scale is not None else (1 / q.shape[-1] ** 0.5)
    q = q * scale

    attn = q @ k.transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape(
                [-1, *attn_bias_tensor.shape[2:]]
            )
        attn = attn + attn_bias_tensor.float()
    attn = attn.softmax(-1)
    if drop_mask is not None:
        attn = attn * (drop_mask / (1 - p))
    return attn @ v


def ref_attention_bmhk(q, k, v, attn_bias, scale=None) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention(T(q), T(k), T(v), attn_bias, scale=scale)
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


def _rand_partition(r: random.Random, total: int, n: int) -> List[int]:
    # returns list of n nonnegative integers summing to total
    idx = {0, total}
    while len(idx) < n + 1:
        idx.add(r.randint(1, total - 1))
    s = sorted(idx)
    return [e - b for b, e in zip(s[:-1], s[1:])]


def get_bias_grad(attn_bias, clear: bool = False) -> Optional[torch.Tensor]:
    tensor_with_grad: Optional[torch.Tensor] = None
    if isinstance(attn_bias, torch.Tensor):
        tensor_with_grad = attn_bias
    if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
        tensor_with_grad = attn_bias._bias
    if tensor_with_grad is not None:
        grad = tensor_with_grad.grad
        if clear:
            tensor_with_grad.grad = None
        return grad
    return None


def create_tensors(
    op: Type[AttentionOpBase],
    device,
    dtype,
    attn_bias_type,
    B,
    q_len,
    kv_len,
    h,
    k,
    kv,
    *,
    attn_bias_requires_grad: bool = False,
    fmt: str = "BMK",
    g: int = 1,
):
    torch.manual_seed(B * q_len + kv_len * k + kv)

    mask_is_bottom_right = attn_bias_type is not None and issubclass(
        attn_bias_type,
        (
            fmha.attn_bias.LowerTriangularFromBottomRightMask,
            fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask,
            fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
            fmha.attn_bias.BlockDiagonalCausalLocalAttentionMask,
            fmha.attn_bias.LocalAttentionFromBottomRightMask,
        ),
    )
    if mask_is_bottom_right and q_len > kv_len:
        # Bottom-right attention and local-attention masks require q_len <= kv_len
        kv_len = q_len
    scale = 3
    if fmt == "BMK":
        query = torch.randn((B * h, q_len, k), device=device, dtype=dtype)
        key = torch.randn((B * h, kv_len, k), device=device, dtype=dtype)
        value = torch.randn((B * h, kv_len, kv), device=device, dtype=dtype)
    elif fmt == "BMHK":
        query = torch.randn((B, q_len, h, k), device=device, dtype=dtype)
        key = torch.randn((B, kv_len, h, k), device=device, dtype=dtype)
        value = torch.randn((B, kv_len, h, kv), device=device, dtype=dtype)
    else:
        assert fmt == "BMGHK"
        query = torch.randn((B, q_len, g, h, k), device=device, dtype=dtype)
        key = torch.randn((B, kv_len, g, 1, k), device=device, dtype=dtype)
        value = torch.randn((B, kv_len, g, 1, kv), device=device, dtype=dtype)

    for x in [query, key, value]:
        x.mul_(scale)

    if fmt == "BMGHK":
        # Expand - after the in-place mul
        key = key.expand((B, kv_len, g, h, k))
        value = value.expand((B, kv_len, g, h, k))

    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(attn_bias_type):
        attn_bias_type = None
    attn_bias = None
    if attn_bias_type is not None:
        attn_bias = create_attn_bias(
            attn_bias_type,
            batch_size=B,
            num_heads=h,
            num_heads_groups=g,
            q_len=q_len,
            kv_len=kv_len,
            dtype=dtype,
            device=device,
            requires_grad=attn_bias_requires_grad,
            fmt=fmt,
            op=op,
        )
        if isinstance(
            attn_bias,
            (
                fmha.attn_bias.BlockDiagonalMask,
                fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
            ),
        ):
            query, key, value = [
                x.reshape([1, -1, *x.shape[2:]]) for x in [query, key, value]
            ]

    inputs = fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias)
    reasons = op.not_supported_reasons(inputs)
    if reasons:
        err_msg = f"{op.NAME}: unsupported ({'/'.join(reasons)})"
        # Ensure we free memory to avoid OOMs
        del query, key, value, attn_bias, inputs
        pytest.skip(err_msg)
    return query, key, value, attn_bias


def bmhk2bmk(tensor) -> torch.Tensor:
    return (
        tensor.permute((0, 2, 1, 3))
        .contiguous()
        .view([tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]])
    )


def bmk2bmhk(tensor, num_heads: int) -> torch.Tensor:
    return tensor.reshape([-1, num_heads, tensor.shape[1], tensor.shape[2]]).permute(
        (0, 2, 1, 3)
    )


@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("packed", [False, True])
@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_forward(opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, packed, fmt, **kwargs):
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    if packed and not (k == kv and q_len == kv_len):
        pytest.skip(
            f"packed incompatible with `k ({k}) != kv ({kv})` or `q_len ({q_len}) != kv_len ({kv_len})`"
        )
    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(bias_type):
        pytest.skip("BMK incompatible with this bias")

    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt="BMHK" if packed else fmt,
        **kwargs,
    )

    if packed:
        c = torch.stack([query, key, value], 2)
        if fmt == "BMK":
            # bm3hk -> 3bhmk -> 3Bmk
            c = c.permute(2, 0, 3, 1, 4).view([3, -1, q_len, k])
            query, key, value = c[0], c[1], c[2]
            # Re-create bias in the right format
            attn_bias = create_attn_bias(
                bias_type=bias_type,
                batch_size=batch_size,
                num_heads=h,
                num_heads_groups=1,
                q_len=q_len,
                kv_len=kv_len,
                device=device,
                dtype=dtype,
                requires_grad=False,
                fmt=fmt,
                op=op,
            )
        elif fmt == "BMHK":
            # bm3hk -> 3 x bmhk
            query, key, value = xformers.ops.unbind(c, 2)
        else:
            assert False, f"Unsupport fmt {fmt} with packing"
        assert not query.is_contiguous()

    out = xformers.ops.memory_efficient_attention_forward(
        query, key, value, attn_bias, op=op
    )
    assert not out.isnan().any(), ("Output has NaNs", attn_bias)
    out2 = xformers.ops.memory_efficient_attention_forward(
        query, key, value, attn_bias, op=op
    )
    assert torch.allclose(out, out2, atol=0.0, rtol=0.0), (
        "Non-deterministic behavior",
        attn_bias,
    )

    ref = ref_attention(query, key, value, attn_bias)
    assert out.shape == ref.shape, out.shape
    assert_allclose(
        out.float(),
        ref,
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )


@cuda_only
@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 512])
@pytest.mark.parametrize("q_len", [128, 512])
def test_key_query_all_ones(q_len, kv_len, batch_size, k_len):
    device = "cuda"
    scale = 3
    query = torch.ones((batch_size, q_len, k_len), device=device)
    key = torch.ones((batch_size, kv_len, k_len), device=device)
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    out = xformers.ops.memory_efficient_attention(query, key, value)
    # this should be equivalent to the average over value
    ref = value.mean(1, keepdim=True).expand_as(query)

    assert_allclose(out, ref, atol=1e-5)


def _block_diag_reshape_lse(
    lse: torch.Tensor, q_seqinfo: fmha.attn_bias._SeqLenInfo
) -> torch.Tensor:
    """LSE can be padded, let's remove the padding"""
    parts = []
    for slice, (start, end) in zip(lse.unbind(0), q_seqinfo.intervals()):
        parts.append(slice[:, : end - start])
    return torch.cat(parts, dim=1).unsqueeze(1)


@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_logsumexp(opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv):
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt="BMK"
    )

    _out, lse = xformers.ops.memory_efficient_attention_forward_requires_grad(
        query,
        key,
        value,
        op=op,
        attn_bias=attn_bias,
    )
    attn = (query.float() / k**0.5) @ key.float().transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            tensor_bias = attn_bias.materialize(
                (query.shape[0], 1, query.shape[1], key.shape[1]),
                device=query.device,
                dtype=torch.float32,
            )
        else:
            assert isinstance(attn_bias, torch.Tensor)
            tensor_bias = attn_bias
        if tensor_bias.ndim == 4:
            tensor_bias = tensor_bias.reshape([-1, *tensor_bias.shape[2:]])
        attn = attn + tensor_bias.float()
    ref_lse = attn.logsumexp(-1)
    if isinstance(attn_bias, fmha.attn_bias.BlockDiagonalMask):
        lse = _block_diag_reshape_lse(lse, attn_bias.q_seqinfo)
    assert_allclose(lse[:, 0, : ref_lse.shape[1]], ref_lse, atol=2e-4)


@cuda_only
@pytest.mark.parametrize("op", [fmha.cutlass.FwOp, fmha.flash.FwOp])
def test_logsumexp_mqa(op):
    if not op.is_available():
        pytest.skip("not available")

    dtype = torch.float16
    s = 3
    query = torch.randn([1, 1, 32, 128], dtype=dtype, device="cuda") * s
    key = (torch.randn([1, 16, 1, 128], dtype=dtype, device="cuda") * s).expand(
        -1, -1, 32, -1
    )
    value = (torch.randn([1, 16, 1, 128], dtype=dtype, device="cuda") * s).expand(
        -1, -1, 32, -1
    )
    assert key.stride(2) == 0

    _, lse = xformers.ops.memory_efficient_attention_forward_requires_grad(
        query,
        key,
        value,
        op=op,
    )
    query, key, value = [x[0].transpose(0, 1) for x in [query, key, value]]
    attn = (query.float() / query.shape[-1] ** 0.5) @ key.float().transpose(-2, -1)
    ref_lse = attn.logsumexp(-1)
    assert_allclose(lse[0, :, 0], ref_lse[:, 0], atol=2e-4)


@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@pytest.mark.parametrize("grad_out_contiguous", [False, True])
@parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
def test_backward(
    opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    grad_out_contiguous,
    fmt,
):
    (
        op_bw,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    attn_bias_requires_grad = (
        random.Random(q_len + kv_len * batch_size).randint(0, 1) > 0
    )
    query, key, value, attn_bias = create_tensors(
        *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        attn_bias_requires_grad=attn_bias_requires_grad,
        fmt=fmt,
    )

    # To understand why we do this, check the comment on the
    # `AttentionBwOpBase` class
    scale = None
    if op_bw.SUPPORTS_CUSTOM_SCALE and query.shape[-1] < 32:
        scale = (1 / 32) ** 0.5
    op_fw = (
        sample_random_supported_fw(
            fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias),
            seed=q_len * kv + kv_len * k,
        )
        if op_bw != fmha.cutlass.BwOp
        else fmha.cutlass.FwOp
    )
    qkv = None

    if (
        fmt == "BMHK"
        and query.shape[3] == value.shape[3]
        and query.shape[1] == value.shape[1]
    ):
        qkv = torch.stack([query, key, value], 2)
        qkv.requires_grad_(True)
        # bm3hk -> 3 x bmhk
        query, key, value = xformers.ops.unbind(qkv, 2)
        assert not query.is_contiguous()

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    if not op_bw.supports(fmha.Inputs(query, key, value, attn_bias)):
        pytest.skip("inputs not supported")

    out = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias, scale=scale, op=(op_fw, op_bw)
    )

    grad_out = torch.randn_like(out)
    if grad_out_contiguous is False:
        grad_out = torch.tensor([1.0], dtype=query.dtype, device=device)[
            None, None, :
        ].expand_as(out)

    out.backward(grad_out)

    if qkv is None and op_bw == fmha.cutlass.BwOp:
        assert query.stride() == query.grad.stride()

    grads = []
    if qkv is None:
        grads = [query.grad, key.grad, value.grad]
        query.grad = None
        key.grad = None
        value.grad = None
    else:
        grads = [qkv.grad]
        qkv.grad = None
    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias, clear=True)
        if attn_bias_grad is not None:
            grads.append(attn_bias_grad)

    ref = ref_attention(query, key, value, attn_bias, scale=scale)
    ref.backward(grad_out)

    assert_allclose(
        out.float(),
        ref.float(),
        "fw pass",
        atol=op_fw.ERROR_ATOL[dtype],
        rtol=op_fw.ERROR_RTOL[dtype],
    )

    del out
    del grad_out
    del ref

    atol = op_bw.ERROR_ATOL[dtype]
    rtol = op_bw.ERROR_RTOL[dtype]

    grads_ref = []
    grads_name = []
    if qkv is None:
        assert isinstance(query.grad, torch.Tensor)
        assert isinstance(key.grad, torch.Tensor)
        assert isinstance(value.grad, torch.Tensor)
        grads_ref = [query.grad, key.grad, value.grad]
        grads_name = ["query", "key", "value"]
    else:
        assert isinstance(qkv.grad, torch.Tensor)
        grads_ref = [qkv.grad]
        grads_name = ["qkv"]

    if attn_bias_requires_grad:
        attn_bias_grad = get_bias_grad(attn_bias)
        if attn_bias_grad is not None:
            grads_ref.append(attn_bias.grad)
            grads_name.append("bias")

    del query
    del key
    del value
    del qkv

    assert len(grads_ref) == len(
        grads
    ), "Wrong number of gradients (maybe bias grad didn't backprop?)"
    for name, calc_grad, ref_grad in zip(grads_name, grads, grads_ref):
        assert_allclose(
            calc_grad,
            ref_grad,
            msg=f"{op_fw.NAME}+{op_bw.NAME}:{name}",
            atol=atol,
            rtol=rtol,
        )


def _vec_binom_test(x, n, p):
    """
    vectorized implementation of scipy.stats.binom_test
    this makes our tests much faster
    reference: https://github.com/scipy/scipy/blob/v1.8.0/scipy/stats/_morestats.py#L2609-L2702
    """
    import numpy as np
    from scipy.stats import distributions

    x = np.atleast_1d(x)
    d = distributions.binom.pmf(x, n, p)[:, None]
    rerr = 1 + 1e-7
    # x < p * n case
    i = np.arange(np.ceil(p * n), n + 1)
    y = np.sum(distributions.binom.pmf(i, n, p) <= d * rerr, axis=1)
    pval1 = distributions.binom.cdf(x, n, p) + distributions.binom.sf(n - y, n, p)

    # other case
    i = np.arange(np.floor(p * n) + 1)
    y = np.sum(distributions.binom.pmf(i, n, p) <= d * rerr, axis=1)
    pval2 = distributions.binom.cdf(y - 1, n, p) + distributions.binom.sf(x - 1, n, p)

    pval = np.where(x < p * n, pval1, pval2)
    pval = np.minimum(1.0, pval)
    return pval


def _get_drop_mask(op, batch_size, q_len, kv_len, p, device):
    if op == fmha.cutlass.FwOp:
        mask = torch.empty((batch_size, 1, q_len, kv_len), device=device)
        rand_uniform = torch.ops.xformers._cutlass_rand_uniform(p, mask)
        mask = (rand_uniform > p).to(torch.float32)
        mask = mask.reshape(batch_size, q_len, kv_len)
    else:
        mask = torch.empty((batch_size, q_len, kv_len), device=device)
        mask = torch.ops.xformers._temp_dropout(mask, p)

    return mask


@cuda_only
@pytest.mark.parametrize("attn_bias", [None, fmha.attn_bias.LowerTriangularMask()])
@pytest.mark.parametrize("seed", [42, 124])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [32])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33, 65])
@pytest.mark.parametrize("q_len", [2, 33])
@pytest.mark.parametrize("op", ALL_FW_OPS, ids=list(map(lambda t: t.NAME, ALL_FW_OPS)))
def test_dropout(op, q_len, kv_len, batch_size, k_len, p, seed, attn_bias):
    device = "cuda"
    scale = 3
    query = torch.randn((batch_size, q_len, k_len), device=device) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    inputs_for_support_check = fmha.Inputs(query, key, value, attn_bias, p, None)
    if not op.supports(inputs_for_support_check):
        del query, key, value, attn_bias
        pytest.skip(f"{op.NAME}: unsupported input")

    torch.manual_seed(seed)
    out = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias, p, op=(op, None)
    )

    torch.manual_seed(seed)
    out2 = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias, p, op=(op, None)
    )

    assert_allclose(out, out2, "dropout reproducibility")

    torch.manual_seed(seed)
    mask = _get_drop_mask(op, batch_size, q_len, kv_len, p, device)
    ref = ref_attention(query, key, value, attn_bias, mask, p)
    assert_allclose(out, ref, atol=2e-4), f"{(out - ref).abs().max()}"

    num_trials = 1000
    p_val_tol = 1e-6
    keep_prob = 1 - p
    masks = []
    for i in range(num_trials):
        mask = _get_drop_mask(op, batch_size, q_len, kv_len, p, device)
        masks.append(mask.clone().cpu())
    masks = torch.stack(masks, dim=0)
    p_value = binomtest(int(masks.sum()), masks.numel(), p=keep_prob).pvalue
    assert p_value > p_val_tol, p_value
    masks = masks.sum(0).flatten()
    p_values = _vec_binom_test(masks, num_trials, p=keep_prob)
    assert all(p_values > p_val_tol)


def _test_dropout_backward(q_len, kv_len, batch_size, k, p, op, dtype):
    if dtype is torch.bfloat16 and compute_capability < (8, 0):
        pytest.skip("bf16 requires Sm80")
    if not op.is_available():
        pytest.skip()

    scale = 3
    device = "cuda"
    query = torch.randn((batch_size, q_len, k), device=device, dtype=dtype) * scale
    key = torch.randn((batch_size, kv_len, k), device=device, dtype=dtype) * scale
    value = torch.randn((batch_size, kv_len, k), device=device, dtype=dtype) * scale

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    grad_out = torch.ones_like(query)

    assert op.supports(fmha.Inputs(query=query, key=key, value=value, p=p))

    seed = 42
    torch.manual_seed(seed)
    out = xformers.ops.memory_efficient_attention(query, key, value, p=p, op=(op, None))

    out.backward(grad_out)

    grad_q = query.grad
    grad_k = key.grad
    grad_v = value.grad

    query.grad = None
    key.grad = None
    value.grad = None

    torch.manual_seed(seed)
    mask = _get_drop_mask(op, batch_size, q_len, kv_len, p, device)

    ref = ref_attention(query, key, value, None, mask, p)
    ref.backward(grad_out)

    atol, rtol = (
        fmha.AttentionBwOpBase.ERROR_ATOL[dtype],
        fmha.AttentionBwOpBase.ERROR_RTOL[dtype],
    )
    assert_allclose(
        grad_v,
        value.grad,
        "grad_v",
        atol=atol,
        rtol=rtol,
    )
    # TODO: Investigate why precision is worse
    if dtype in [torch.float16, torch.bfloat16]:
        atol = atol * 2 + 0.15
        rtol = rtol * 2
    assert_allclose(
        grad_q,
        query.grad,
        "grad_q",
        atol=atol,
        rtol=rtol,
    )
    assert_allclose(
        grad_k,
        key.grad,
        "grad_k",
        atol=atol,
        rtol=rtol,
    )


@cuda_only
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33])
@pytest.mark.parametrize("q_len", [2, 33])
def test_dropout_backward_small_k(q_len, kv_len, batch_size, k, p):
    _test_dropout_backward(
        q_len, kv_len, batch_size, k, p, op=fmha.small_k.FwOp, dtype=torch.float32
    )


@cuda_only
@pytest.mark.parametrize("p", [0.000001, 0.3, 0.7])
@pytest.mark.parametrize("k", [16, 128, 256])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kv_len", [3, 248, 256])
@pytest.mark.parametrize("q_len", [3, 248, 256])
@pytest.mark.parametrize("dt", ["f16", "bf16", "f32"])
def test_dropout_backward_cutlass(dt, q_len, kv_len, batch_size, k, p):
    _test_dropout_backward(
        q_len,
        kv_len,
        batch_size,
        k,
        p,
        op=fmha.cutlass.FwOp,
        dtype={"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dt],
    )


@cuda_only
@pytest.mark.parametrize("k_len", [32])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("kv_len", [3 * 32])
@pytest.mark.parametrize("q_len", [3 * 32])
def test_memory_efficient_attention_full_block_masked(q_len, kv_len, batch_size, k_len):
    device = "cuda"
    op_fw = fmha.small_k.FwOp
    op_bw = fmha.small_k.BwOp

    scale = 3
    query = torch.randn((batch_size, q_len, k_len), device=device) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    # in this case, most of the blocks in a row get masked
    attn_bias = torch.full((3, 32), float("-inf"), device=device)
    attn_bias[:2, :4] = 0
    attn_bias = attn_bias.flatten()[None, None, :].expand(1, q_len, -1)

    out = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias, op=(op_fw, op_bw)
    )
    ref = ref_attention(query, key, value, attn_bias)

    assert_allclose(
        out, ref, atol=op_fw.ERROR_ATOL[query.dtype], rtol=op_fw.ERROR_RTOL[query.dtype]
    )

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    grad_out = torch.ones_like(query)

    out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias)
    out.backward(grad_out)

    grad_q = query.grad
    grad_k = key.grad
    grad_v = value.grad

    query.grad = None
    key.grad = None
    value.grad = None

    ref = ref_attention(query, key, value, attn_bias)
    ref.backward(grad_out)

    atol = op_bw.ERROR_ATOL[query.dtype]
    rtol = op_bw.ERROR_RTOL[query.dtype]
    assert_allclose(grad_q, query.grad, "grad_q", atol=atol, rtol=rtol)
    assert_allclose(grad_k, key.grad, "grad_k", atol=atol, rtol=rtol)
    assert_allclose(grad_v, value.grad, "grad_v", atol=atol, rtol=rtol)


@pytest.mark.parametrize("fmt", ["BMK", "BMHK"])
@parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_lowlevel_api_shapes(opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt):
    query, key, value, attn_bias = create_tensors(
        *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt=fmt
    )
    grad_out = torch.ones_like(query)
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    out, lse = xformers.ops.memory_efficient_attention_forward_requires_grad(
        query, key, value, attn_bias
    )
    assert out.ndim == query.ndim
    dq, dk, dv = xformers.ops.memory_efficient_attention_backward(
        grad_out, out, lse, query, key, value, attn_bias
    )
    assert dq.shape == query.shape
    assert dk.shape == key.shape
    assert dv.shape == value.shape


@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_cuda_streams(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
):
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    if device != "cuda":
        pytest.skip("Not CUDA")
    bias_type = None
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = [
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ]
    s_hipri = torch.cuda.Stream(priority=-1)
    s_lopri = torch.cuda.Stream(priority=0)
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt="BMHK"
    )
    torch.cuda.synchronize()
    with torch.cuda.stream(s_lopri):
        torch.cuda._sleep(100_000_000)  # wait 100m cycles
        query *= 2
    s_hipri.wait_stream(s_lopri)
    with torch.cuda.stream(s_hipri):
        # If the kernel is scheduled in the main stream
        # `query * 2` has not been executed yet
        out = xformers.ops.memory_efficient_attention(query, key, value, op=(op, None))
    # Test that `s_lopri` is still sleeping
    # and that `query *= 2` has not been executed yet
    query2_main_stream = query * 2
    torch.cuda.synchronize()
    # TODO: Figure out why this is failing sometimes
    # The sleep timer seems to be high enough already ...
    # assert torch.allclose(query2_main_stream, query), "Need to increase sleep time"
    del query2_main_stream

    ref = ref_attention(query, key, value)
    assert out.shape == ref.shape, out.shape

    assert_allclose(
        out.float(),
        ref.float(),
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )


@parametrize_opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_custom_scale(opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv):
    p = 0.0
    scale = 0.1

    (
        op_bw,
        device,
        dtype,
        _,
        B,
        q_len,
        kv_len,
        H,
        k,
        Kv,
    ) = opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    torch.manual_seed(q_len + kv_len + k)
    if device != "cuda":
        pytest.skip("Not CUDA")

    query, key, value, attn_bias = create_tensors(
        *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt="BMK"
    )
    inputs = fmha.Inputs(
        query=query, key=key, value=value, attn_bias=attn_bias, scale=scale
    )
    op_fw = sample_random_supported_fw(inputs, seed=q_len * k + kv_len * k)
    grad_out = query.new_ones(B * H, q_len, Kv)
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    reasons = op_fw.not_supported_reasons(inputs)
    if reasons:
        pytest.skip(f"{op_fw.NAME}: unsupported ({'/'.join(reasons)})")
    reasons = op_bw.not_supported_reasons(inputs)
    if reasons:
        pytest.skip(f"{op_bw.NAME}: unsupported ({'/'.join(reasons)})")

    # NOTE: we still need to scale the inputs to not blowup
    # the pre-softmax values (numerical stability)
    s = k**-0.5
    out = xformers.ops.memory_efficient_attention(
        query * s, key, value, attn_bias, p, scale, op=(op_fw, op_bw)
    )
    out.backward(grad_out)
    grad_q, grad_k, grad_v = query.grad, key.grad, value.grad
    query.grad = key.grad = value.grad = None

    ref = ref_attention(query * s, key, value, attn_bias, None, p, scale)
    ref.backward(grad_out)
    ref_grad_q, ref_grad_k, ref_grad_v = query.grad, key.grad, value.grad
    query.grad = key.grad = value.grad = None

    atol = op_fw.ERROR_ATOL[dtype]
    rtol = op_fw.ERROR_RTOL[dtype]
    assert_allclose(out.float(), ref.float(), "out", atol=atol, rtol=rtol)
    atol = op_bw.ERROR_ATOL[dtype]
    rtol = op_bw.ERROR_RTOL[dtype]
    assert_allclose(grad_q, ref_grad_q, "grad_q", atol=atol, rtol=rtol)
    assert_allclose(grad_k, ref_grad_k, "grad_k", atol=atol, rtol=rtol)
    assert_allclose(grad_v, ref_grad_v, "grad_v", atol=atol, rtol=rtol)


def apply_attention(query, key, value, attn_bias, op_fw, proj):
    x = xformers.ops.memory_efficient_attention(
        query, key, value, attn_bias=attn_bias, op=(op_fw, None)
    )
    x = proj(x)
    return x


@pytest.mark.parametrize("use_reentrant", [False, True])
@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_grad_checkpointing(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    use_reentrant,
):
    fmt = "BMHK"
    (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    ) = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv
    bias_type = None
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv = (
        op,
        device,
        dtype,
        bias_type,
        batch_size,
        q_len,
        kv_len,
        h,
        k,
        kv,
    )
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt=fmt,
    )
    qkv = None

    if (
        fmt == "BMHK"
        and query.shape[3] == value.shape[3]
        and query.shape[1] == value.shape[1]
    ):
        qkv = torch.stack([query, key, value], 2)
        qkv.requires_grad_(True)
        # bm3hk -> 3 x bmhk
        query, key, value = xformers.ops.unbind(qkv, 2)
        assert not query.is_contiguous()

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    proj = torch.nn.Linear(kv, k, device=device, dtype=dtype)

    x = query
    for _ in range(5):
        x = checkpoint(
            apply_attention,
            x,
            key,
            value,
            attn_bias,
            op,
            proj,
            use_reentrant=use_reentrant,
        )
    x.mean().backward()


ALL_FW_OPS_NO_SMALLK = [op for op in ALL_FW_OPS if op is not fmha.small_k.FwOp]


@pytest.mark.parametrize(
    "op", ALL_FW_OPS_NO_SMALLK, ids=[op.NAME for op in ALL_FW_OPS_NO_SMALLK]
)
def test_unsupported_cpu(op: Type[fmha.AttentionFwOpBase]):
    q = torch.empty([1, 1, 1, 32])
    with pytest.raises(ValueError):
        fmha.memory_efficient_attention(q, q, q, op=(op, None))


@cuda_only
@pytest.mark.parametrize(
    "op", ALL_FW_OPS_NO_SMALLK, ids=[op.NAME for op in ALL_FW_OPS_NO_SMALLK]
)
def test_unsupported_stride_lastdim(op: Type[fmha.AttentionFwOpBase]):
    q = torch.empty([1, 1, 32, 4], device="cuda", dtype=torch.float16).permute(
        0, 3, 1, 2
    )
    try:
        fmha.memory_efficient_attention(q, q, q, op=(op, None))
    except ValueError as e:
        if "Only work on pre-MLIR triton for now" in str(e):
            pytest.skip("Only work on pre-MLIR triton for now")
        q = q.contiguous()
        fmha.memory_efficient_attention(q, q, q, op=(op, None))


@cuda_only
@pytest.mark.parametrize(
    "op", ALL_FW_OPS_NO_SMALLK, ids=[op.NAME for op in ALL_FW_OPS_NO_SMALLK]
)
def test_unsupported_stride_alignment(op: Type[fmha.AttentionFwOpBase]):
    q = torch.empty([1, 2, 1, 33], device="cuda", dtype=torch.float16)[:, :, :, :32]
    try:
        fmha.memory_efficient_attention(q, q, q, op=(op, None))
    except ValueError as e:
        if "Only work on pre-MLIR triton for now" in str(e):
            pytest.skip("Only work on pre-MLIR triton for now")
        q = q.contiguous()
        fmha.memory_efficient_attention(q, q, q, op=(op, None))


@sm75_or_better_only
def test_unsupported_dropout_combine_flash_cutlass() -> None:
    q = torch.empty(
        [1, 4, 1, 16], device="cuda", dtype=torch.float16, requires_grad=True
    )
    with pytest.raises(ValueError):
        out = fmha.memory_efficient_attention(
            q, q, q, p=0.1, op=(fmha.cutlass.FwOp, fmha.flash.BwOp)
        )
        out.backward(out)
    with pytest.raises(ValueError):
        out = fmha.memory_efficient_attention(
            q, q, q, p=0.1, op=(fmha.flash.FwOp, fmha.cutlass.BwOp)
        )
        out.backward(out)


def test_attn_bias_causal() -> None:
    m = -math.inf
    causal_mask = torch.tensor([[0, m], [0, 0], [0, 0]])
    tensor_bias = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    attn_bias = fmha.attn_bias.LowerTriangularMask()
    assert_allclose(attn_bias.materialize(causal_mask.shape), causal_mask, "causal")
    attn_bias = attn_bias.add_bias(tensor_bias)
    assert_allclose(
        attn_bias.materialize(causal_mask.shape),
        tensor_bias + causal_mask,
        "causal+tensor_bias",
    )


def test_attn_bias_torch_tensor() -> None:
    tensor_bias = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    attn_bias = fmha.attn_bias.LowerTriangularMaskWithTensorBias(tensor_bias)
    m = -math.inf
    causal_bias = torch.tensor([[0, m, m], [0, 0, m]])
    assert_allclose(
        attn_bias.materialize((2, 3)), causal_bias + tensor_bias, "tensor_bias+causal"
    )


def test_attn_bias_blockdiag() -> None:
    queries = [
        torch.randn([1, 3, 1, 8]),
        torch.randn([1, 2, 1, 8]),
        torch.randn([1, 5, 1, 8]),
    ]
    attn_bias, q = fmha.BlockDiagonalMask.from_tensor_list(queries)

    # Verify mask
    as_tensor = attn_bias.materialize((10, 10))
    assert int((as_tensor != -math.inf).sum().item()) == 3 * 3 + 2 * 2 + 5 * 5
    assert_allclose(as_tensor[0:3, 0:3], torch.zeros([3, 3]), "batch0")
    assert_allclose(as_tensor[3:5, 3:5], torch.zeros([2, 2]), "batch1")
    assert_allclose(as_tensor[5:, 5:], torch.zeros([5, 5]), "batch2")

    # Verify we can split it back
    queries2 = attn_bias.split(q)
    assert len(queries) == len(queries2)
    for q1, q2 in zip(queries, queries2):
        assert_allclose(q1, q2)


def test_attn_bias_blockdiag_batched() -> None:
    queries = [
        torch.randn([1, 3, 1, 8]),
        torch.randn([3, 2, 1, 8]),
        torch.randn([1, 5, 1, 8]),
    ]
    attn_bias, q = fmha.BlockDiagonalMask.from_tensor_list(queries)

    # Verify mask
    as_tensor = attn_bias.materialize((14, 14))
    assert int((as_tensor != -math.inf).sum().item()) == 3 * 3 + 3 * 2 * 2 + 5 * 5
    assert_allclose(as_tensor[0:3, 0:3], torch.zeros([3, 3]), "batch0")
    assert_allclose(as_tensor[3:5, 3:5], torch.zeros([2, 2]), "batch1.0")
    assert_allclose(as_tensor[5:7, 5:7], torch.zeros([2, 2]), "batch1.1")
    assert_allclose(as_tensor[7:9, 7:9], torch.zeros([2, 2]), "batch1.2")
    assert_allclose(as_tensor[9:, 9:], torch.zeros([5, 5]), "batch2")

    # Verify we can split it back
    queries2 = attn_bias.split(q)
    assert len(queries) == len(queries2)
    for q1, q2 in zip(queries, queries2):
        assert_allclose(q1, q2)


def test_attn_bias_blockdiag_crossattn_causal() -> None:
    # Q / KV have different seqlen
    list_q = [
        torch.randn([1, 3, 1, 8]),
        torch.randn([2, 1, 1, 8]),
    ]
    list_k = [
        torch.randn([1, 2, 1, 8]),
        torch.randn([2, 3, 1, 8]),
    ]

    attn_bias, q, k, _ = fmha.attn_bias.BlockDiagonalMask.from_tensor_lists_qkv(
        list_q, list_k
    )

    # Verify mask
    as_tensor = attn_bias.materialize((q.shape[1], k.shape[1]))
    assert int((as_tensor != -math.inf).sum().item()) == 3 * 2 + 2 * 3 * 1
    assert_allclose(as_tensor[0:3, 0:2], torch.zeros([3, 2]), "batch0")
    assert_allclose(as_tensor[3:4, 2:5], torch.zeros([1, 3]), "batch1.0")
    assert_allclose(as_tensor[4:, 5:], torch.zeros([1, 3]), "batch1.1")

    # Also test causal version
    as_tensor = attn_bias.make_causal().materialize((q.shape[1], k.shape[1]))
    assert_allclose(
        as_tensor[3:4, 2:5],
        fmha.attn_bias.LowerTriangularMask().materialize((1, 3)),
        "batch1.0[causal]",
    )

    # Verify we can split it back
    list_q2 = attn_bias.split_queries(q)
    assert len(list_q) == len(list_q2)
    for q1, q2 in zip(list_q, list_q2):
        assert_allclose(q1, q2)
    with pytest.raises(ValueError):
        attn_bias.split_queries(k)
    list_k2 = attn_bias.split_kv(k)
    assert len(list_k) == len(list_k2)
    for k1, k2 in zip(list_k, list_k2):
        assert_allclose(k1, k2)


def test_attn_bias_blockdiag_crossattn_causal_with_prefix_qk_cond() -> None:
    list_q = [
        torch.randn([1, 3, 1, 8]),
    ]
    list_k = [
        torch.randn([1, 2, 1, 8]),
    ]
    attn_bias, q, k, _ = fmha.attn_bias.BlockDiagonalMask.from_tensor_lists_qkv(
        list_q, list_k
    )
    with pytest.raises(ValueError):
        attn_bias.make_causal_from_bottomright()


def test_attn_bias_blockdiag_crossattn_causal_with_prefix() -> None:
    # Q / KV have different seqlen
    list_q = [
        torch.randn([1, 2, 1, 8]),
        torch.randn([2, 2, 1, 8]),
    ]
    list_k = [
        torch.randn([1, 2, 1, 8]),
        torch.randn([2, 5, 1, 8]),
    ]

    attn_bias, q, k, _ = fmha.attn_bias.BlockDiagonalMask.from_tensor_lists_qkv(
        list_q, list_k
    )
    as_tensor = attn_bias.make_causal_from_bottomright().materialize(
        (q.shape[1], k.shape[1])
    )
    m = -math.inf
    assert_allclose(
        as_tensor[0:2, 0:2],
        torch.tensor([[0, m], [0, 0]], dtype=torch.float32),
        "batch1.1[causal_with_prefix]",
    )
    assert_allclose(
        as_tensor[2:4, 2:7],
        torch.tensor([[0, 0, 0, 0, m], [0, 0, 0, 0, 0]], dtype=torch.float32),
        "batch2.1[causal_with_prefix]",
    )
    assert_allclose(
        as_tensor[4:6, 7:12],
        torch.tensor([[0, 0, 0, 0, m], [0, 0, 0, 0, 0]], dtype=torch.float32),
        "batch2.2[causal_with_prefix]",
    )


@cuda_only
def test_attn_bias_padded() -> None:
    bsize, n_heads, d, padding = 8, 3, 8, 32

    # Q / KV have different seqlen
    k = torch.randn((bsize, padding, n_heads, d), device="cuda", dtype=torch.float16)
    k_seqlen = [5, 8, 7, 1, 9, 3, 12, 32]
    other = bsize - 1
    v = torch.randn((bsize, padding, n_heads, d), device="cuda", dtype=torch.float16)
    n_q_first = 4
    q = [
        torch.randn((1, n_q_first, n_heads, d), device="cuda", dtype=torch.float16),
        torch.randn((1, other, n_heads, d), device="cuda", dtype=torch.float16),
    ]
    q_cat = torch.cat([x.view(1, -1, n_heads, d) for x in q], dim=1)
    q_seqlen = [n_q_first] + [1] * other

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=q_seqlen,
        kv_seqlen=k_seqlen,
        kv_padding=padding,
    )

    v = v.view(1, -1, n_heads, d)
    k = k.view(1, -1, n_heads, d)

    scores = (q_cat.transpose(1, 2) @ k.transpose(1, 2).transpose(2, 3)).float()
    assert not scores.isnan().any()
    mask = torch.full_like(scores, -float("inf"))
    for i, (slen, qlen) in enumerate(zip(k_seqlen, q_seqlen)):
        kseq_start = i * padding
        qstart = sum(q_seqlen[:i])
        mask[:, :, qstart : qstart + qlen, kseq_start : kseq_start + slen] = torch.triu(
            mask[:, :, qstart : qstart + qlen, kseq_start : kseq_start + slen].float(),
            diagonal=1 + slen - qlen,
        ).float()

    scores += mask
    assert not scores.isnan().any()
    # 1,3,10,8 @ 1,3,8,256 -> 1,3,10,256
    scores = torch.nn.functional.softmax(scores, -1).half()
    # torch.Size([1, 3, 3, 32]) @ torch.Size([1, 3, 32, 8])
    output = scores @ v.transpose(1, 2)  # 1,3,10,256 @ 1,3,256, 8 -> 1,3,10,8
    output = output.transpose(1, 2).contiguous()

    fmha_output = fmha.memory_efficient_attention_forward(
        q_cat, k, v, attn_bias, scale=1.0
    )

    # assert torch.allclose(output, fmha_output)
    assert_allclose(
        output,
        fmha_output,
        atol=fmha.cutlass.FwOp.ERROR_ATOL[torch.float16],
        rtol=fmha.cutlass.FwOp.ERROR_RTOL[torch.float16],
    )


def _kv_heads_label(kv_heads: Optional[int]) -> str:
    if kv_heads is None:
        return ""
    if kv_heads == 1:
        return "mq"
    return f"gqa{kv_heads}"


@sm70_or_better_only
@pytest.mark.parametrize(
    "op",
    [
        fmha.decoder.FwOp,
    ],
)
@pytest.mark.parametrize("kv_heads", [None, 1, 2], ids=_kv_heads_label)
@pytest.mark.parametrize("bsz,n_heads", [(1, 1), (1, 16), (1, 32), (8, 1), (4, 8)])
@pytest.mark.parametrize("padding", [32, 4096])
@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32"])
def test_decoder(
    op,
    n_heads: int,
    kv_heads: Optional[int],
    padding: int,
    bsz: int,
    dtype: str,
    dequant: bool = False,
    num_queries: int = 1,
) -> None:
    # kv_heads = 1: multiquery
    # kv_heads = None: neither MQA nor GQA
    # kv_heads > 1: BMGHK
    if dtype == "bf16" and compute_capability < (8, 0):
        raise pytest.skip("BF16 is only supported on SM80+")
    dtype_ = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]
    torch.manual_seed(1)
    d = 128
    if kv_heads is not None and kv_heads > 1:
        k_shape: Tuple[int, ...] = (1, bsz * padding, kv_heads, n_heads, d)
        q_shape: Tuple[int, ...] = (
            1,
            bsz * num_queries,
            kv_heads,
            n_heads,
            d,
        )
    else:
        k_shape = (1, bsz * padding, n_heads, d)
        q_shape = (1, bsz * num_queries, n_heads, d)

    # TODO: support 2 kv heads etc.
    k = torch.randn(k_shape, dtype=dtype_, device="cuda")
    k_seqlen = torch.randint(num_queries, padding + 1, (bsz,)).tolist()
    v = torch.randn(k_shape, dtype=dtype_, device="cuda")
    q = torch.randn(q_shape, dtype=dtype_, device="cuda")

    if dequant:
        k_shape = k_shape[:-1] + (d // 8 + op.NUM_GROUPS,)
        k = torch.zeros(k_shape, dtype=torch.int32, device="cuda")
        k.random_()
        k[..., : op.NUM_GROUPS].view(torch.float16).fill_(1.0)
        v = torch.zeros(k_shape, dtype=torch.int32, device="cuda")
        v.random_()
        v[..., : op.NUM_GROUPS].view(torch.float16).fill_(1.0)

    if kv_heads is not None:
        k = k[..., :1, :].expand(k_shape)
        v = v[..., :1, :].expand(k_shape)

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[num_queries] * bsz,
        kv_seqlen=k_seqlen,
        kv_padding=padding,
    )

    decoder_output = fmha.memory_efficient_attention_forward(
        q,
        k,
        v,
        attn_bias,
        op=op,
    )

    def dequant_cache(x):
        x = x[..., op.NUM_GROUPS :, None].expand(k_shape[:-1] + (d // 8, 8))
        x = x // (2 ** (4 * torch.arange(8, device="cuda")))
        x = (x % 16).flatten(start_dim=-2)
        return x.to(dtype_) + 1.0

    if dequant:
        k = dequant_cache(k)
        v = dequant_cache(v)

    cutlass_output = fmha.memory_efficient_attention_forward(
        q, k, v, attn_bias, op=fmha.cutlass.FwOp
    )
    assert_allclose(
        decoder_output,
        cutlass_output,
        atol=fmha.cutlass.FwOp.ERROR_ATOL[dtype_] * 4,
        rtol=fmha.cutlass.FwOp.ERROR_RTOL[dtype_],
    )


@sm80_or_better_only
@pytest.mark.skipif(
    fmha.triton_splitk.FwOp_S2.OPERATOR is None, reason="splitK disabled"
)
@pytest.mark.parametrize(
    "op,dequant,dtype",
    [
        (fmha.triton_splitk.FwOp_S2, False, "bf16"),
        (fmha.triton_splitk.FwOp_S2, False, "f16"),
        (fmha.triton_splitk.FwOp_S2, True, "bf16"),
        (
            type(
                "S2_8", (fmha.triton_splitk.FwOp_S2,), {"NUM_GROUPS": 8, "NAME": "S2_8"}
            ),
            True,
            "bf16",
        ),
    ],
)
@pytest.mark.parametrize("kv_heads", [None, 1, 2], ids=_kv_heads_label)
@pytest.mark.parametrize("n_heads", [16])
@pytest.mark.parametrize("padding, bsz", [(32, 8), (4096, 1)])
def test_triton_splitk_decoder(
    op,
    dequant: bool,
    kv_heads: Optional[int],
    n_heads: int,
    padding: int,
    bsz: int,
    dtype: str,
) -> None:
    if dequant:
        pytest.skip("dequant is not supported")

    # We omit dequant with f16: it needs a very high tol
    test_decoder(
        op,
        kv_heads=kv_heads,
        n_heads=n_heads,
        padding=padding,
        bsz=bsz,
        dtype=dtype,
        dequant=dequant,
    )


def test_attn_bias_from_seqlens() -> None:
    bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens([3, 5, 1])
    out = bias.split(torch.randn([1, 3 + 5 + 1, 16]))
    assert len(out) == 3
    assert tuple(out[0].shape) == (1, 3, 16)


@cuda_only
def test_attn_bias_blockdiag_doc() -> None:
    """IMPORTANT:
    This is the example in the doc for `BlockDiagonalMask`.
    If this example needs to be updated, please also update the doc
    """
    import torch

    from xformers.ops import fmha

    K = 16
    dtype = torch.float16
    device = "cuda"
    list_x = [
        torch.randn([1, 3, 1, K], dtype=dtype, device=device),
        torch.randn([1, 6, 1, K], dtype=dtype, device=device),
        torch.randn([1, 2, 1, K], dtype=dtype, device=device),
    ]
    attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(list_x)

    linear = torch.nn.Linear(K, K * 3).to(device=device, dtype=dtype)  # type: ignore

    q, k, v = linear(x).reshape([1, -1, 1, 3, K]).unbind(-2)
    out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
    list_out = attn_bias.split(out)
    assert tuple(list_out[0].shape) == (1, 3, 1, K)


@cuda_only
class TestAttnBias:
    @staticmethod
    def create_tensors(
        dtype,
        B: int = 2,
        Mq: int = 32,
        Mkv: int = 32,
        H: int = 3,
        K: int = 16,
        Kv: int = 16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.randn([B, Mq, H, K], device="cuda", dtype=dtype) * 3,
            torch.randn([B, Mkv, H, K], device="cuda", dtype=dtype) * 3,
            torch.randn([B, Mkv, H, Kv], device="cuda", dtype=dtype) * 3,
            torch.randn([B, H, Mq, Mkv], device="cuda", dtype=dtype) * 3,
        )

    @staticmethod
    def pad_bias(bias: torch.Tensor) -> torch.Tensor:
        align_to = 16
        if (bias.shape[-1] % align_to) == 0:
            return bias
        pad_count = align_to - (bias.shape[-1] % align_to)
        return torch.nn.functional.pad(bias, [0, pad_count])[:, :, :, : bias.shape[-1]]

    def test_f16_biasf32(self) -> None:
        q, k, v, bias = self.create_tensors(torch.float16)
        fmha.memory_efficient_attention(q, k, v, attn_bias=bias)
        bias = bias.to(torch.float32)
        with pytest.raises((ValueError, RuntimeError)):
            fmha.memory_efficient_attention(q, k, v, attn_bias=bias)

    def test_f32_biasf16(self) -> None:
        q, k, v, bias = self.create_tensors(torch.float32)
        fmha.memory_efficient_attention(q, k, v, attn_bias=bias)
        bias = bias.to(torch.float16)
        with pytest.raises((ValueError, RuntimeError)):
            fmha.memory_efficient_attention(q, k, v, attn_bias=bias)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_wrong_alignment(self, dtype) -> None:
        op = fmha.cutlass.FwOp
        q, k, v, bias = self.create_tensors(dtype, Mq=7, Mkv=5)
        try:
            fmha.memory_efficient_attention(q, k, v, attn_bias=bias, op=(op, None))
            return
        except (ValueError, RuntimeError):
            pass
        # This case is not supported, likely due to padding issues
        # Let's make sure it works with padding
        assert bias.ndim == 4, bias.shape
        bias_padded = self.pad_bias(bias)
        out = fmha.memory_efficient_attention(
            q, k, v, attn_bias=bias_padded, op=(op, None)
        ).float()
        ref_out = ref_attention_bmhk(q, k, v, bias)
        assert_allclose(
            out, ref_out, atol=op.ERROR_ATOL[dtype], rtol=op.ERROR_RTOL[dtype]
        )

    def test_permuted_attn_bias(self) -> None:
        op = fmha.cutlass.FwOp
        dtype = torch.float16
        q, k, v, bias = self.create_tensors(dtype, Mq=7, Mkv=7)
        bias = bias.transpose(-1, -2)  # now `stride(-1) != 1`
        # Either it works, or it raises an exception
        # but we should never get a CUDA error
        try:
            out = fmha.memory_efficient_attention(
                q, k, v, attn_bias=bias, op=(op, None)
            ).float()
            ref_out = ref_attention_bmhk(q, k, v, bias)
            assert_allclose(
                out, ref_out, atol=op.ERROR_ATOL[dtype], rtol=op.ERROR_RTOL[dtype]
            )
        except (ValueError, RuntimeError):
            pass


SM_AND_SHMEM_KBYTES = [
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
    (50, 64),
    (60, 64),
    (70, 96),
    (75, 64),
    (80, 163),
    (86, 99),
    (89, 99),
    # (90, 227),
]


@cuda_only
@pytest.mark.parametrize("dtype_str", ["f32", "f16", "bf16"])
@pytest.mark.parametrize(
    "sm_shmem",
    SM_AND_SHMEM_KBYTES,
    ids=[f"cc{sm}_shmem{shmem}kb" for sm, shmem in SM_AND_SHMEM_KBYTES],
)
def test_has_kernel_for(sm_shmem: Tuple[int, int], dtype_str: str) -> None:
    dtype = {"f32": torch.float, "f16": torch.half, "bf16": torch.bfloat16}[dtype_str]
    sm, shmem_kbytes = sm_shmem
    if sm < 80 and dtype_str == "bf16":
        return

    for k in [16, 32, 64, 128, 256]:
        assert torch.ops.xformers._has_cutlassF_kernel_for(
            dtype, sm, shmem_kbytes * 1024, k
        ), f"k={k}"
        assert torch.ops.xformers._has_cutlassB_kernel_for(
            dtype, sm, shmem_kbytes * 1024, k
        ), f"k={k}"


def test_window_size_materialize() -> None:
    seqlens = [4, 6]
    attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(
        q_seqlen=seqlens,
        kv_seqlen=seqlens,
    ).make_local_attention(2)
    mask = attn_bias.materialize(
        (1, 1, sum(seqlens), sum(seqlens)),
        device="cpu",
        dtype=torch.float32,
    )
    true_mask = torch.log(
        torch.Tensor(
            [
                [
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    ]
                ]
            ]
        )
    )
    assert torch.all(mask == true_mask)


@cuda_only
@pytest.mark.parametrize(
    "opFW_biasT",
    [
        (op, biasT)
        for op in ALL_FW_OPS
        for biasT in op.SUPPORTED_ATTN_BIAS_TYPES
        if op.SUPPORTS_BMGHK
    ],
)
def test_forward_gqa(opFW_biasT):
    opFW, biasT = opFW_biasT
    B_Mq_Mkv_H_K_Kv = (3, 512, 512, 16, 128, 128)
    test_forward(
        (
            opFW,
            "cuda",
            torch.float16,
            biasT,
            *B_Mq_Mkv_H_K_Kv,
        ),
        packed=False,
        fmt="BMGHK",
        g=2,
    )


@cuda_only
@pytest.mark.parametrize(
    "opBW",
    [
        fmha.flash.BwOp,
        fmha.cutlass.BwOp,
    ],
)
def test_backward_gqa(opBW):
    H = 8
    B_Mq_Mkv_H_K_Kv = (3, 512, 512, H, 128, 128)
    dtype = torch.float16
    query, key, value, attn_bias = create_tensors(
        *(opBW, "cuda", dtype, type(None), *B_Mq_Mkv_H_K_Kv),
        attn_bias_requires_grad=False,
        fmt="BMHK",
    )
    op = (fmha.cutlass.FwOp, opBW)
    key = key[:, :, :1].expand(-1, -1, H, -1)
    value = value[:, :, :1].expand(-1, -1, H, -1)
    key.requires_grad_(True)
    out = fmha.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
    out_ref = ref_attention_bmhk(query, key, value, attn_bias=attn_bias)
    assert_allclose(
        out.float(),
        out_ref.float(),
        atol=op[0].ERROR_ATOL[dtype],
        rtol=op[0].ERROR_RTOL[dtype],
    )
    out.backward(query)
    dk = key.grad
    key.grad = None
    out_ref.backward(query)
    assert_allclose(
        dk.float(),
        key.grad.float(),
        atol=op[1].ERROR_ATOL[dtype],
        rtol=op[1].ERROR_RTOL[dtype],
    )


@cuda_only
@pytest.mark.parametrize("opFW", [op for op in ALL_FW_OPS if op.SUPPORTS_BMGHK])
def test_forward_gqa_one_group(opFW):
    dtype = torch.float16
    B, Mq, Mkv, H, K = 3, 13, 16, 5, 128
    q = torch.randn([B, Mq, 1, H, K], dtype=dtype, device="cuda") * 3
    k = torch.randn([B, Mkv, 1, H, K], dtype=dtype, device="cuda") * 3
    v = torch.randn([B, Mkv, 1, H, K], dtype=dtype, device="cuda") * 3

    supported = opFW.supports(fmha.Inputs(q, k, v))
    if not supported:
        supported_bmhk = opFW.supports(fmha.Inputs(q[:, :, 0], k[:, :, 0], v[:, :, 0]))
        assert supported == supported_bmhk
        pytest.skip("not supported")
    out = fmha.memory_efficient_attention_forward(q, k, v, op=opFW)
    ref = ref_attention(q, k, v)
    assert_allclose(
        out.float(),
        ref,
        atol=opFW.ERROR_ATOL[dtype],
        rtol=opFW.ERROR_RTOL.get(dtype, 1e-5),
    )


@sm80_or_better_only
def test_flash_gqa_wrong_strides() -> None:
    op = (fmha.flash.FwOp, None)
    device = "cuda"
    B, Mq, Mkv, G, H, K = 3, 1, 512, 2, 8, 128
    q = torch.empty((B, Mq, G, H, K), dtype=torch.float16, device=device)
    kv = torch.empty((B, Mkv, G, H, K), dtype=torch.float16, device=device)
    fmha.memory_efficient_attention(q, kv, kv, op=op)

    kv = torch.empty((B, Mkv, H, G, K), dtype=torch.float16, device=device).permute(
        0, 1, 3, 2, 4
    )
    with pytest.raises(ValueError):
        fmha.memory_efficient_attention(q, kv, kv, op=op)

    kv = torch.empty((B, Mkv, G, 1, K), dtype=torch.float16, device=device)
    with pytest.raises(ValueError):
        fmha.memory_efficient_attention(q, kv, kv, op=op)
    kv = kv.expand(-1, -1, -1, H, K)
    fmha.memory_efficient_attention(q, kv, kv, op=op)

    kv = torch.empty((B, Mkv, G, H, 2 * K), dtype=torch.float16, device=device)[
        :, :, :, :, :K
    ]
    fmha.memory_efficient_attention(q, kv, kv, op=op)


def _dispatches_to_splitK(q, kv):
    return (
        _dispatch_fw_priority_list(fmha.Inputs(q, kv, kv), False)[0]
        is fmha.triton_splitk.FwOp
    )


def _dispatches_to_flash_decoding(q, kv):
    return (
        _dispatch_fw_priority_list(fmha.Inputs(q, kv, kv), False)[0] is fmha.flash.FwOp
    )


def test_dispatch_decoding_bmhk() -> None:
    assert not _dispatches_to_splitK(
        torch.empty([1, 8, 1, 128]), torch.empty([1, 2048, 1, 128])
    ), "Should not use SplitK with 1 head (no tensorcores)"
    assert _dispatches_to_flash_decoding(
        torch.empty([1, 8, 32, 128]),
        torch.empty([1, 2048, 1, 128]).expand(-1, -1, 32, -1),
    ), "Should use Flash-Decoding with BMHK MQA"
    assert not _dispatches_to_splitK(
        torch.empty([1, 8, 32, 128]),
        torch.empty([1, 2048, 32, 128]),
    ), "Should not use SplitK when no TensorCores"
    assert not _dispatches_to_splitK(
        torch.empty([1, 128, 32, 128]),
        torch.empty([1, 2048, 1, 128]).expand(-1, -1, 32, -1),
    ), "Should not use SplitK if q seqlen is long"
    assert not _dispatches_to_splitK(
        torch.empty([128, 8, 32, 128]),
        torch.empty([128, 2048, 1, 128]).expand(-1, -1, 32, -1),
    ), "Should not use SplitK if B is big"


def test_dispatch_decoding_bmghk() -> None:
    assert not _dispatches_to_splitK(
        torch.empty([1, 8, 1, 1, 128]), torch.empty([1, 2048, 1, 1, 128])
    ), "Should not use SplitK with 1 head (no tensorcores)"
    assert _dispatches_to_flash_decoding(
        torch.empty([1, 8, 1, 32, 128]),
        torch.empty([1, 2048, 1, 1, 128]).expand(-1, -1, -1, 32, -1),
    ), "Should use Flash-Decoding with MQA"
    assert _dispatches_to_flash_decoding(
        torch.empty([1, 8, 4, 32, 128]),
        torch.empty([1, 2048, 4, 1, 128]).expand(-1, -1, -1, 32, -1),
    ), "Should use Flash-Decoding with GQA"
    assert not _dispatches_to_splitK(
        torch.empty([1, 8, 1, 32, 128]),
        torch.empty([1, 2048, 1, 32, 128]),
    ), "Should not use SplitK when no TensorCores"
    assert not _dispatches_to_splitK(
        torch.empty([1, 128, 1, 32, 128]),
        torch.empty([1, 2048, 1, 1, 128]).expand(-1, -1, -1, 32, -1),
    ), "Should not use SplitK if q seqlen is long"
    assert not _dispatches_to_splitK(
        torch.empty([128, 8, 1, 32, 128]),
        torch.empty([128, 2048, 1, 1, 128]).expand(-1, -1, -1, 32, -1),
    ), "Should not use SplitK if B is big"


shapes_triton_splitk = [
    (1, 8, 2**16, 1, 128, 128),
    (1, 4, 2**16, 1, 128, 128),
    (1, 16, 2**16, 1, 128, 128),
    (1, 16, 2**16, 1, 32, 32),
    (1, 8, 1025, 1, 128, 128),
    (2, 8, 4096, 1, 128, 128),
    (10, 8, 2**16, 1, 128, 128),
    (10, 15, 2**16, 1, 128, 128),
    (1, 3, 2**16, 1, 128, 128),
    (1, 3, 2**16 - 10, 1, 128, 128),
    (2, 3, 73, 1, 128, 128),
    (2, 7, 7328, 1, 128, 128),
    (2, 7, 7328, 1, 120, 120),
    (2, 7, 63, 1, 120, 120),
]
op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_splitk = [
    (fmha.triton_splitk.FwOp, "cuda", torch.float16, type(None), *s)
    for s in shapes_triton_splitk
] + [
    (fmha.triton_splitk.FwOp, "cuda", torch.bfloat16, type(None), *s)
    for s in shapes_triton_splitk
]


@pytest.mark.parametrize(
    "opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv",
    op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_splitk,
    ids=[make_id(*c) for c in op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv_splitk],
)
@cuda_only
def test_forward_splitk(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
    packed=False,
    fmt="BMHK",
):
    test_forward(opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, packed=packed, fmt=fmt)


@cuda_only
@pytest.mark.parametrize("op", [fmha.triton_splitk.FwOp])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "B_Mkv_H_K",
    [
        (1, 2**16, 3, 128),
        (5, 53, 4, 64),
    ],
)
def test_mqa_decoding(op: Type[fmha.AttentionFwOpBase], dtype, B_Mkv_H_K):
    B, Mkv, H, K = B_Mkv_H_K
    q = torch.randn([B, 1, H, K], dtype=dtype, device="cuda") * 3
    k = torch.randn([B, Mkv, 1, K], dtype=dtype, device="cuda") * 3
    v = torch.randn([B, Mkv, 1, K], dtype=dtype, device="cuda") * 3
    k = k.expand(-1, -1, H, -1)
    v = v.expand(-1, -1, H, -1)

    if not op.supports(fmha.Inputs(q, k, v)):
        pytest.skip("not supported")
    out = fmha.memory_efficient_attention_forward(q, k, v, op=op)
    ref = ref_attention(q, k, v)
    assert_allclose(
        out.float(),
        ref,
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )


@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_empty_tensors_empty_query(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
):
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt="BMHK",
    )
    opFW = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv[0]

    query = query[:, :0]
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    out = xformers.ops.memory_efficient_attention(query, key, value, op=(opFW, None))
    assert out.shape[1] == 0
    out.backward(out)
    # dK/dV should be all zeros
    assert_allclose(key.grad, torch.zeros_like(key.grad), "key.grad")
    assert_allclose(value.grad, torch.zeros_like(value.grad), "value.grad")


@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_empty_tensors_empty_kv(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
):
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt="BMHK",
    )
    opFW = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv[0]

    key = key[:, :0]
    value = value[:, :0]
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    out = xformers.ops.memory_efficient_attention(query, key, value, op=(opFW, None))
    assert_allclose(out, torch.zeros_like(out), "out")
    out.backward(out)
    # dQ should be all zeros
    assert_allclose(query.grad, torch.zeros_like(query.grad), "query.grad")


@parametrize_opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv__xs
def test_empty_tensors_empty_b(
    opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
):
    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt="BMHK",
    )
    opFW = opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv[0]

    query, key, value = query[:0], key[:0], value[:0]
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    out = xformers.ops.memory_efficient_attention(query, key, value, op=(opFW, None))
    out.backward(out)


def test_local_attn_bias() -> None:
    mask = (
        fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=1, window_right=2)
        .materialize(shape=(4, 4))
        .exp()
    )

    expected = torch.tensor(
        [[1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.float32
    )
    assert (mask == expected).all().item()


@cuda_only
@pytest.mark.parametrize("cc", [60, 70, 80])
@pytest.mark.parametrize("maxK", [32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "custom_mask_type",
    [
        fmha.cutlass._CustomMaskType.NoCustomMask,
        fmha.cutlass._CustomMaskType.CausalFromTopLeft,
        fmha.cutlass._CustomMaskType.CausalFromBottomRight,
    ],
)
@pytest.mark.parametrize("window_size", [0, 3, 300])
@pytest.mark.parametrize(
    "num_queries,num_keys",
    [
        (30, 66),
        (256, 256),
        # Edge cases
        (314, 320),
        (32, 256),
        (224, 226),
        (5, 531),
        (320, 332),  # for win_size=300
        # Others
        (256, 62),
        (256, 63),
        (256, 64),
        (256, 65),
        (256, 66),
    ],
)
def test_cutlassB_iter_order(
    dtype,
    cc: int,
    maxK: int,
    num_queries: int,
    num_keys: int,
    custom_mask_type,
    window_size,
) -> None:
    """
    This tests some internals of the cutlassB kernel
    We test the iteration across blocks of [queries, keys] to ensure
    that we correctly:
    * Iterate over all the blocks that should be iterated
    * Do *not* iterate over blocks that are completely masked out
    * Correctly compute the number of parallel blocks that will compute
        the same block of dQ
    .. and we test this across variable causal masks+local attention combinations
    """
    if (
        window_size > 0
        and custom_mask_type == fmha.cutlass._CustomMaskType.NoCustomMask
    ):
        pytest.skip("LocalAttention is only supported for causal")
    get_iteration_data = partial(
        torch.ops.xformers._cutlassB_iteration_data,
        dtype=dtype,
        cc=cc,
        maxK=maxK,
        num_queries=num_queries,
        num_keys=num_keys,
        custom_mask_type=custom_mask_type,
        window_size=window_size,
    )
    bias = torch.zeros([num_queries, num_keys], dtype=torch.float32)
    if custom_mask_type != fmha.cutlass._CustomMaskType.NoCustomMask:
        bias = fmha.attn_bias._materialize_causal_mask(
            (num_queries, num_keys),
            dtype=torch.float32,
            device="cpu",
            window_size=None if window_size == 0 else window_size,
            from_bottomright=(
                custom_mask_type == fmha.cutlass._CustomMaskType.CausalFromBottomRight
            ),
        )

    block_queries, block_keys = get_iteration_data()[:2]
    mask_pooled = (
        F.max_pool2d(bias.unsqueeze(0), (block_queries, block_keys), ceil_mode=True)
        == 0
    ).int()[0]
    attn_computed = torch.zeros_like(mask_pooled)
    for key_start in range(0, num_keys, block_keys):
        it = 0
        new_key_start = key_start
        new_query_start = get_iteration_data(key_start=key_start)[2]
        try:
            expected_first_query = (
                mask_pooled[:, key_start // block_keys].tolist().index(1)
                * block_queries
            )
            assert (
                new_query_start == expected_first_query
            ), f"Wrong first query for K={key_start}: {new_query_start} (expected {expected_first_query})"
        except ValueError:  # Nothing to compute in this column
            pass

        while new_key_start == key_start and new_query_start < num_queries:
            query_start = new_query_start
            attn_computed[query_start // block_queries, key_start // block_keys] += 1
            # print(f"Compute [{query_start}, {key_start}]")

            # Is there something to compute here?
            assert mask_pooled[
                query_start // block_queries, key_start // block_keys
            ].item(), "Computing a block that is not needed!"
            new_query_start, new_key_start = get_iteration_data(
                key_start=key_start, query_start=query_start
            )[3:5]
            it += 1
            assert it < num_queries, ""
        assert (attn_computed == mask_pooled)[
            :, key_start // block_keys
        ].all(), "some blocks were not computed!"

    # Now check that the number returned by `getNumParallelBlocksForQuery` is correct
    for query_start in range(0, num_queries, block_queries):
        num_parallel_blocks = get_iteration_data(
            query_start=query_start, num_splits_key=num_keys
        )[5]
        num_actual = mask_pooled[query_start // block_queries].sum().item()
        assert num_parallel_blocks == num_actual


# end of file
