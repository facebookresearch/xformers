# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import random
from contextlib import nullcontext
from typing import Any, List, Optional, Sequence, Tuple, Type, TypeVar

import pytest
import torch

try:
    from mtia.re.re_unittest_lib import init_mtia_device

    init_mtia_device()
except ImportError:
    # Failed to load MTIA libraries, so just keep going without MTIA devices
    pass

import xformers.ops
from scipy.stats import binomtest
from torch.utils.checkpoint import checkpoint
from xformers.attn_bias_utils import create_attn_bias, pack_kv_cache
from xformers.ops import fmha
from xformers.ops.fmha import ALL_BW_OPS, ALL_FW_OPS
from xformers.ops.fmha.common import (
    AttentionFwOpBase,
    AttentionOpBase,
    pack_fp8_tensorwise_per_head,
)
from xformers.ops.fmha.dispatch import _dispatch_fw_priority_list

from .utils import (
    assert_allclose,
    construct_fp8_attention_inputs,
    cuda_only,
    cuda_or_mtia_only,
    disable_on_mtia,
    disable_on_rocm,
    disable_tf32,
    ref_attention_bmhk_for_test,
    ref_attention_for_test,
    rocm_only,
    use_cpu_ref,
)

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
sm70_or_better_only = pytest.mark.skipif(
    torch.version.cuda is not None and compute_capability < (7, 0),
    reason="requires sm70+",
)
sm75_or_better_only = pytest.mark.skipif(
    torch.version.cuda is not None and compute_capability < (7, 5),
    reason="requires sm75+",
)
sm80_or_better_only = pytest.mark.skipif(
    torch.version.cuda is not None and compute_capability < (8, 0),
    reason="requires sm80+",
)
sm90_or_better_only = pytest.mark.skipif(
    compute_capability < (9, 0),
    reason="requires sm90+",
)
sm100_or_better_only = pytest.mark.skipif(
    compute_capability < (10, 0), reason="requires sm100+"
)
skip_if_sm100_or_better = pytest.mark.skipif(
    compute_capability >= (10, 0), reason="not supported on Blackwell"
)

skip_if_rocm = pytest.mark.skipif(
    torch.version.hip is not None, reason="not supported on ROCm"
)
_devices = ["cpu"]
_devices += ["cuda"] if torch.cuda.is_available() else []

try:
    import mtia.host_runtime.torch_mtia.dynamic_library  # noqa

    # torch.mtia.is_available() will not work here, since test collection can be done
    # on a machine without MTIA devices
    _devices.append("mtia")
except (ImportError, OSError):
    # Failed to load MTIA libraries, so just keep going without MTIA devices
    pass

T = TypeVar(
    "T", Type[fmha.common.AttentionFwOpBase], Type[fmha.common.AttentionBwOpBase]
)

logger = logging.getLogger("xformers")


def _filter_unsupported_ops(ops: Sequence[T]) -> List[T]:
    return [
        op
        for op in ops
        if (
            "cpu" in op.SUPPORTED_DEVICES
            or "mtia" in op.SUPPORTED_DEVICES
            or (
                op.CUDA_MINIMUM_COMPUTE_CAPABILITY <= compute_capability
                and (
                    op.CUDA_MAXIMUM_COMPUTE_CAPABILITY is None
                    or op.CUDA_MAXIMUM_COMPUTE_CAPABILITY >= compute_capability
                )
            )
        )
        and op.is_available()
    ]


ALL_FW_OPS = _filter_unsupported_ops(ALL_FW_OPS)
ALL_BW_OPS = _filter_unsupported_ops(ALL_BW_OPS)


def sample_random_supported_fw(
    inp: fmha.Inputs, seed, op_bw: Type[fmha.common.AttentionBwOpBase]
) -> Type[fmha.common.AttentionFwOpBase]:
    r = random.Random(seed)
    fw_ops = list(ALL_FW_OPS)
    if op_bw == fmha.cutlass_blackwell.BwOp:
        fw_ops = [fmha.cutlass_blackwell.FwOp, fmha.flash.FwOp]
    if (
        isinstance(inp.attn_bias, fmha.attn_bias.VARLEN_BIASES)
        and inp.attn_bias.q_seqinfo.seqstart.shape[0] > 2
    ):
        fw_ops = [
            op for op in fw_ops if op.VARLEN_LSE_PACKED == op_bw.VARLEN_LSE_PACKED
        ]
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
        Ks = [1, 2, 3, 31, 34, 36, 38, 40, 64, 80, 160, 256 + 2, 256 + 8, 512]
        for _K in Ks:
            if op.SUPPORTED_MIN_K <= _K <= op.SUPPORTED_MAX_K:
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
        fmha.cutlass_blackwell.FwOp,
        fmha.cutlass_blackwell.BwOp,
        fmha.flash.BwOp,
        fmha.ck.FwOp,
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


# This temporary working is necessary because the MTIA test collection might not happen
# on the same device as the device the tests are actually executed on. If test collection
# is done on a device without MTIA, the supported masks will contain masks that MTIA support
# and the corresponding tests will get collected. But when it comes time to actually run the
# tests, the mask won't be supported because it is run on an actual MTIA device.
def get_supported_attn_bias_types(op):
    supported_attn_bias_types = op.SUPPORTED_ATTN_BIAS_TYPES

    try:
        import mtia.host_runtime.torch_mtia.dynamic_library  # noqa

        supported_attn_bias_types = [
            b
            for b in supported_attn_bias_types
            if not issubclass(
                b,
                (
                    fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                    fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                ),
            )
        ]
    except (ImportError, OSError):
        pass

    return supported_attn_bias_types


def _generate_op_device_dtype_biasT_B_Mq_Mkv_H_K_Kv(
    ops_list: Sequence[Type[fmha.AttentionOpBase]], max_shapes_per_op: int = 65000
):
    r = random.Random(0)
    combination = []
    for op in ops_list:
        op_count = 0
        # Sort list of masks, so it's deterministic across runs
        LIST_MASKS = sorted(get_supported_attn_bias_types(op), key=str)
        for shape in generate_test_shapes_B_Mq_Mkv_H_K_Kv(op):
            has_one = False
            for device in _devices:
                if device not in op.SUPPORTED_DEVICES:
                    continue
                # Sort set of dtypes to make it deterministic across runs
                for dtype in sorted(op.SUPPORTED_DTYPES, key=str):
                    # "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'
                    if dtype in [torch.float8_e4m3fn]:
                        continue
                    bias_type = r.choice(LIST_MASKS)
                    # Avoid using too much memory
                    B, Mq, Mkv, H, K, Kv = shape
                    if bias_type not in [
                        type(None),
                        fmha.attn_bias.LowerTriangularMask,
                    ]:
                        B = min(B, 12)

                        if bias_type in {
                            fmha.attn_bias.BlockDiagonalCausalFromBottomRightMask,
                            fmha.attn_bias.BlockDiagonalCausalLocalAttentionFromBottomRightMask,
                        }:
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq) + 2
                        elif bias_type in {
                            fmha.attn_bias.BlockDiagonalCausalLocalAttentionPaddedKeysMask,
                            fmha.attn_bias.BlockDiagonalLocalAttentionPaddedKeysMask,
                            fmha.attn_bias.BlockDiagonalCausalWithOffsetGappyKeysMask,
                            fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
                            fmha.attn_bias.BlockDiagonalPaddedKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
                            fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                        }:
                            Mq, Mkv = min(Mkv, Mq), max(Mkv, Mq)
                    new_shape = (B, Mq, Mkv, H, K, Kv)
                    combination.append((op, device, dtype, bias_type, *new_shape))
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
                # Sort set of dtypes to make it deterministic across runs
                for dtype in sorted(op.SUPPORTED_DTYPES, key=str):
                    # "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'
                    if dtype in [torch.float8_e4m3fn]:
                        continue
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
    if tensor_with_grad is not None:
        grad = tensor_with_grad.grad
        if clear:
            tensor_with_grad.grad = None
        return grad
    return None


def create_tensors(
    op: Optional[Type[AttentionOpBase]],
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
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

    if attn_bias_type is not None and issubclass(
        attn_bias_type,
        (
            fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
        ),
    ):
        page_size_choices = [256, 512]
        if op is not None and issubclass(op, fmha.triton_splitk.FwOp):
            # TODO: enable small pages for flash attention when that's implemented
            page_size_choices.extend([64, 128])
        page_size = random.choice(page_size_choices)
        kv_len_paged = (kv_len + page_size - 1) // page_size * page_size
    else:
        kv_len_paged = kv_len
        page_size = None

    scale = 3
    if fmt == "BMK":
        query = torch.randn((B * h, q_len, k), device=device, dtype=dtype)
        key = torch.randn((B * h, kv_len_paged, k), device=device, dtype=dtype)
        value = torch.randn((B * h, kv_len_paged, kv), device=device, dtype=dtype)
    elif fmt == "BMHK":
        query = torch.randn((B, q_len, h, k), device=device, dtype=dtype)
        key = torch.randn((B, kv_len_paged, h, k), device=device, dtype=dtype)
        value = torch.randn((B, kv_len_paged, h, kv), device=device, dtype=dtype)
    else:
        assert fmt == "BMGHK"
        query = torch.randn((B, q_len, g, h, k), device=device, dtype=dtype)
        key = torch.randn((B, kv_len_paged, g, 1, k), device=device, dtype=dtype)
        value = torch.randn((B, kv_len_paged, g, 1, kv), device=device, dtype=dtype)

    for x in [query, key, value]:
        x.mul_(scale)

    if fmt == "BMGHK":
        # Expand - after the in-place mul
        key = key.expand((B, kv_len_paged, g, h, k))
        value = value.expand((B, kv_len_paged, g, h, k))

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
            page_size=page_size,
        )
        if isinstance(
            attn_bias,
            (
                fmha.attn_bias.BlockDiagonalMask,
                fmha.attn_bias.BlockDiagonalGappyKeysMask,
                fmha.attn_bias.BlockDiagonalPaddedKeysMask,
                fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
                fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
            ),
        ):
            query, key, value = [
                x.reshape([1, -1, *x.shape[2:]]) for x in [query, key, value]
            ]

    inputs = fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias)
    if op is not None:
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


def nanify_oob_seqlen(x: torch.Tensor) -> torch.Tensor:
    align_to = 256
    if x.shape[1] % align_to == 0:
        return x
    pad = [0, 0] * x.ndim
    pad[-3] = align_to - (x.shape[1] % align_to)
    x_pad = torch.nn.functional.pad(x, pad, value=math.nan)
    return x_pad[:, : x.shape[1]]


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
    if packed and issubclass(
        bias_type,
        (
            fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
            fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
        ),
    ):
        pytest.skip(
            "packed doesn't make sense with paged attention, since q has different shape than k/v"
        )
    if packed and not (k == kv and q_len == kv_len):
        pytest.skip(
            f"packed incompatible with `k ({k}) != kv ({kv})` or `q_len ({q_len}) != kv_len ({kv_len})`"
        )
    if fmt == "BMK" and not fmha.common._is_bias_type_supported_in_BMK(bias_type):
        pytest.skip("BMK incompatible with this bias")

    if op is fmha.ck.FwOp:
        if (k > 256 or kv > 256) and issubclass(
            bias_type,
            (
                fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            ),
        ):
            pytest.skip("ck.FwOp hdim-512 is not supported when Paged-KVCache is used!")

    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt="BMHK" if packed else fmt,
        **kwargs,
    )
    if attn_bias is not None:
        assert type(attn_bias.to(query.device)) is type(attn_bias)

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
        nanify_oob_seqlen(query),
        nanify_oob_seqlen(key),
        nanify_oob_seqlen(value),
        attn_bias,
        op=op,
    )
    assert not out2.isnan().any(), "Output has NaNs - most likely reading out-of-bounds"
    assert torch.allclose(out, out2, atol=0.0, rtol=0.0), (
        "Non-deterministic behavior",
        attn_bias,
    )

    ref = ref_attention_for_test(query, key, value, attn_bias)
    assert out.shape == ref.shape, out.shape
    assert_allclose(
        out.float(),
        ref,
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )


def _block_diag_reshape_lse(
    lse: torch.Tensor, q_seqinfo: fmha.attn_bias._SeqLenInfo
) -> torch.Tensor:
    """LSE can be padded, let's remove the padding"""
    parts = []
    for slice, (start, end) in zip(lse.unbind(0), q_seqinfo.intervals()):
        parts.append(slice[:, : end - start])
    return torch.cat(parts, dim=1).unsqueeze(0)


@disable_tf32
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

    if op is fmha.ck.FwOp:
        if issubclass(
            bias_type,
            (
                fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask,
                fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            ),
        ):
            pytest.skip(
                "With ck.FwOp Paged-KVCache has some problem with forward training!"
            )

    query, key, value, attn_bias = create_tensors(
        *opFW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv,
        fmt="BMHK",
    )

    _out, lse = xformers.ops.memory_efficient_attention_forward_requires_grad(
        query,
        key,
        value,
        op=op,
        attn_bias=attn_bias,
    )
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    attn = (query.float() / k**0.5) @ key.float().transpose(-2, -1)
    if attn_bias is not None:
        if isinstance(
            attn_bias,
            fmha.attn_bias.AttentionBias,
        ):
            bias_shape = (1, 1, query.shape[2], key.shape[2])
            tensor_bias = attn_bias.materialize(
                bias_shape,
                device=query.device,
                dtype=torch.float32,
            )
        else:
            assert type(attn_bias) is torch.Tensor
            tensor_bias = attn_bias
        attn = attn + tensor_bias.float()
    ref_lse = attn.logsumexp(-1)
    if isinstance(attn_bias, fmha.attn_bias.VARLEN_BIASES):
        # Convert to packed format if not already the case
        if not op.VARLEN_LSE_PACKED:
            lse = _block_diag_reshape_lse(lse, attn_bias.q_seqinfo)
    if op is fmha.cutlass.FwOp:
        # CUTLASS kernel pads the last dimention of LSE to 32
        lse = lse[:, :, : ref_lse.shape[2]]
    if op is fmha.ck.FwOp:
        # relax numerical tolerance for CK FwOp
        assert_allclose(lse, ref_lse, atol=2e-4, rtol=2e-4)
    else:
        assert_allclose(lse, ref_lse, atol=2e-4)


@cuda_or_mtia_only
@pytest.mark.parametrize(
    "op",
    _filter_unsupported_ops(
        [
            fmha.cutlass.FwOp,
            fmha.cutlass_blackwell.FwOp,
            fmha.flash.FwOp,
        ]
    ),
)
def test_logsumexp_mqa(op):
    device = torch._C._get_accelerator().type

    if not op.is_available():
        pytest.skip("not available")

    if device == "mtia" and op == fmha.cutlass.FwOp:
        pytest.skip("cutlass FwOp is not supported on MTIA")

    if device == "cuda" and op.CUDA_MINIMUM_COMPUTE_CAPABILITY > compute_capability:
        skip_reason = (
            f"requires device with capability >= {op.CUDA_MINIMUM_COMPUTE_CAPABILITY} "
            f"but your GPU has capability {compute_capability} (too old)"
        )
        pytest.skip(skip_reason)

    dtype = torch.float16
    s = 3
    query = torch.randn([1, 1, 32, 128], dtype=dtype, device=device) * s
    key = (torch.randn([1, 16, 1, 128], dtype=dtype, device=device) * s).expand(
        -1, -1, 32, -1
    )
    value = (torch.randn([1, 16, 1, 128], dtype=dtype, device=device) * s).expand(
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


@disable_tf32
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

    # Big batch sizes can be slow on MTIA, especially older devices because
    # it doesn't have attention fast paths. Testing on very big batch sizes
    # doesn't meaningfully increase our test coverage here, as long as most
    # permutations of parameters are tested on lower batch sizes.
    if device.startswith("mtia") and batch_size >= 11:
        pytest.skip("Skipping big batch test cases on MTIA")

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
            op_bw=op_bw,
        )
        if op_bw != fmha.cutlass.BwOp
        else fmha.cutlass.FwOp
    )

    if op_bw == fmha.ck.BwOp:
        op_fw = fmha.ck.FwOp
        if dtype == torch.bfloat16:
            # bfloat16 testing can be enabled by export ENABLE_HIP_FMHA_RTN_BF16_CONVERT=1 when
            # building xformers and get accurate results
            pytest.skip(
                "CK Fmha backward for bfloat16 currently is not very accurate for some cases!"
            )
        if not grad_out_contiguous:
            pytest.skip("CK Fmha does not support non-contiguous layout for grad_out!")
        if k % 2 != 0:
            pytest.skip(
                "CK Fmha currently requires the headdim size of query input be an even value!"
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

    if use_cpu_ref(device):
        query = query.detach().cpu()
        key = key.detach().cpu()
        value = value.detach().cpu()
        grad_out = grad_out.detach().cpu()

        if qkv is not None:
            qkv = torch.stack([query, key, value], 2)
            qkv.requires_grad_(True)
            query, key, value = xformers.ops.unbind(qkv, 2)

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)
        grad_out.requires_grad_(True)

        if isinstance(attn_bias, torch.Tensor):
            attn_bias = attn_bias.cpu()

    ref = ref_attention_for_test(query, key, value, attn_bias, scale=scale)
    ref.backward(grad_out)

    assert_allclose(
        out.float().to(ref.device),
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

    # This is a special case without masks where the accumulated numbers become so big that
    # we lose too much precision, especially on bfloat16. For this reason, the default bfloat16
    # tolerance for the backward pass is set to 0.9, but in some cases on MTIA we get up to 1.3,
    # probably due to the fact that the implementation doesn't use the fused kernels yet, which
    # increases the precision loss caused by the accumulation.
    if (
        device.startswith("mtia")
        and issubclass(bias_type, type(None))
        and q_len >= 2**16
    ):
        atol *= 1.6

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
            calc_grad.to(ref_grad.device),
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
    assert op == fmha.ck.FwOp, f"Op {op.NAME} does not expose dropout mask"
    mask = torch.empty((batch_size, 1, q_len, kv_len), device=device)
    # rand_uniform is an int8_t tensor
    rand_uniform = torch.ops.xformers._ck_rand_uniform(p, mask)
    mask = (rand_uniform <= int((1.0 - p) * 255.0)).to(torch.float32)
    mask = mask.reshape(batch_size, q_len, kv_len)

    return mask


@rocm_only
@pytest.mark.parametrize("attn_bias", [None, fmha.attn_bias.LowerTriangularMask()])
@pytest.mark.parametrize("seed", [42, 124])
@pytest.mark.parametrize("p", [0.3, 0.7])
@pytest.mark.parametrize("k_len", [32])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33, 65])
@pytest.mark.parametrize("q_len", [2, 33])
def test_dropout_ck(q_len, kv_len, batch_size, k_len, p, seed, attn_bias):
    op = fmha.ck.FwOp
    device = "cuda"
    scale = 3

    dtype = torch.float16

    query = torch.randn((batch_size, q_len, k_len), device=device, dtype=dtype) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device, dtype=dtype) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device, dtype=dtype) * scale

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
    ref = ref_attention_for_test(query, key, value, attn_bias, mask, p)

    if dtype is torch.float:
        assert_allclose(out, ref, atol=2e-4), f"{(out - ref).abs().max()}"
    else:
        assert_allclose(out.float(), ref, atol=2.8e-2), f"{(out - ref).abs().max()}"

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


@rocm_only
@pytest.mark.parametrize("p", [0.000001, 0.3, 0.7])
@pytest.mark.parametrize("k", [16, 64, 128])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("kv_len", [3, 248, 256])
@pytest.mark.parametrize("q_len", [3, 248, 256])
def test_dropout_backward_ck(q_len, kv_len, batch_size, k, p):
    op = fmha.ck.FwOp
    dtype = torch.float16
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

    ref = ref_attention_for_test(query, key, value, None, mask, p)
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

    inputs = fmha.Inputs(query=query, key=key, value=value, attn_bias=attn_bias)
    op_bw = opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv[0]
    op_fw = sample_random_supported_fw(
        inputs, seed=f"{op_bw.NAME}{query.shape}", op_bw=op_bw
    )
    out, lse = xformers.ops.memory_efficient_attention_forward_requires_grad(
        query, key, value, attn_bias, op=op_fw
    )
    assert out.ndim == query.ndim
    dq, dk, dv = xformers.ops.memory_efficient_attention_backward(
        grad_out, out, lse, query, key, value, attn_bias, op=op_bw
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

    ref = ref_attention_for_test(query, key, value)
    assert out.shape == ref.shape, out.shape

    assert_allclose(
        out.float(),
        ref.float(),
        atol=op.ERROR_ATOL[dtype],
        rtol=op.ERROR_RTOL.get(dtype, 1e-5),
    )


@disable_tf32
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
    if device not in {"cuda", "mtia"}:
        pytest.skip("Not CUDA or MTIA")

    query, key, value, attn_bias = create_tensors(
        *opBW_device_dtype_biasT_B_Mq_Mkv_H_K_Kv, fmt="BMK"
    )
    inputs = fmha.Inputs(
        query=query, key=key, value=value, attn_bias=attn_bias, scale=scale
    )
    op_fw = sample_random_supported_fw(inputs, seed=q_len * k + kv_len * k, op_bw=op_bw)
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

    ref = ref_attention_for_test(query * s, key, value, attn_bias, None, p, scale)
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


# TODO: Enable this for MTIA
# MTIA doesn't support this yet
@disable_on_mtia
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
    if op is fmha.triton_splitk.FwOp:
        pytest.skip("Triton Flash Decoding doesn't support backward pass yet")
    if op is fmha.ck.FwOp:
        pytest.skip("ck-tiled FMHA doesn't supported backward pass yet")

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


@pytest.mark.parametrize("op", ALL_FW_OPS, ids=[op.NAME for op in ALL_FW_OPS])
def test_unsupported_cpu(op: Type[fmha.AttentionFwOpBase]):
    q = torch.empty([1, 1, 1, 32])
    with pytest.raises(ValueError):
        fmha.memory_efficient_attention(q, q, q, op=(op, None))


@cuda_only
@pytest.mark.parametrize("op", ALL_FW_OPS, ids=[op.NAME for op in ALL_FW_OPS])
def test_unsupported_stride_lastdim(op: Type[fmha.AttentionFwOpBase]):
    K = max(op.SUPPORTED_MIN_K, 32)
    q = torch.empty([1, 1, K, 4], device="cuda", dtype=torch.float16).permute(
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
@pytest.mark.parametrize("op", ALL_FW_OPS, ids=[op.NAME for op in ALL_FW_OPS])
def test_unsupported_stride_alignment(op: Type[fmha.AttentionFwOpBase]):
    K = max(op.SUPPORTED_MIN_K, 32)
    q = torch.empty([1, 2, 1, K + 1], device="cuda", dtype=torch.float16)[:, :, :, :K]

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
    attn_bias, q = fmha.attn_bias.BlockDiagonalMask.from_tensor_list(queries)

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
    attn_bias, q = fmha.attn_bias.BlockDiagonalMask.from_tensor_list(queries)

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
    torch.manual_seed(0)

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


def _test_decoder(
    op,
    n_heads: int,
    kv_heads: Optional[int],
    padding: int,
    bsz: int,
    dtype: str,
    dequant: bool = False,
    num_queries: int = 1,
    d: int = 128,
) -> None:
    if not op.is_available():
        raise pytest.skip("not available")
    # kv_heads = 1: multiquery
    # kv_heads = None: neither MQA nor GQA
    # kv_heads > 1: BMGHK
    if dtype == "bf16" and torch.version.cuda and compute_capability < (8, 0):
        raise pytest.skip("BF16 is only supported on SM80+")
    import triton

    if dequant and triton.__version__[:4] < "3.0.":
        raise pytest.skip("dequant needs triton updates")
    dtype_ = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]
    torch.manual_seed(1)
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

    if skip_reasons := op.not_supported_reasons(fmha.Inputs(q, k, v)):
        pytest.skip("; ".join(skip_reasons))

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

    ref_output = ref_attention_for_test(q, k, v, attn_bias)

    assert_allclose(
        decoder_output.to(ref_output.dtype),
        ref_output,
        atol=op.ERROR_ATOL[dtype_] * 4,
        rtol=op.ERROR_RTOL[dtype_],
    )


@sm80_or_better_only
@pytest.mark.parametrize(
    "op,dequant,dtype",
    [
        (fmha.triton_splitk.FwOp_S1, False, "bf16"),
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
    # We omit dequant with f16: it needs a very high tol
    _test_decoder(
        op,
        kv_heads=kv_heads,
        n_heads=n_heads,
        padding=padding,
        bsz=bsz,
        dtype=dtype,
        dequant=dequant,
    )


@rocm_only
@pytest.mark.parametrize(
    "op", [fmha.ck_splitk.FwOp_S1, fmha.ck_splitk.FwOp_S2, fmha.ck_splitk.FwOp_S4]
)
@pytest.mark.parametrize("dtype", ["f32"])
@pytest.mark.parametrize("kv_heads", [None, 1, 2], ids=_kv_heads_label)
@pytest.mark.parametrize("n_heads", [16])
@pytest.mark.parametrize("d", [128, 256])
@pytest.mark.parametrize("padding, bsz", [(32, 8), (4096, 1), (32, 1), (4096, 8)])
def test_ck_splitk_decoder(
    op,
    kv_heads: Optional[int],
    n_heads: int,
    padding: int,
    bsz: int,
    dtype: str,
    d: int,
) -> None:
    # no quantized impl compared to cuda
    _test_decoder(
        op,
        kv_heads=kv_heads,
        n_heads=n_heads,
        padding=padding,
        bsz=bsz,
        dtype=dtype,
        d=d,
    )


@sm80_or_better_only
@pytest.mark.parametrize(
    "op",
    [
        fmha.triton_splitk.FwOp_S1,
        fmha.triton_splitk.FwOp_S2,
    ],
    ids=lambda op: f"splitk{op.SPLIT_K}",
)
@pytest.mark.parametrize("multiquery", [True, False], ids=lambda x: "mq" if x else "")
# n_heads=1 => it's ambiguous whether can count as multiquery
@pytest.mark.parametrize("padding, bsz", [(32, 8), (44, 1)])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("n_heads, num_queries", [(2, 4), (2, 5), (6, 7), (20, 3)])
def test_triton_splitk_decoder_manyqueries(
    op,
    multiquery: bool,
    n_heads: int,
    padding: int,
    bsz: int,
    dtype: str,
    num_queries: int,
) -> None:
    kv_heads = 1 if multiquery else None
    _test_decoder(
        op,
        kv_heads=kv_heads,
        n_heads=n_heads,
        padding=padding,
        bsz=bsz,
        dtype=dtype,
        num_queries=num_queries,
        dequant=False,
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

    if torch.version.hip:
        pytest.skip("backward pass/gradience is not yet supported by ck-tiled fmha!")

    K = 16
    dtype = torch.float16
    device = "cuda"
    list_x = [
        torch.randn([1, 3, 1, K], dtype=dtype, device=device),
        torch.randn([1, 6, 1, K], dtype=dtype, device=device),
        torch.randn([1, 2, 1, K], dtype=dtype, device=device),
    ]
    attn_bias, x = fmha.attn_bias.BlockDiagonalMask.from_tensor_list(list_x)

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

    @skip_if_sm100_or_better
    def test_f16_biasf32(self) -> None:
        q, k, v, bias = self.create_tensors(torch.float16)
        fmha.memory_efficient_attention(q, k, v, attn_bias=bias)
        bias = bias.to(torch.float32)
        with pytest.raises((ValueError, RuntimeError)):
            fmha.memory_efficient_attention(q, k, v, attn_bias=bias)

    @skip_if_sm100_or_better
    @disable_on_rocm
    def test_f32_biasf16(self) -> None:
        q, k, v, bias = self.create_tensors(torch.float32)
        fmha.memory_efficient_attention(q, k, v, attn_bias=bias)
        bias = bias.to(torch.float16)
        with pytest.raises((ValueError, RuntimeError)):
            fmha.memory_efficient_attention(q, k, v, attn_bias=bias)

    @skip_if_sm100_or_better
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_wrong_alignment(self, dtype) -> None:
        op = fmha.cutlass.FwOp if torch.version.cuda else fmha.ck.FwOp
        if dtype not in op.SUPPORTED_DTYPES:
            pytest.skip(
                f"{dtype=} is not supported by {op.__module__}.{op.__qualname__}"
            )

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
        ref_out = ref_attention_bmhk_for_test(q, k, v, bias)
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
            ref_out = ref_attention_bmhk_for_test(q, k, v, bias)
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


@cuda_or_mtia_only
@pytest.mark.parametrize("Mq", [1, 512])
@pytest.mark.parametrize(
    "opFW_biasT",
    [
        (op, biasT)
        for op in ALL_FW_OPS
        for biasT in get_supported_attn_bias_types(op)
        if op.SUPPORTS_BMGHK
    ],
    ids=lambda o: f"{o[0].NAME}-{o[1].__name__}" if isinstance(o, tuple) else "",
)
def test_forward_gqa(opFW_biasT, Mq: int):
    device = torch._C._get_accelerator().type

    opFW, biasT = opFW_biasT
    if Mq < 512 and (
        issubclass(biasT, fmha.attn_bias.LowerTriangularMask)
        or issubclass(biasT, fmha.attn_bias.BlockDiagonalCausalMask)
    ):
        pytest.skip("undefined upper left")
    B_Mq_Mkv_H_K_Kv = (3, Mq, 512, 16, 128, 128)
    test_forward(
        (
            opFW,
            device,
            torch.float16,
            biasT,
            *B_Mq_Mkv_H_K_Kv,
        ),
        packed=False,
        fmt="BMGHK",
        g=2,
    )


@cuda_or_mtia_only
@pytest.mark.parametrize(
    "opBW",
    [
        fmha.flash.BwOp,
        fmha.ck.BwOp if torch.version.hip else fmha.cutlass.BwOp,
        fmha.cutlass_blackwell.BwOp,
    ],
)
def test_backward_gqa(opBW):
    device = torch._C._get_accelerator().type

    H = 8
    B_Mq_Mkv_H_K_Kv = (3, 512, 512, H, 128, 128)
    dtype = torch.float16
    query, key, value, attn_bias = create_tensors(
        *(opBW, device, dtype, type(None), *B_Mq_Mkv_H_K_Kv),
        attn_bias_requires_grad=False,
        fmt="BMHK",
    )
    op = (fmha.ck.FwOp if torch.version.hip else fmha.cutlass.FwOp, opBW)
    key = key[:, :, :1].expand(-1, -1, H, -1)
    value = value[:, :, :1].expand(-1, -1, H, -1)
    key.requires_grad_(True)
    out = fmha.memory_efficient_attention(query, key, value, attn_bias=attn_bias)
    out.backward(query)
    dk = key.grad
    key.grad = None

    if use_cpu_ref(device):
        query = query.detach().cpu()
        key = key.detach().cpu()
        value = value.detach().cpu()
        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

    out_ref = ref_attention_bmhk_for_test(query, key, value, attn_bias=attn_bias)
    out_ref.backward(query)

    assert_allclose(
        out.float().to(out_ref.device),
        out_ref.float(),
        atol=op[0].ERROR_ATOL[dtype],
        rtol=op[0].ERROR_RTOL[dtype],
    )
    assert_allclose(
        dk.float().to(key.grad.device),
        key.grad.float(),
        atol=op[1].ERROR_ATOL[dtype],
        rtol=op[1].ERROR_RTOL[dtype],
    )


@cuda_or_mtia_only
@pytest.mark.parametrize("opFW", [op for op in ALL_FW_OPS if op.SUPPORTS_BMGHK])
def test_forward_gqa_one_group(opFW):
    device = torch._C._get_accelerator().type

    dtype = torch.float16
    B, Mq, Mkv, H, K = 3, 13, 16, 5, 128
    q = torch.randn([B, Mq, 1, H, K], dtype=dtype, device=device) * 3
    k = torch.randn([B, Mkv, 1, H, K], dtype=dtype, device=device) * 3
    v = torch.randn([B, Mkv, 1, H, K], dtype=dtype, device=device) * 3

    supported = opFW.supports(fmha.Inputs(q, k, v))
    if not supported:
        supported_bmhk = opFW.supports(fmha.Inputs(q[:, :, 0], k[:, :, 0], v[:, :, 0]))
        assert supported == supported_bmhk
        pytest.skip("not supported")
    out = fmha.memory_efficient_attention_forward(q, k, v, op=opFW)
    ref = ref_attention_for_test(q, k, v)
    assert_allclose(
        out.float(),
        ref,
        atol=opFW.ERROR_ATOL[dtype],
        rtol=opFW.ERROR_RTOL.get(dtype, 1e-5),
    )


@sm80_or_better_only
@disable_on_rocm
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


@disable_on_rocm
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


@disable_on_rocm
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


@cuda_or_mtia_only
@pytest.mark.parametrize(
    "op",
    [fmha.triton_splitk.FwOp, fmha.flash.FwOp, fmha.ck.FwOp],
    ids=lambda op: op.NAME,
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize(
    "B_Mkv_H_K",
    [
        (1, 2**16, 3, 128),
        (5, 53, 4, 64),
        (7, 51, 4, 256),
        (3, 51, 2, 512),
    ],
)
def test_mqa_decoding(op: Type[fmha.AttentionFwOpBase], dtype, B_Mkv_H_K):
    device = torch._C._get_accelerator().type

    B, Mkv, H, K = B_Mkv_H_K
    q = torch.randn([B, 1, H, K], dtype=dtype, device=device) * 3
    k = torch.randn([B, Mkv, 1, K], dtype=dtype, device=device) * 3
    v = torch.randn([B, Mkv, 1, K], dtype=dtype, device=device) * 3
    k = k.expand(-1, -1, H, -1)
    v = v.expand(-1, -1, H, -1)

    if skip_reasons := op.not_supported_reasons(fmha.Inputs(q, k, v)):
        pytest.skip("; ".join(skip_reasons))
    out = fmha.memory_efficient_attention_forward(q, k, v, op=op)
    ref = ref_attention_for_test(q, k, v)
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

    if torch.version.hip:
        pytest.skip("backward pass/gradience is not yet supported by ck-tiled fmha!")

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
    if opFW == fmha.triton_splitk.FwOp:
        pytest.skip("triton_splitk doesn't support empty kv")

    if torch.version.hip:
        pytest.skip("backward pass/gradience is not yet supported by ck-tiled fmha!")

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

    if torch.version.hip:
        pytest.skip("backward pass/gradience is not yet supported by ck-tiled fmha!")

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


@sm80_or_better_only
@pytest.mark.parametrize("B", [1, 5, 128])
@pytest.mark.parametrize("MAX_T", [64, 128, 2048, 4096, 8192])
@pytest.mark.parametrize(
    "op",
    [
        fmha.triton_splitk.FwOp,
        fmha.triton_splitk.FwOp_S8,
        fmha.triton_splitk.FwOp_Map[48],
    ],
    ids=lambda op: op.NAME,
)
@pytest.mark.parametrize("num_quant_groups", [0, 1, 8])
@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("gappy", [False, True], ids=lambda x: "gappy" if x else "")
def test_paged_attention(
    B,
    MAX_T: int,
    num_quant_groups: int,
    page_size: int,
    op: Type[AttentionFwOpBase],
    gappy: bool,
):
    msg = "Notional padding should be divisible by the page size"
    if gappy and MAX_T % page_size:
        context = pytest.raises(ValueError, match=msg)
    else:
        context = nullcontext()  # type: ignore[assignment]
    with context:
        paged_attention_run_inner(
            B, MAX_T, num_quant_groups, page_size, op, bench=False, gappy=gappy
        )


@rocm_only
@pytest.mark.parametrize("B", [1, 5, 128])
@pytest.mark.parametrize("MAX_T", [64, 128, 2048, 4096, 8192])
@pytest.mark.parametrize("page_size", [128, 256])
@pytest.mark.parametrize("gappy", [False, True], ids=lambda x: "gappy" if x else "")
def test_paged_attention_ck(B, MAX_T: int, page_size: int, gappy: bool):
    op = fmha.ck.FwOp
    num_quant_groups = 0
    msg = "Notional padding should be divisible by the page size"
    if gappy and MAX_T % page_size:
        context = pytest.raises(ValueError, match=msg)
    else:
        context = nullcontext()  # type: ignore[assignment]
    with context:
        paged_attention_run_inner(
            B, MAX_T, num_quant_groups, page_size, op, bench=False, gappy=gappy
        )


@sm80_or_better_only
@disable_on_rocm
@pytest.mark.parametrize("B", [1, 5, 128])
@pytest.mark.parametrize("MAX_T", [64, 128, 2048, 4096, 8192])
@pytest.mark.parametrize("page_size", [256])
def test_paged_attention_flash(B, MAX_T: int, page_size: int):
    # TODO: add smaller page sizes when https://github.com/Dao-AILab/flash-attention/pull/824 is merged
    op = fmha.flash.FwOp
    if (
        fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask
        not in get_supported_attn_bias_types(op)
    ):
        pytest.skip("Not supported bias")
    num_quant_groups = 0
    paged_attention_run_inner(B, MAX_T, num_quant_groups, page_size, op, bench=False)


@skip_if_sm100_or_better
@sm90_or_better_only
@disable_on_rocm
@pytest.mark.parametrize(
    "op", _filter_unsupported_ops([fmha.flash3.FwOp, fmha.flash3.FwOp_KVSplit])
)
@pytest.mark.parametrize("B", [1, 5, 128])
@pytest.mark.parametrize("MAX_T", [64, 128, 2048, 4096, 8192])
@pytest.mark.parametrize("page_size", [256])
def test_paged_attention_flash3(
    op: Type[AttentionFwOpBase], B: int, MAX_T: int, page_size: int
):
    if (
        fmha.attn_bias.PagedBlockDiagonalPaddedKeysMask
        not in get_supported_attn_bias_types(op)
    ):
        pytest.skip("Not supported bias")
    num_quant_groups = 0
    paged_attention_run_inner(B, MAX_T, num_quant_groups, page_size, op, bench=False)


def paged_attention_run_inner(
    B: int,
    MAX_T: int,
    num_quant_groups: int,
    page_size: int,
    op: Type[AttentionFwOpBase],
    bench: bool,
    gappy: bool = False,
) -> None:
    import triton

    torch.manual_seed(10)
    TEST_WARMUP_MS = 500
    TEST_RUN_MS = 5000

    N_H_L = 8
    N_KVH_L = 1
    D_H = 128
    D_H_KV = D_H // 8 + num_quant_groups if num_quant_groups else D_H
    kv_seqlens = torch.randint(low=1, high=MAX_T + 1, size=(B,)).tolist()
    # Paged attention requires k.shape[1] and v.shape[1] to be divisible by page_size, so pad
    padded_per_row_len = ((MAX_T + page_size - 1) // page_size) * page_size

    if gappy:
        make_paged_kwargs = {
            "paged_type": fmha.attn_bias.PagedBlockDiagonalGappyKeysMask,
            "notional_padding": MAX_T,
        }
        attn_bias = fmha.attn_bias.BlockDiagonalGappyKeysMask.from_seqlens(
            q_seqlen=[1] * B,
            kv_seqstarts=list(range(0, MAX_T * (B + 1), MAX_T)),
            kv_seqlen=kv_seqlens,
        )
    else:
        make_paged_kwargs = {
            "paged_type": fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        }

        block_type = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask
        attn_bias = block_type.from_seqlens(  # type: ignore
            q_seqlen=[1] * B,
            kv_padding=MAX_T,
            kv_seqlen=kv_seqlens,
        )

    q = torch.randn((B, 1, N_H_L, D_H), dtype=torch.bfloat16, device="cuda")
    if num_quant_groups:
        if triton.__version__[:4] < "3.0.":
            raise pytest.skip("dequant needs triton updates")

        # Using high=64 below, because with 256 both paged and non-paged paths
        # will produce NaNs - probably some quantization coeffitions are NaNs
        # after the bitwise cast.
        cache_k = torch.randint(
            0, 64, (B, MAX_T, N_KVH_L, D_H_KV * 4), dtype=torch.uint8, device="cuda"
        )
        cache_k = cache_k.view(dtype=torch.int32)
        cache_v = torch.randint(
            0, 64, (B, MAX_T, N_KVH_L, D_H_KV * 4), dtype=torch.uint8, device="cuda"
        )
        cache_v = cache_v.view(dtype=torch.int32)

        op = type(
            f"{op.__name__}_{num_quant_groups}",
            (op,),
            {"NUM_GROUPS": num_quant_groups},
        )
    else:
        cache_k = torch.randn(
            (B, MAX_T, N_KVH_L, D_H), dtype=torch.bfloat16, device="cuda"
        )
        cache_v = torch.randn_like(cache_k)

    axq = q.view(1, B * 1, N_H_L, D_H)
    axk = cache_k.view(1, B * MAX_T, N_KVH_L, D_H_KV).expand(
        1, B * MAX_T, N_H_L, D_H_KV
    )
    axv = cache_v.view(1, B * MAX_T, N_KVH_L, D_H_KV).expand(
        1, B * MAX_T, N_H_L, D_H_KV
    )

    k_cache_size_usual = axk.numel()

    # First, create "wasteful" K/V cache, where every block in logical cache
    # has a physical representation, even if there's nothing stored there

    block_tables = torch.arange(
        B * padded_per_row_len // page_size, device="cuda", dtype=torch.int32
    ).reshape(B, -1)

    shape_padded = (B, padded_per_row_len, N_KVH_L, D_H_KV)
    axk_padded = torch.empty(shape_padded, device=axk.device, dtype=axk.dtype)
    axv_padded = torch.empty(shape_padded, device=axv.device, dtype=axv.dtype)
    axk_padded[:, :MAX_T] = axk.view(B, -1, N_H_L, D_H_KV)[:, :, :1, :]
    axv_padded[:, :MAX_T] = axv.view(B, -1, N_H_L, D_H_KV)[:, :, :1, :]

    axk_padded = axk_padded.view(1, B * padded_per_row_len, N_KVH_L, D_H_KV)
    axv_padded = axv_padded.view(1, B * padded_per_row_len, N_KVH_L, D_H_KV)

    axk_padded = axk_padded.expand(-1, -1, N_H_L, -1)
    axv_padded = axv_padded.expand(-1, -1, N_H_L, -1)

    attn_bias_paged = attn_bias.make_paged(
        block_tables=block_tables,
        page_size=page_size,
        **make_paged_kwargs,  # type: ignore
    )
    if type(attn_bias_paged) not in op.SUPPORTED_ATTN_BIAS_TYPES:
        pytest.skip(f"{type(attn_bias_paged)} not supported")
    if type(attn_bias) not in op.SUPPORTED_ATTN_BIAS_TYPES:
        pytest.skip(f"{type(attn_bias_paged)} not supported")
    y_usual = fmha.memory_efficient_attention_forward(
        axq,
        axk,
        axv,
        attn_bias,
        op=op,
    )
    if bench:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_usual = fmha.memory_efficient_attention_forward(
                axq,
                axk,
                axv,
                attn_bias,
                op=op,
            )
        t_ms = triton.testing.do_bench(
            lambda g=g: g.replay(),
            warmup=TEST_WARMUP_MS,
            rep=TEST_RUN_MS,
        )
        logger.info(f"Non-paged attention took {t_ms * 1e3:.2f}us")

    y_wasteful = fmha.memory_efficient_attention_forward(
        axq,
        axk_padded,
        axv_padded,
        attn_bias_paged,
        op=op,
    )
    if bench:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_wasteful = fmha.memory_efficient_attention_forward(
                axq,
                axk_padded,
                axv_padded,
                attn_bias_paged,
                op=op,
            )
        t_ms = triton.testing.do_bench(
            lambda g=g: g.replay(),
            warmup=TEST_WARMUP_MS,
            rep=TEST_RUN_MS,
        )
        logger.info(f"Paged attention with wasteful K/V-cache took {t_ms * 1e3:.2f}us")

    torch.testing.assert_close(
        y_wasteful,
        y_usual,
        atol=1.0e-2,
        rtol=1.0e-2,
    )

    # Now let's create a "packed" K/V cache, where only meaniningful logical blocks are mapped to physical blocks
    (block_tables, packed_cache_k, packed_cache_v) = pack_kv_cache(
        cache_k,
        cache_v,
        kv_seqlens,
        page_size,
    )
    attn_bias_paged = attn_bias.make_paged(
        block_tables=block_tables,
        page_size=page_size,
        **make_paged_kwargs,  # type: ignore
    )
    axk = packed_cache_k.view(1, -1, N_KVH_L, D_H_KV).expand(1, -1, N_H_L, D_H_KV)
    axv = packed_cache_v.view(1, -1, N_KVH_L, D_H_KV).expand(1, -1, N_H_L, D_H_KV)

    k_cache_size_packed = axk.numel()

    y_packed = fmha.memory_efficient_attention_forward(
        axq,
        axk,
        axv,
        attn_bias_paged,
        op=op,
    )

    logger.info(
        f"KV-cache size reduced by {(100 * (1 - k_cache_size_packed/k_cache_size_usual)):.2f}%"
    )

    torch.testing.assert_close(y_wasteful, y_packed)

    # Let's swap two blocks, and adjust two corresponding entries in the block table. The result shouldn't change
    i, j = 0, axk.shape[1] // page_size - 1

    axk = axk[:, :, :1, :]
    axv = axv[:, :, :1, :]

    vals_i = axk[:, i * page_size : (i + 1) * page_size, :, :].clone()
    vals_j = axk[:, j * page_size : (j + 1) * page_size, :, :].clone()
    axk[:, i * page_size : (i + 1) * page_size, :, :] = vals_j
    axk[:, j * page_size : (j + 1) * page_size, :, :] = vals_i

    vals_i = axv[:, i * page_size : (i + 1) * page_size, :, :].clone()
    vals_j = axv[:, j * page_size : (j + 1) * page_size, :, :].clone()
    axv[:, i * page_size : (i + 1) * page_size, :, :] = vals_j
    axv[:, j * page_size : (j + 1) * page_size, :, :] = vals_i

    axk = axk.expand(-1, -1, N_H_L, -1)
    axv = axv.expand(-1, -1, N_H_L, -1)

    where_i = block_tables == i
    where_j = block_tables == j
    block_tables.masked_fill_(where_i, j)
    block_tables.masked_fill_(where_j, i)

    y_swapped = fmha.memory_efficient_attention_forward(
        axq,
        axk,
        axv,
        attn_bias_paged,
        op=op,
    )
    if bench:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y_swapped = fmha.memory_efficient_attention_forward(
                axq,
                axk,
                axv,
                attn_bias_paged,
                op=op,
            )
        t_ms = triton.testing.do_bench(
            lambda g=g: g.replay(),
            warmup=TEST_WARMUP_MS,
            rep=TEST_RUN_MS,
        )
        logger.info(f"Paged attention with packed K/V-cache took {t_ms * 1e3:.2f}us")

    torch.testing.assert_close(y_swapped, y_packed)


@sm80_or_better_only
@pytest.mark.parametrize(
    "bias_t",
    [None, fmha.attn_bias.LowerTriangularMask, fmha.attn_bias.BlockDiagonalMask],
)
@pytest.mark.parametrize("create_bias_inside_compiled", [False, True])
@pytest.mark.parametrize(
    "op",
    [
        None,
        (fmha.flash.FwOp, fmha.flash.BwOp),
        (fmha.flash3.FwOp, fmha.flash3.BwOp),
        (fmha.cutlass_blackwell.FwOp, fmha.cutlass_blackwell.BwOp),
    ],
)
def test_memeff_compile(bias_t, create_bias_inside_compiled: bool, op) -> None:
    torch.manual_seed(0)
    if op is not None and not op[0].is_available():
        pytest.skip("Op is not available")
    torch._dynamo.reset_code_caches()  # avoids hitting recompilation limit
    B, M, H, K = 1, 256, 2, 64
    q, k, v, bias = create_tensors(
        op if op is None else op[0],
        "cuda",
        torch.float16,
        bias_t,
        B,
        M,
        M,
        H,
        K,
        K,
        fmt="BMHK",
    )
    grad = torch.randn_like(q)
    if create_bias_inside_compiled:
        bias = None
        if bias_t is not None:
            pytest.skip("Can't create this mask inside compile")
    if bias is not None:
        bias.to(q.device)
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    def fmha_fn(q, k, v, bias):
        if create_bias_inside_compiled and bias_t is not None:
            bias = bias_t()
        return fmha.memory_efficient_attention(q, k, v, attn_bias=bias, op=op)

    # Eager reference
    out_ref = fmha_fn(q, k, v, bias)
    out_ref.backward(grad)
    dq_ref, dk_ref, dv_ref = q.grad, k.grad, v.grad
    q.grad, k.grad, v.grad = None, None, None

    # Compiled version
    fmha_c = torch.compile(fmha_fn, fullgraph=True, dynamic=False)
    out = fmha_c(q, k, v, bias)
    out.backward(grad)

    assert_allclose(
        out,
        out_ref,
        "out",
        atol=fmha.flash.FwOp.ERROR_ATOL[q.dtype],
        rtol=fmha.flash.FwOp.ERROR_RTOL[q.dtype],
    )
    atol, rtol = (
        fmha.flash.BwOp.ERROR_ATOL[q.dtype],
        fmha.flash.BwOp.ERROR_RTOL[q.dtype],
    )
    assert_allclose(q.grad, dq_ref, "dq", atol=atol, rtol=rtol)
    assert_allclose(k.grad, dk_ref, "dk", atol=atol, rtol=rtol)
    assert_allclose(v.grad, dv_ref, "dv", atol=atol, rtol=rtol)


@sm90_or_better_only
@pytest.mark.parametrize("B", [1, 16, 64])
@pytest.mark.parametrize("Mkv", [2048, 8192])
@pytest.mark.parametrize("Hkv", [1, 2])
@pytest.mark.parametrize("G", [1, 8])
@pytest.mark.parametrize("page_size", [64, 256])
@torch.no_grad()
def test_triton_splitk_rowwise_fp8(
    B: int,
    Mkv: int,
    Hkv: int,
    G: int,
    page_size: int,
    Mq: int = 1,
    K: int = 128,
) -> None:
    torch.manual_seed(10)

    Hq = Hkv * G

    device = torch.device("cuda")
    dtype = torch.bfloat16

    inp, inp_ref, inp_fp8_paged, inp_bf16_paged = construct_fp8_attention_inputs(
        B, Mkv, Mq, Hkv, Hq, K, page_size, device, dtype
    )

    (
        attn_output_fp8,
        context_fp8,
    ) = fmha._memory_efficient_attention_forward_requires_grad(
        inp, op=fmha.triton_splitk.FwOp
    )

    (
        attn_output_ref,
        context_ref,
    ) = fmha._memory_efficient_attention_forward_requires_grad(
        inp_ref, op=fmha.triton_splitk.FwOp
    )

    torch.testing.assert_close(attn_output_fp8, attn_output_ref, atol=5e-3, rtol=5e-3)
    assert context_fp8 is not None and context_ref is not None
    torch.testing.assert_close(context_fp8.lse, context_ref.lse, atol=5e-4, rtol=5e-4)

    # Paged K/V cache

    paged_bias = fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
    if paged_bias not in fmha.triton_splitk.FwOp.SUPPORTED_ATTN_BIAS_TYPES:
        return

    (
        attn_output_fp8_paged,
        context_fp8_paged,
    ) = fmha._memory_efficient_attention_forward_requires_grad(
        inp_fp8_paged, op=fmha.triton_splitk.FwOp
    )
    torch.testing.assert_close(
        attn_output_fp8, attn_output_fp8_paged, atol=2e-3, rtol=2e-3
    )
    assert context_fp8_paged is not None
    torch.testing.assert_close(
        context_fp8.lse, context_fp8_paged.lse, atol=1e-4, rtol=1e-4
    )


def fp8_per_head_quantize(
    x: torch.Tensor,
    dtype_fp8: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    MAX_FP8 = torch.finfo(dtype_fp8).max
    EPS = 1e-12
    SCALE_UP = 1200.0
    tensor_max = torch.amax(torch.abs(x), dim=(1, 3), keepdim=False).to(torch.float32)
    clamp_max = torch.clamp(tensor_max, min=EPS, max=SCALE_UP)
    scale = MAX_FP8 / clamp_max  # Shape: [batch, num_heads]
    x_quantized = (x * scale[:, None, :, None]).to(
        dtype_fp8
    )  # Shape: [B, seq_len, num_heads, head_dim]

    return x_quantized, 1 / scale  # Shape: [batch, num_heads]


@disable_on_rocm
@sm90_or_better_only
@pytest.mark.parametrize("dtype_init", [torch.bfloat16])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B", [4, 8, 16])
@pytest.mark.parametrize("nheads", [6, 16])
@pytest.mark.parametrize("seq_len", [256, 512, 1024])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_fp8_attention(dtype_init, deterministic, causal, B, nheads, seq_len, head_dim):
    op = fmha.flash3.FwOp
    if not op.is_available():
        pytest.skip("FAv3 is not available")
    dtype_fp8 = torch.float8_e4m3fn
    if dtype_fp8 not in op.SUPPORTED_DTYPES:
        pytest.skip("FP8 is not supported")

    q = torch.randn(B, seq_len, nheads, head_dim, device="cuda", dtype=dtype_init)
    k = torch.randn(B, seq_len, nheads, head_dim, device="cuda", dtype=dtype_init)
    v = torch.randn(B, seq_len, nheads, head_dim, device="cuda", dtype=dtype_init)

    q_fp8, descale_q = fp8_per_head_quantize(q, dtype_fp8)
    k_fp8, descale_k = fp8_per_head_quantize(k, dtype_fp8)
    v_fp8, descale_v = fp8_per_head_quantize(v, dtype_fp8)

    q_fp8_packed = pack_fp8_tensorwise_per_head(q_fp8, descale_q, dtype_init)
    k_fp8_packed = pack_fp8_tensorwise_per_head(k_fp8, descale_k, dtype_init)
    v_fp8_packed = pack_fp8_tensorwise_per_head(v_fp8, descale_v, dtype_init)

    q_fp8_fake = q_fp8_packed.dequantize()
    k_fp8_fake = k_fp8_packed.dequantize()
    v_fp8_fake = v_fp8_packed.dequantize()

    out_ref = fmha.memory_efficient_attention_forward(
        q_fp8_fake, k_fp8_fake, v_fp8_fake, None, op=op
    )

    out = fmha.memory_efficient_attention_forward(
        q_fp8_packed,
        k_fp8_packed,
        v_fp8_packed,
        None,
        op=op,
    )

    # NOTE: output dtype of FP8 attention is hard-code to BF16 for now: https://fburl.com/jcfiqmg0
    assert out.dtype == torch.bfloat16, "FP8 output is not BF16"
    torch.testing.assert_close(out, out_ref, atol=3e-2, rtol=1e-4)


def _pack_xformer_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache_seqlens: List[int],
    bias_type,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    fmha.attn_bias.BlockDiagonalPaddedKeysMask,
]:
    batch, seq_len_q, head_q, head_d = q.shape
    _, max_len_kv, head_kv, _ = k.shape

    attn_bias = bias_type.from_seqlens(
        q_seqlen=[seq_len_q] * batch,
        kv_seqlen=cache_seqlens,
        kv_padding=max_len_kv,
    )

    q = q.view(1, -1, head_q, head_d)
    k = k.expand(-1, -1, head_q, -1).view(1, -1, head_q, k.shape[-1])
    v = v.expand(-1, -1, head_q, -1).view(1, -1, head_q, v.shape[-1])
    return q, k, v, attn_bias


@disable_on_rocm
@sm90_or_better_only
@pytest.mark.parametrize("dtype_init", [torch.bfloat16])
@pytest.mark.parametrize("deterministic", [True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("B", [4, 8, 16])
@pytest.mark.parametrize("nheads_q", [8, 16])
@pytest.mark.parametrize("seq_len_q", [1, 2, 4, 8])
@pytest.mark.parametrize("seq_len_kv", [256, 512, 1024, 2048])
@pytest.mark.parametrize("max_len_kv", [4096, 8192])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize(
    "bias",
    [
        fmha.attn_bias.BlockDiagonalPaddedKeysMask,
        fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask,
    ],
)
def test_fav3_kvsplit_attn(
    dtype_init,
    deterministic,
    causal,
    B,
    nheads_q,
    seq_len_q,
    seq_len_kv,
    max_len_kv,
    head_dim,
    bias,
):
    op = fmha.flash3.FwOp_KVSplit
    if not op.is_available():
        pytest.skip("FAv3 KVSplit is not available")
    nheads_kv = 1

    q = torch.randn(B, seq_len_q, nheads_q, head_dim, device="cuda", dtype=dtype_init)
    k = torch.randn(B, seq_len_kv, nheads_kv, head_dim, device="cuda", dtype=dtype_init)
    v = torch.randn(B, seq_len_kv, nheads_kv, head_dim, device="cuda", dtype=dtype_init)

    xq, xk, xv, attn_bias = _pack_xformer_input(q, k, v, [seq_len_kv] * B, bias)

    out_ref, lse_ref = fmha.memory_efficient_attention_forward_requires_grad(
        xq, xk, xv, attn_bias, op=fmha.flash3.FwOp
    )

    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        xq,
        xk,
        xv,
        attn_bias,
        op=op,
    )

    torch.testing.assert_close(out, out_ref, atol=4e-3, rtol=1e-4)

    torch.testing.assert_close(lse, lse_ref, atol=4e-3, rtol=1e-4)


@sm90_or_better_only
@pytest.mark.parametrize(
    "op",
    _filter_unsupported_ops(
        (
            [
                fmha.flash.FwOp,
                fmha.cutlass.FwOp,
                fmha.flash3.FwOp,
                fmha.flash3.FwOp_KVSplit,
            ]
            if not torch.version.hip
            else [fmha.ck.FwOp]
        )
        + [fmha.triton_splitk.FwOp]
    ),
)
def test_nans_in_padding(op):
    """
    Create a batch of sequences with variable lengths, stored in padded format,
    and fill the unused positions in K/V with NaNs.
    This shouldn't affect the result, but currently FA3 produces NaNs in the output.
    The reason is probably that some sequences end in the middle of a K/V block,
    and NaNs leak into quantities like per-block maximum of Q@K used in FA algorithm.
    """

    if "cuda" not in _devices:
        pytest.skip("CUDA device is not available")

    nheads_kv = 1
    nheads_q = 8
    B = 64
    seq_len_q = 15
    max_len_kv = 256
    head_dim = 128
    dtype = torch.bfloat16
    page_size = 64

    padded_per_row_len = ((max_len_kv + page_size - 1) // page_size) * page_size

    assert padded_per_row_len == max_len_kv

    q = torch.randn(B, seq_len_q, nheads_q, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(B, max_len_kv, nheads_kv, head_dim, device="cuda", dtype=dtype)
    v = torch.randn(B, max_len_kv, nheads_kv, head_dim, device="cuda", dtype=dtype)

    xq = q.view(1, -1, nheads_q, head_dim)
    xk = k.view(1, -1, nheads_kv, head_dim).expand(1, -1, nheads_q, -1)
    xv = v.view(1, -1, nheads_kv, head_dim).expand(1, -1, nheads_q, -1)

    # For non-paged FA3, the seqlens need to be a multiple of tile_size (128) since TMA loading V uses fixed tile-size.
    if op in [fmha.flash3.FwOp, fmha.flash3.FwOp_KVSplit]:
        tile_sz = 128
        seqlens = torch.randint(max_len_kv // tile_sz, size=(B,), device="cuda")
        seqlens = seqlens * tile_sz
    else:
        seqlens = torch.randint(max_len_kv, size=(B,), device="cuda")

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[seq_len_q] * B, kv_seqlen=seqlens.tolist(), kv_padding=max_len_kv
    )
    if type(attn_bias) not in op.SUPPORTED_ATTN_BIAS_TYPES:
        pytest.skip(f"Op {op.NAME} doesn't support {type(attn_bias)}")

    out_ref, lse_ref = fmha.memory_efficient_attention_forward_requires_grad(
        xq, xk, xv, attn_bias, op=op
    )

    # Fill K/V with NaNs at padding positions.
    mask_uninitialized = (
        torch.arange(max_len_kv, device="cuda")[None, :].expand(B, max_len_kv)
        >= seqlens[:, None]
    )
    mask_uninitialized = mask_uninitialized[:, :, None, None]
    k.masked_fill_(mask_uninitialized, float("nan"))
    v.masked_fill_(mask_uninitialized, float("nan"))

    if op in [fmha.flash3.FwOp, fmha.flash3.FwOp_KVSplit]:
        # NOTE: FA3 without paged attention uses TMA to load KV from global memory to shared memory based on a fixed length,
        # which may load NaN elements for variable length cases.
        # So the current FA3 implementation (TMA KV loading) cannot handle NaNs during V loading, which can propagate NaNs to the output.
        # NaNs can occur in pre-allocated KV cache initialized with torch.empty()
        # In these cases, FA3 paged attention should be used instead, which uses cp.async to load KV from global memory to shared memory
        # based on the variable length (see: https://fburl.com/9q0rsyco)
        make_paged_kwargs = {
            "paged_type": fmha.attn_bias.PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        }

        block_tables = torch.arange(
            B * padded_per_row_len // page_size, device="cuda", dtype=torch.int32
        ).reshape(B, -1)

        attn_bias_paged = attn_bias.make_paged(
            block_tables=block_tables,
            page_size=page_size,
            **make_paged_kwargs,  # type: ignore
        )
        axq = q.view(1, -1, nheads_q, head_dim)
        axk_padded = k.view(1, -1, nheads_kv, head_dim).expand(
            1, -1, nheads_q, head_dim
        )
        axv_padded = v.view(1, -1, nheads_kv, head_dim).expand(
            1, -1, nheads_q, head_dim
        )

        if type(attn_bias_paged) not in op.SUPPORTED_ATTN_BIAS_TYPES:
            pytest.skip(f"Op {op.NAME} doesn't support {type(attn_bias)}")
        out, lse = fmha.memory_efficient_attention_forward_requires_grad(
            axq,
            axk_padded,
            axv_padded,
            attn_bias_paged,
            op=op,
        )
        torch.testing.assert_close(out, out_ref, atol=4e-3, rtol=1e-4)

        torch.testing.assert_close(lse, lse_ref, atol=4e-3, rtol=1e-4)
    out, lse = fmha.memory_efficient_attention_forward_requires_grad(
        xq,
        xk,
        xv,
        attn_bias,
        op=op,
    )

    torch.testing.assert_close(out, out_ref, atol=4e-3, rtol=1e-4)

    torch.testing.assert_close(lse, lse_ref, atol=4e-3, rtol=1e-4)


# end of file
