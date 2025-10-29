# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import functools
import sys
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import torch

from ... import _is_triton_available
from ..common import register_operator
from .attn_bias import (
    BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalLocalAttentionPaddedKeysMask,
    BlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
)
from .common import AttentionFwOpBase, check_lastdim_alignment_stride1, Context, Inputs


def _strides(x: Optional[torch.Tensor], *stride_names: str):
    if x is None:
        return {f"stride_{name}": None for name in stride_names}
    assert x.ndim == len(stride_names)
    return {f"stride_{name}": s for name, s in zip(stride_names, x.stride())}


def _is_supported_causal_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            BlockDiagonalCausalWithOffsetGappyKeysMask,
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
            PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
        ),
    )


def _is_supported_local_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            BlockDiagonalCausalLocalAttentionPaddedKeysMask,
            BlockDiagonalLocalAttentionPaddedKeysMask,
        ),
    )


def _is_supported_gappy_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            BlockDiagonalGappyKeysMask,
            PagedBlockDiagonalGappyKeysMask,
        ),
    )


def _is_supported_paged_bias(attn_bias: Any) -> bool:
    return isinstance(
        attn_bias,
        (
            PagedBlockDiagonalGappyKeysMask,
            PagedBlockDiagonalPaddedKeysMask,
        ),
    )


@dataclass
class InputsFp8(Inputs):
    """
    Each of k/v_fp8_scales is an int32 tensor of shape (1, B * Mkv, Hq),
    or (1, page_size * max_pages_per_lane, Hq) in the paged case.
    Each int32 element contains two packed fp16 number
    - scales and shifts for row-wise FP8 quantization.
    """

    k_fp8_scale_shift: Optional[torch.Tensor] = None
    v_fp8_scale_shift: Optional[torch.Tensor] = None

    @property
    def nbytes(self) -> int:
        """
        Number of bytes in the input, not counting the attention bias.
        """
        return (
            super(InputsFp8, self).nbytes
            + (
                self.k_fp8_scale_shift.untyped_storage().nbytes()
                if self.k_fp8_scale_shift is not None
                else 0
            )
            + (
                self.v_fp8_scale_shift.untyped_storage().nbytes()
                if self.v_fp8_scale_shift is not None
                else 0
            )
        )


if TYPE_CHECKING or _is_triton_available():
    from ._triton.splitk_kernels import _fwd_kernel_splitK, _splitK_reduce
else:
    _fwd_kernel_splitK = None
    _splitK_reduce = None


def _is_cuda() -> bool:
    return torch.version.cuda is not None


def _is_cuda_at_least_sm80(device: torch.device) -> bool:
    return _is_cuda() and torch.cuda.get_device_capability(device) >= (
        8,
        0,
    )


@register_operator
class FwOp(AttentionFwOpBase):
    """Flash-Attention with Split-K. Supports fused int4 and fp8 K/V quantization.
    Quantized path will be taken if input K/V have type int32.

    Int4 quantization can be row-wise or group-wise (when cls.NUM_GROUPS > 1) along
    the last dimension of K and V. Currently 1, 2, 4, or 8 groups per row are supported.
    Quantization coefficients (scale and shift) are represented as two
    float16 constants per group, packed into int32. Quantization coefficients of
    all groups are placed at the beginning of the row. So, if unquantized K/V have head
    dimension D, the quantized versions have head dimension D // 8 + NUM_GROUPS
    and dtype int32.
    Pseudocode for dequantizing one row can look like:
    group_size = D // 8
    for i in range(NUM_GROUPS):
        group_start = NUM_GROUPS + i * group_size
        group_quant = K[..., group_start: group_start + group_size]
        scale, shift = unpack_int32_into_float16x2(group_quant[0])
        group_dequant = group_quant[..., 1:] * scale + shift
    ...

    For fp8 only row-wise quantization is supported. To use it, provide input of type
    xformers.ops.fmha.triton_splitk.InputsFp8 (instead of the usual xformers.ops.fmha.Inputs) to
    xformers.ops.fmha.triton_splitk.FwOp.apply or xformers.ops.fmha._memory_efficient_attention_forward.

    This op uses Paged Attention when bias is one of the Paged* classes.
    In this case bias has additional fields:
    - block_tables of shape [batch_size, max_num_pages]
    - K/V of shape [1, max_num_pages * page_size, num_heads, head_dim]
      or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]

    The shape which the kernel takes the queries and the output
    is quite different from the user interface. There are three
    types of input (a) no bias / tensor bias, (b) variable q_len
    (which is only for non causal) and (c) other bias objects.
    From the interface to the kernel the following changes happen.

    (0) In all cases, a group dimension may need to be added.

    (1) For (c), a batch dimension is created, reshaping from (1, B*Mq, G, Hq, K)
        to (B, Mq, G, Hq, K)

    (2) For (a) and (c), in the case of multiquery (i.e. the head dimension
        of keys and values is expanded), the head-swapping trick
        reshaping from (B, Mq, G, Hq, K) to (B, M=Hq*Mq, G, H=1, K)

    (3) For (b), in the case of multiquery, the head-swapping trick
        trick, reshaping from (1, Mq, G, Hq, K) to (1, Mq*Hq, G, H=1, K)
        Note here that Mq is a single long dimension which spans all the queries
        in the batch, unlike in case (C). Also that Hq has to run faster than
        Mq in order that the queries in a batch element remain evenly spaced.

    In all cases, the shape as seen by the kernel is called (Bqq, Mqq, G, H, K).
    The kernel operates on B batch elements and M queries per batch element.
    """

    OPERATOR = True
    SUPPORTED_DEVICES = {"cuda"}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (8, 0)
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
    }  # Those are dtypes of Q. In the quantized case K/V has dtype int32
    SUPPORTED_MAX_K = 512
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        torch.Tensor,
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
        BlockDiagonalLocalAttentionPaddedKeysMask,
        BlockDiagonalGappyKeysMask,
        BlockDiagonalCausalWithOffsetGappyKeysMask,
        BlockDiagonalPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
        PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
        PagedBlockDiagonalGappyKeysMask,
        PagedBlockDiagonalPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    SUPPORTS_OUTPUT_DTYPE = True
    SUPPORTS_PARTIAL = True
    NAME = "triton_splitKF"

    SPLIT_K: Optional[int] = None
    MAX_BLOCK_M = 32

    # Whether blocks attending to no part of a variable sequence length
    # should exit early. This requires extra kernels to run beforehand
    # to initialise the outputs.
    # TODO: avoid these by making the reduce kernel work out it doesn't need
    # to look at the irrelevant places.
    SPLIT_K_EARLY_EXIT: bool = False

    # Perform kernel-level Triton autotune
    AUTOTUNE = False

    NUM_GROUPS = 1  # Default quantization is row-wise
    NUM_GROUPS_VALUES = [1, 2, 4, 8]

    # Values below are used when autotune=False.
    # Note that under certain conditions different values might be used, see the code just before the kernel launch.
    BLOCK_M: int = 16  # When M > 1, different BLOCK_M can be used.
    BLOCK_N: int = 64
    # On AMD or for M > 1 different NUM_STAGES and NUM_WARPS can be used.
    NUM_STAGES: int = 1
    NUM_WARPS: int = 2

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        if K not in {16, 32, 64, 128, 256, 512}:
            reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        if (sys.version_info.major, sys.version_info.minor) < (3, 9):
            reasons.append("triton_splitk requires python 3.9 or above!")
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        if d.key.dtype != torch.int32:
            check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
            check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            # Has only been tested on 8.0 / 9.0.
            if _is_cuda() and not _is_cuda_at_least_sm80(d.device):
                reasons.append(
                    "requires NVidia GPU with sm80 minimum compute capacity, e.g., A100/H100/L4"
                )
            # TODO: AMD GPU support matrix needs to be figured out. MI300X is tested to work.

        q_len = d.query.shape[1]
        is_block_diagonal = isinstance(
            d.attn_bias, (BlockDiagonalPaddedKeysMask, BlockDiagonalGappyKeysMask)
        )
        is_paged = _is_supported_paged_bias(d.attn_bias)
        is_causal = _is_supported_causal_bias(d.attn_bias)
        is_local = _is_supported_local_bias(d.attn_bias)
        if is_block_diagonal or is_paged:
            seqinfo = d.attn_bias.q_seqinfo  # type: ignore
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.max_seqlen
            if is_causal and q_len != seqinfo.min_seqlen:
                reasons.append(
                    f"Variable query len is not supported for causal masks: got {seqinfo.max_seqlen=} {seqinfo.min_seqlen=}."
                )
            elif is_local and q_len != seqinfo.min_seqlen:
                reasons.append(
                    f"Variable query len is not supported for local masks: got {seqinfo.max_seqlen=} {seqinfo.min_seqlen=}."
                )
        if q_len > 16 and (is_causal or is_local):
            # 16 is the minimum BLOCK_M which gets used
            # XXX I don't really understand why this is needed.
            reasons.append(
                "Query length should not be larger than 16 for causal or local attention biases"
            )

        if is_paged:
            page_size = d.attn_bias.page_size  # type: ignore
            if d.key.shape[1] % page_size:
                reasons.append(
                    "For paged attention, key.shape[1] should be divisible "
                    "by the page size, "
                    f"but got {d.key.shape[1]=}, {page_size=}."
                )
            if page_size % cls.BLOCK_N:
                reasons.append(
                    "For paged attention, page size should be divisible "
                    "by the block size, "
                    f"but got {page_size=}, {cls.BLOCK_N=}."
                )

        if isinstance(d.attn_bias, torch.Tensor):
            if d.attn_bias.ndim not in (4, 5):
                reasons.append(
                    "Additive attention bias has to have shape (B, G, H, Mq, Mkv) "
                    f"or (B, H, Mq, Mkv), but got {d.attn_bias.shape}."
                )
            if cls.SPLIT_K is not None and cls.SPLIT_K > 1:
                reasons.append(
                    "Additive attention bias is not supported with split-k > 1."
                )

        return reasons

    @classmethod
    def get_split_k(
        cls, B: int, G: int, H: int, Mk: int, Mq: int, page_size: int, is_paged=False
    ) -> int:
        """Heuristic for the number of splits"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        if torch.version.hip:
            split_k = max(Mk + bh - 1, 1024) // bh
            max_chunk_size = 64
            split_k_stop_val = max(1024 / (B * G * H), 1)
            while split_k > 1 and Mk / (split_k - 1) < max_chunk_size:
                split_k = split_k - 1

            while split_k > split_k_stop_val:
                split_k = split_k // 2

            split_size = (Mk + split_k - 1) // max(split_k, 1)

            chunk_size = split_size // max_chunk_size * max_chunk_size
            if chunk_size < split_size:
                split_k += 1

            split_k_upper_bound = 512
        else:
            if Mq > 1 and B * G * H > 64:
                return 1
            split_k = max(Mk, 1024) // bh
            max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
            split_k_stop_val = Mk / max_chunk_size
            split_k_upper_bound = 64

            while split_k > split_k_stop_val:
                split_k = split_k // 2

        split_k = min(split_k, split_k_upper_bound)
        split_k = max(split_k, 1)

        # makes no sense that split_size is larger than page_size
        if is_paged and torch.version.hip:
            split_size = (Mk + split_k - 1) // split_k
            if split_size > page_size:
                split_size = page_size
                split_k = (Mk + split_size - 1) // split_size

        return split_k

    @classmethod
    def get_kernel(cls):
        from ._triton.splitk_kernels import (
            _fwd_kernel_splitK_autotune,
            _get_splitk_kernel,
        )

        if cls.AUTOTUNE:
            return _fwd_kernel_splitK_autotune[cls.NUM_GROUPS]
        else:
            return _get_splitk_kernel(cls.NUM_GROUPS)

    @classmethod
    def get_fp8_scale_shift(
        cls, inp: Inputs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not hasattr(inp, "k_fp8_scale_shift"):
            return None, None
        inp_ = cast(InputsFp8, inp)
        k_fp8_scale_shift = inp_.k_fp8_scale_shift
        v_fp8_scale_shift = inp_.v_fp8_scale_shift
        assert k_fp8_scale_shift is not None
        assert v_fp8_scale_shift is not None
        if k_fp8_scale_shift.ndim == 3:
            return k_fp8_scale_shift.unsqueeze(2), v_fp8_scale_shift.unsqueeze(2)
        if k_fp8_scale_shift.ndim == 4:
            return k_fp8_scale_shift, v_fp8_scale_shift
        raise ValueError(
            "FP8 scales have to be provided in BMH or BMGH format, "
            f"but got {k_fp8_scale_shift.shape=}"
        )

    @classmethod
    def get_extra_args(
        cls,
        *,
        is_paged: bool,
        B: int,
        M: int,
        Kkv: int,
        Kq: int,
        Mq: int,
        split_k: int,
        attn_bias: Any,
        k_fp8_scale_shift: Any,
    ) -> Dict[str, Any]:
        BLOCK_M = cls.BLOCK_M
        BLOCK_N = cls.BLOCK_N
        if cls.AUTOTUNE:
            extra_args = {}
        else:
            # TODO: remove this when autotuning on AMD is working
            num_warps = cls.NUM_WARPS
            num_stages = cls.NUM_STAGES
            if torch.version.hip and attn_bias is not None:
                # TODO: Double check paged.
                mkv = attn_bias.k_seqinfo.max_seqlen
                # TODO: Determine heuristics for paged attention
                use_fp8_path = k_fp8_scale_shift is not None
                if B == 1:
                    if use_fp8_path:
                        # Use specialized configs for FP8
                        if mkv <= 256:
                            BLOCK_N = 16
                            num_warps = 4
                            num_stages = 1
                        elif mkv <= 2048:
                            BLOCK_N = 32
                            num_warps = 4
                            num_stages = 1
                        elif mkv <= 16384:
                            BLOCK_N = 64
                            num_warps = 4
                            num_stages = 1
                        elif mkv >= 131072:
                            BLOCK_N = 128
                            num_warps = 2
                            num_stages = 1
                        else:
                            # Note: We don't have data for when transitioning num_wraps works well
                            BLOCK_N = 64
                            num_warps = 4
                            num_stages = 1
                    else:
                        num_warps = 4
                        num_stages = 1  # TODO num_stages = 0 gives better perf on AMD, but sometimes produces NaNs
                        BLOCK_N = 32
                elif B <= 4 and split_k <= 128:
                    num_warps = 2
                    num_stages = 1
                    BLOCK_N = 32
                elif B <= 16:
                    if use_fp8_path:
                        if mkv <= 256:
                            BLOCK_N = 16
                            num_warps = 4
                            num_stages = 1
                        elif mkv <= 4096:
                            BLOCK_N = 32
                            num_warps = 4
                            num_stages = 1
                        elif mkv <= 8192:
                            BLOCK_N = 16
                            num_warps = 2
                            num_stages = 1
                        elif mkv < 131072:
                            # Note: This isn't benchmarked, but fp8 seems to scale well.
                            BLOCK_N = 64
                            num_warps = 1
                            num_stages = 1
                        else:
                            BLOCK_N = 128
                            num_warps = 1
                            num_stages = 1
                    else:
                        if M < 16:
                            num_warps = 2
                            num_stages = 1
                        else:
                            num_warps = 1
                            num_stages = 1
                        BLOCK_N = 32
                elif B <= 64 and use_fp8_path:
                    if is_paged:
                        num_stages = 1
                        if mkv <= 256:
                            BLOCK_N = 64
                            num_warps = 8
                        elif mkv <= 8192:
                            BLOCK_N = 64
                            num_warps = 1
                        elif mkv <= 16384:
                            BLOCK_N = 128
                            num_warps = 2
                        else:
                            # Note: This isn't benchmarked, but fp8 seems to scale well.
                            BLOCK_N = 128
                            num_warps = 1
                    else:
                        if mkv <= 256:
                            BLOCK_N = 16
                            num_warps = 4
                            num_stages = 1
                        elif mkv < 131072:
                            # Note: This isn't benchmarked, but fp8 seems to scale well.
                            BLOCK_N = 64
                            num_warps = 1
                            num_stages = 1
                        else:
                            BLOCK_N = 128
                            num_warps = 1
                            num_stages = 1
                elif B <= 128 and use_fp8_path:
                    num_stages = 1
                    if is_paged:
                        if mkv <= 256:
                            num_warps = 4
                            BLOCK_N = 16
                        elif mkv <= 2048:
                            num_warps = 1
                            BLOCK_N = 64
                        elif mkv < 131072:
                            num_warps = 2
                            BLOCK_N = 128
                        else:
                            # Note: This isn't benchmarked, but fp8 seems to scale well.
                            num_warps = 1
                            BLOCK_N = 128
                    else:
                        if mkv <= 128:
                            num_warps = 4
                            BLOCK_N = 16
                        else:
                            num_warps = 1
                            BLOCK_N = 64
                elif B <= 256 and use_fp8_path:
                    num_stages = 1
                    if is_paged:
                        if mkv <= 2048:
                            num_warps = 1
                            BLOCK_N = 64
                        elif mkv < 131072:
                            num_warps = 2
                            BLOCK_N = 128
                        else:
                            # Note: This isn't benchmarked, but fp8 seems to scale well.
                            num_warps = 1
                            BLOCK_N = 128
                    else:
                        if mkv <= 256:
                            num_warps = 2
                            BLOCK_N = 32
                        else:
                            num_warps = 1
                            BLOCK_N = 64
                else:
                    num_warps = 1
                    num_stages = 1
                    BLOCK_N = 64
            else:
                should_modify_warp_and_block = (
                    Kkv == 128
                    and Kq == 128
                    and torch.cuda.get_device_capability() >= (8, 9)
                )
                if should_modify_warp_and_block:
                    if Mq > 1:
                        num_warps = 4
                    # Choose minimal round block size which covers M.
                    if M > 16:
                        BLOCK_M = 32
                    if M > 32:
                        BLOCK_M = 64
                    if M > 64:
                        BLOCK_M = 128
            extra_args = {
                "BLOCK_M": BLOCK_M,
                "BLOCK_N": BLOCK_N,
                "num_warps": num_warps,
                "num_stages": num_stages,
            }
        return extra_args

    @classmethod
    def apply(
        cls,
        inp: Inputs,
        needs_gradient: bool,
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        """
        Note that inp can be of type InputsFp8, in which case K/V are assumed to be row-wise FP8-quantized.
        This is different from int4 quantization, where coefficients are kept together with the quantized
        values at the beginning of each row, and inp has type Inputs.
        """

        output_dtype = inp.get_output_dtype()
        # LSE may need higher precision than output
        output_f64_lse = output_dtype in (torch.float32, torch.float64)
        lse_dtype = torch.float64 if output_f64_lse else torch.float32

        if inp.query.numel() == 0 or inp.key.numel() == 0:
            out = torch.zeros_like(inp.query)
            if needs_gradient:
                lse_out = torch.full(
                    (inp.query.shape[0],)
                    + inp.query.shape[2:-1]
                    + (inp.query.shape[1],),
                    float("-inf"),
                    device=inp.query.device,
                    dtype=lse_dtype,
                )
                return out, Context(out=out, lse=lse_out)
            return out, None

        k_fp8_scale_shift, v_fp8_scale_shift = cls.get_fp8_scale_shift(inp)

        if not isinstance(inp.attn_bias, torch.Tensor):
            attn_bias_tensor = None
            attn_bias = cast(
                Optional[
                    Union[
                        BlockDiagonalCausalWithOffsetPaddedKeysMask,
                        BlockDiagonalCausalLocalAttentionPaddedKeysMask,
                        BlockDiagonalLocalAttentionPaddedKeysMask,
                        BlockDiagonalGappyKeysMask,
                        BlockDiagonalCausalWithOffsetGappyKeysMask,
                        BlockDiagonalPaddedKeysMask,
                        PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
                        PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
                        PagedBlockDiagonalGappyKeysMask,
                        PagedBlockDiagonalPaddedKeysMask,
                    ]
                ],
                inp.attn_bias,
            )
        else:
            attn_bias_tensor = inp.attn_bias
            attn_bias = None

        seq_len = None
        seq_starts_k = None
        seq_starts_q = None
        seq_starts_q_multiplier = None
        q, k, v = inp.get_qkv_in_bmghk()
        IS_CAUSAL = False
        IS_LOCAL = False
        NUM_QUERIES_CAUSAL = 1
        variable_q = False
        window_left = -1
        window_right = -1

        is_block_diagonal = isinstance(attn_bias, BlockDiagonalPaddedKeysMask)
        is_gappy = _is_supported_gappy_bias(attn_bias)
        is_paged = _is_supported_paged_bias(attn_bias)
        if attn_bias is not None:
            assert is_paged or is_block_diagonal or is_gappy
            assert attn_bias.k_seqinfo.seqlen.device == inp.query.device
            seq_len = attn_bias.k_seqinfo.seqlen
            assert seq_len.stride(0) == 1
            if is_gappy:
                seq_starts_k = attn_bias.k_seqinfo.seqstart
                assert seq_starts_k.stride(0) == 1
            assert q.shape[0] == 1
            B = len(seq_len)
            G, Hq, Kq = q.shape[-3:]
            # force a bool because triton cannot take np.bool_
            multiple_q = bool(attn_bias.q_seqinfo.max_seqlen > 1)
            IS_CAUSAL = multiple_q and _is_supported_causal_bias(attn_bias)
            IS_LOCAL = _is_supported_local_bias(attn_bias)
            variable_q = multiple_q and not IS_CAUSAL
            Kkv = v.shape[-1]
            if isinstance(attn_bias, BlockDiagonalLocalAttentionPaddedKeysMask):
                window_left = attn_bias.window_left
                window_right = attn_bias.window_right
            elif isinstance(attn_bias, BlockDiagonalCausalLocalAttentionPaddedKeysMask):
                window_left = attn_bias._window_size - 1

            if variable_q:
                seq_starts_q = attn_bias.q_seqinfo.seqstart
                seq_starts_q_multiplier = 1
                assert seq_starts_q.stride(0) == 1
            else:
                q = q.view(B, -1, G, Hq, Kq)

            kv_shape = (1 if is_paged or is_gappy else B, -1, G, Hq, Kkv)
            k = k.view(kv_shape)
            v = v.view(kv_shape)
            if k_fp8_scale_shift is not None and v_fp8_scale_shift is not None:
                k_fp8_scale_shift = k_fp8_scale_shift.view(kv_shape[:-1])
                v_fp8_scale_shift = v_fp8_scale_shift.view(kv_shape[:-1])

            Mq = q.shape[1]
            NUM_QUERIES_CAUSAL = Mq
        else:
            B, Mq, G, Hq, Kq = q.shape

        if attn_bias_tensor is not None and attn_bias_tensor.ndim == 4:
            # (B, H, Mq, Mkv) -> (B, G, H, Mq, Mkv)
            attn_bias_tensor = attn_bias_tensor.unsqueeze(1)

        # In the case of MQA/GQA, we make q have sequence length (H * Mq) and only one "head".
        mqa_swap_seqlen_head = False
        if (
            k.shape[3] > 1
            and k.stride(3) == 0
            and v.stride(3) == 0
            and attn_bias_tensor is None
        ):
            mqa_swap_seqlen_head = True
            if variable_q:
                seq_starts_q_multiplier = Hq
                assert q.shape[0] == 1
                # The idea is Hq,Mq are reshaped to (M=Mq*Hq, H=1)
                q = q.permute(0, 1, 3, 2, 4).reshape(1, -1, G, 1, Kq)
            else:
                # This is a copy iff Mq, G and H are all > 1.
                # The idea is Hq,Mq are reshaped to (M=Hq*Mq, H=1)
                q = q.permute(0, 3, 1, 2, 4).reshape(q.shape[0], -1, G, 1, Kq)
            k = k[:, :, :, :1]
            v = v[:, :, :, :1]
            if k_fp8_scale_shift is not None and v_fp8_scale_shift is not None:
                k_fp8_scale_shift = k_fp8_scale_shift[:, :, :, :1]
                v_fp8_scale_shift = v_fp8_scale_shift[:, :, :, :1]

        if k.dtype == torch.int32:
            if k_fp8_scale_shift is not None:
                Lk = k.shape[-1] * 4
                PACKED_PER_VAL = 4
            else:
                # Quantized K/V
                PACKED_PER_VAL = 8
                Lk = (k.shape[-1] - cls.NUM_GROUPS) * 8
        else:
            Lk = k.shape[-1]
            PACKED_PER_VAL = 1
            assert cls.NUM_GROUPS == 1, f"{cls.NUM_GROUPS=}"

        _, Mk, G, H, Kkv = k.shape
        Bqq, Mqq, G, H, Kq = q.shape
        assert Lk == Kq, f"Keys have head dim {Lk} but queries have head dim {Kq}"
        if variable_q:
            assert attn_bias is not None
            assert seq_starts_q_multiplier is not None
            M = attn_bias.q_seqinfo.max_seqlen * seq_starts_q_multiplier
        else:
            M = Mqq
        page_size = inp.attn_bias.page_size if is_paged else 0  # type: ignore
        block_tables = None
        kv_cache_blocks_per_row = 0
        if is_paged:
            block_tables = inp.attn_bias.block_tables  # type: ignore
            kv_cache_blocks_per_row = block_tables.shape[1]
            Mk = block_tables.shape[1] * page_size
        elif attn_bias is not None:
            Mk = min(Mk, attn_bias.k_seqinfo.max_seqlen)

        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = (
                cls.get_split_k(B, G, H, Mk, Mq, page_size, is_paged)
                if attn_bias_tensor is None
                else 1
            )

        # M_ceil = Mqq rounded up to a multiple of MAX_BLOCK_M
        M_ceil = (Mqq + cls.MAX_BLOCK_M - 1) // cls.MAX_BLOCK_M * cls.MAX_BLOCK_M
        IS_SPLITK = split_k > 1  # or cls.autotune?
        output_shape = (Bqq, Mq, G, Hq, Kq)
        if IS_SPLITK:
            o_splitk_dtype = (
                torch.float64 if output_dtype == torch.float64 else torch.float32
            )
            if cls.SPLIT_K_EARLY_EXIT:
                o_splitk = torch.zeros(
                    [Bqq, G, H, split_k, M_ceil, Kq],
                    dtype=o_splitk_dtype,
                    device=q.device,
                )
            else:
                o_splitk = torch.empty(
                    [Bqq, G, H, split_k, M_ceil, Kq],
                    dtype=o_splitk_dtype,
                    device=q.device,
                )
        else:
            o_splitk = torch.empty(
                [Bqq, split_k, Mqq, G, H, Kq],
                dtype=output_dtype,
                device=q.device,
            ).permute(0, 3, 4, 1, 2, 5)
        lse, lse_splitk = None, None
        if IS_SPLITK or needs_gradient:
            if IS_SPLITK or output_f64_lse:
                lse_splitk_dtype = torch.float64
            else:
                lse_splitk_dtype = torch.float32
            if cls.SPLIT_K_EARLY_EXIT:
                lse_splitk = torch.full(
                    [Bqq, G, H, split_k, Mqq],
                    -float("inf"),
                    dtype=lse_splitk_dtype,
                    device=q.device,
                )
            else:
                lse_splitk = torch.empty(
                    [Bqq, G, H, split_k, Mqq],
                    dtype=lse_splitk_dtype,
                    device=q.device,
                )

        def grid(META):
            import triton

            return triton.cdiv(M, META["BLOCK_M"]), B * G * H, split_k

        split_size = (Mk + split_k - 1) // split_k
        use_seq_len = seq_len is not None

        kernel = cls.get_kernel()
        extra_args = cls.get_extra_args(
            is_paged=is_paged,
            B=B,
            M=M,
            Kkv=Kkv,
            Kq=Kq,
            Mq=Mq,
            split_k=split_k,
            attn_bias=attn_bias,
            k_fp8_scale_shift=k_fp8_scale_shift,
        )

        if _is_triton_available():
            # Triton 3.3.1+fb is required for AMD specific changes to
            # improve performance.
            # TODO: Remove once the triton update lands everywhere.
            import triton

            IS_TRITON_UPGRADE = triton.__version__ == "3.3.1+fb"
        else:
            IS_TRITON_UPGRADE = False
        IS_HIP = IS_TRITON_UPGRADE and torch.version.hip is not None
        kernel[grid](
            Q=q,
            K=k,
            V=v,
            sm_scale=inp.scale_float,
            Out_splitK=o_splitk,
            LSE_splitk=lse_splitk,
            block_tables=block_tables,
            Seq_len=seq_len,
            Seq_starts_k=seq_starts_k,
            Seq_starts_q=seq_starts_q,
            Seq_starts_q_multiplier=seq_starts_q_multiplier,
            additive_bias=attn_bias_tensor,
            K_fp8_scale_shift=k_fp8_scale_shift,
            V_fp8_scale_shift=v_fp8_scale_shift,
            **_strides(q, "qz", "qm", "qg", "qh", "qk"),
            **_strides(k, "kz", "kn", "kg", "kh", "kk"),
            **_strides(v, "vz", "vn", "vg", "vh", "vk"),
            **_strides(o_splitk, "osk_z", "osk_g", "osk_h", "osk_s", "osk_m", "osk_k"),
            **_strides(lse_splitk, "lsek_z", "lsek_g", "lsek_h", "lsek_s", "lsek_m"),
            **_strides(block_tables, "blocktablesz", "blocktablesl"),
            **_strides(
                attn_bias_tensor, "bias_b", "bias_g", "bias_h", "bias_qm", "bias_km"
            ),
            **_strides(
                k_fp8_scale_shift,
                "k_fp8_scale_shift_z",
                "k_fp8_scale_shift_n",
                "k_fp8_scale_shift_g",
                "k_fp8_scale_shift_h",
            ),
            **_strides(
                v_fp8_scale_shift,
                "v_fp8_scale_shift_z",
                "v_fp8_scale_shift_n",
                "v_fp8_scale_shift_g",
                "v_fp8_scale_shift_h",
            ),
            kv_cache_blocks_per_row=kv_cache_blocks_per_row,
            Z=B,
            H=H,
            G=G,
            N_CTX_Q=M,
            N_CTX_K=Mk,
            BLOCK_N_PER_SPLIT=split_size,
            BLOCK_DMODEL=Lk,
            USE_SEQ_LEN=use_seq_len,
            PACKED_PER_VAL=PACKED_PER_VAL,
            N_GROUPS=cls.NUM_GROUPS,
            IS_CAUSAL=IS_CAUSAL,
            IS_LOCAL=IS_LOCAL,
            NUM_QUERIES_CAUSAL=NUM_QUERIES_CAUSAL,
            IS_SPLITK=IS_SPLITK,
            SPLIT_K_EARLY_EXIT=cls.SPLIT_K_EARLY_EXIT,
            USE_PAGED_ATTENTION=is_paged,
            PAGE_SIZE=page_size,
            WINDOW_LEFT=window_left,
            WINDOW_RIGHT=window_right,
            WRITE_LSE=IS_SPLITK or needs_gradient,
            HAS_ADDITIVE_BIAS=attn_bias_tensor is not None,
            NUM_PROGRAMS_DIM2_CONST=split_k,
            IS_HIP=IS_HIP,
            **extra_args,
        )
        if not IS_SPLITK:
            out = o_splitk[:, :, :, 0]  # Bqq, G, H, Mqq, Kq
            if variable_q and mqa_swap_seqlen_head:
                out = out.view(1, G, Mq, Hq, Kq).permute(0, 2, 1, 3, 4).contiguous()
            else:
                out = out.view(Bqq, G, Hq, Mq, Kq)
                # This is a copy iff mqa_swap_seqlen_head and Mq, G and Hq are all > 1.
                out = out.permute(0, 3, 1, 2, 4).contiguous()
            if needs_gradient:
                assert lse_splitk is not None
                lse = lse_splitk[:, :, :, 0]  # Bqq, G, H, Mqq
                if variable_q and mqa_swap_seqlen_head:
                    lse = lse.view(1, G, Mq, Hq).permute(0, 1, 3, 2)
                else:
                    lse = lse.view(Bqq, G, Hq, Mq)
                    if attn_bias is not None and not variable_q:
                        lse = lse.permute(1, 2, 0, 3).reshape(1, G, Hq, B * Mq)
            else:
                lse = None

            if inp.query.ndim == 4:
                # BMGHK -> BMHK
                assert G == 1
                if lse is not None:
                    lse = lse[:, 0]
                out = out[:, :, 0]

            if lse is None:
                return out, None
            return out, Context(out=out, lse=lse)

        out = torch.empty(output_shape, device=q.device, dtype=output_dtype)

        # Merge attention and LSE outputs from different split-k chunks
        assert lse_splitk is not None
        output_lse = None
        if needs_gradient:
            if attn_bias is None or variable_q:
                output_lse = torch.empty(
                    (Bqq, G, Hq, Mq), device=q.device, dtype=lse_dtype
                )
                lse = output_lse
            else:
                output_lse = torch.empty(
                    (1, G, Hq, B * Mq), device=q.device, dtype=lse_dtype
                )
                lse = output_lse.view(G, Hq, B, Mq).permute(2, 0, 1, 3)

        o_splitk = o_splitk[:, :, :, :, :Mqq]

        if mqa_swap_seqlen_head:
            if variable_q:
                o_splitk = o_splitk.view(Bqq, G, split_k, Mq, Hq, Kq).permute(
                    0, 1, 4, 2, 3, 5
                )
                lse_splitk = lse_splitk.view(Bqq, G, split_k, Mq, Hq).permute(
                    0, 1, 4, 2, 3
                )
            else:
                o_splitk = o_splitk.view(Bqq, G, split_k, Hq, Mq, Kq).permute(
                    0, 1, 3, 2, 4, 5
                )
                lse_splitk = lse_splitk.view(Bqq, G, split_k, Hq, Mq).permute(
                    0, 1, 3, 2, 4
                )

        merge_attentions(out, lse, o_splitk, lse_splitk)

        if inp.query.ndim == 4:
            # BMGHK -> BMHK
            assert G == 1
            out = out[:, :, 0]
            if output_lse is not None:
                output_lse = output_lse[:, 0]
        if Mk == 0:
            out.zero_()

        if attn_bias is not None and not variable_q:
            out = out.view(1, B * Mq, G, Hq, Kq)

        if output_lse is None:
            return out, None

        return out, Context(out=out, lse=output_lse)

    @classmethod
    @functools.lru_cache
    def get_operator(
        cls,
        splitk: int,
        *,
        block_m: Optional[int] = None,
        block_n: Optional[int] = None,
        num_warps: Optional[int] = None,
        num_stages: Optional[int] = None,
        split_k_early_exit: Optional[bool] = None,
    ) -> Type[AttentionFwOpBase]:
        kwargs = {
            "NAME": f"triton_splitK{splitk}",
            "SPLIT_K": splitk,
        }
        if block_m is not None:
            kwargs["BLOCK_M"] = block_m
        if block_n is not None:
            kwargs["BLOCK_N"] = block_n
        if num_warps is not None:
            kwargs["NUM_WARPS"] = num_warps
        if num_stages is not None:
            kwargs["NUM_STAGES"] = num_stages
        if split_k_early_exit is not None:
            kwargs["SPLIT_K_EARLY_EXIT"] = split_k_early_exit
        return type(
            f"FwOp_S{splitk}",
            (cls,),
            kwargs,
        )


def merge_attentions(
    attn_out: torch.Tensor,
    lse_out: Optional[torch.Tensor],
    attn_split: torch.Tensor,
    lse_split: torch.Tensor,
):
    import triton

    from ._triton.splitk_kernels import _splitK_reduce

    B, M, G, H, Kq = attn_out.shape
    B1, G1, H1, split_k, M1, Kq1 = attn_split.shape
    B2, G2, H2, split_k1, M2 = lse_split.shape

    assert (
        B == B1 == B2
        and G == G1 == G2
        and H == H1 == H2
        and M == M1 == M2
        and Kq == Kq1
    ), f"Incompatible shapes: {attn_out.shape=}, {attn_split.shape=}, {lse_split.shape=}"
    assert (
        split_k == split_k1
    ), f"Incompatible shapes: {attn_split.shape=}, {lse_split.shape=}"
    if lse_out is not None:
        B3, G3, H3, M3 = lse_out.shape
        assert (
            B == B3 and G == G3 and H == H3 and M == M3
        ), f"Incompatible shapes: {attn_out.shape=}, {lse_out.shape=}"

    num_warps = 4 if B * G * H < 32 or torch.version.hip else 2
    splitK_pow2 = triton.next_power_of_2(split_k)
    head_dim = attn_out.shape[-1]
    grid = (M, B * G * H, 1)
    _splitK_reduce[grid](
        attn_split,
        lse_split,
        attn_out,
        lse_out,
        split_k=split_k,
        splitK_pow2=splitK_pow2,
        **_strides(attn_split, "osk_z", "osk_g", "osk_h", "osk_s", "osk_m", "osk_k"),
        **_strides(lse_split, "lsek_z", "lsek_g", "lsek_h", "lsek_s", "lsek_m"),
        **_strides(attn_out, "oz", "om", "og", "oh", "ok"),
        **_strides(lse_out, "lse_z", "lse_g", "lse_h", "lse_m"),
        head_dim=head_dim,
        head_dim_pow_2=triton.next_power_of_2(head_dim),
        G=G,
        H=H,
        WRITE_LSE=lse_out is not None,
        num_warps=num_warps,
    )


@torch.library.custom_op(
    "xformers::fmha_merge_attentions_varargs",
    mutates_args=(),
    device_types=["cuda"],
)
def merge_attentions_varargs(
    attn_split: Sequence[torch.Tensor],
    lse_split: Sequence[torch.Tensor],
    write_lse: bool,
    output_dtype: Optional[torch.dtype],
    B: int,
    M: int,
    G: int,
    H: int,
    Kq: int,
) -> List[torch.Tensor]:
    import triton

    from xformers.triton.vararg_kernel import unroll_varargs

    from ._triton.splitk_kernels import _splitK_reduce_varargs

    attn_out = torch.empty(
        (B, M, G, H, Kq),
        device=attn_split[0].device,
        dtype=output_dtype or attn_split[0].dtype,
    )
    if write_lse:
        lse_out = torch.empty(
            (B, G, H, M),
            device=attn_split[0].device,
            dtype=lse_split[0].dtype,
        )
    else:
        lse_out = None
    kernel_args, grid = _prepare_reduce_kernel_params(
        attn_out, lse_out, attn_split, lse_split
    )
    reduce_kernel = unroll_varargs(_splitK_reduce_varargs, N=len(attn_split))
    head_dim = attn_out.shape[-1]
    reduce_kernel[grid](
        *attn_split,
        *lse_split,
        Out=attn_out,
        LSE=lse_out,
        **kernel_args,
        head_dim=head_dim,
        head_dim_pow_2=triton.next_power_of_2(head_dim),
        WRITE_LSE=lse_out is not None,
    )
    if write_lse:
        assert lse_out is not None
        return [attn_out, lse_out]
    return [attn_out]


@torch.library.register_fake("xformers::fmha_merge_attentions_varargs")
def merge_attentions_varargs_fake(
    attn_split: Sequence[torch.Tensor],
    lse_split: Sequence[torch.Tensor],
    write_lse: bool,
    output_dtype: Optional[torch.dtype],
    B: int,
    M: int,
    G: int,
    H: int,
    Kq: int,
) -> List[torch.Tensor]:
    attn_out = torch.empty(
        (B, M, G, H, Kq),
        device=attn_split[0].device,
        dtype=output_dtype or attn_split[0].dtype,
    )
    if write_lse:
        lse_out = torch.empty(
            (B, G, H, M),
            device=attn_split[0].device,
            dtype=lse_split[0].dtype,
        )
        return [attn_out, lse_out]
    return [attn_out]


def _merge_attentions_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad: List[torch.Tensor],
) -> Tuple[None, ...]:
    raise NotImplementedError(
        "Backward pass is not implemented for merge_attentions. "
        "If it was, it would be easy to get wrong attention gradients, "
        "because the gradients of the LSEs "
        "don't get propagated by attention backward."
    )


merge_attentions_varargs.register_autograd(_merge_attentions_backward)


@torch.library.custom_op(
    "xformers::merge_attentions_varargs_backward",
    mutates_args=(),
    device_types=["cuda"],
)
def merge_attentions_varargs_backward(
    attn_split: List[torch.Tensor],
    lse_split: List[torch.Tensor],
    attn_out: torch.Tensor,
    lse_out: torch.Tensor,
    grad_attn: torch.Tensor,
    grad_lse: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    from xformers.triton.vararg_kernel import unroll_varargs

    from ._triton.splitk_kernels import _splitK_reduce_varargs_backward

    dattn_splitk = [torch.empty_like(x) for x in attn_split]
    dlse_splitk = [torch.empty_like(x) for x in lse_split]

    kernel_args, grid = _prepare_reduce_kernel_params(
        attn_out, lse_out, attn_split, lse_split, grad_attn, grad_lse
    )

    reduce_kernel_backward = unroll_varargs(
        _splitK_reduce_varargs_backward, N=len(attn_split)
    )
    reduce_kernel_backward[grid](
        *attn_split,
        *lse_split,
        *dattn_splitk,
        *dlse_splitk,
        Out=attn_out,
        LSE=lse_out,
        DOut=grad_attn,
        DLSE=grad_lse,
        **kernel_args,
        BLOCK_SIZE=attn_out.shape[-1],
    )

    return dattn_splitk, dlse_splitk


@torch.library.register_fake("xformers::merge_attentions_varargs_backward")
def merge_attentions_varargs_backward_fake(
    attn_split: List[torch.Tensor],
    lse_split: List[torch.Tensor],
    attn_out: torch.Tensor,
    lse_out: torch.Tensor,
    grad_attn: torch.Tensor,
    grad_lse: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    dattn_splitk = [torch.empty_like(x) for x in attn_split]
    dlse_splitk = [torch.empty_like(x) for x in lse_split]
    return dattn_splitk, dlse_splitk


def _prepare_reduce_kernel_params(
    attn_out: torch.Tensor,
    lse_out: Optional[torch.Tensor],
    attn_split: Sequence[torch.Tensor],
    lse_split: Sequence[torch.Tensor],
    grad_attn: Optional[torch.Tensor] = None,
    grad_lse: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, int], Tuple[int, int, int]]:
    B, M, G, H, Kq = attn_out.shape
    B1, G1, H1, M1, Kq1 = attn_split[0].shape
    B2, G2, H2, M2 = lse_split[0].shape

    assert (
        B == B1 == B2
        and G == G1 == G2
        and H == H1 == H2
        and M == M1 == M2
        and Kq == Kq1
    ), f"Incompatible shapes: {attn_out.shape=}, {attn_split[0].shape=}, {lse_split[0].shape=}"
    if lse_out is not None:
        B3, G3, H3, M3 = lse_out.shape
        assert (
            B == B3 and G == G3 and H == H3 and M == M3
        ), f"Incompatible shapes: {attn_out.shape=}, {lse_out.shape=}"

    attn_split_strides = {}
    lse_split_strides = {}
    for i in range(len(attn_split)):
        attn_split_strides.update(
            _strides(
                attn_split[i],
                "osk_z" + str(i),
                "osk_g" + str(i),
                "osk_h" + str(i),
                "osk_m" + str(i),
                "osk_k" + str(i),
            )
        )
        lse_split_strides.update(
            _strides(
                lse_split[i],
                "lsek_z" + str(i),
                "lsek_g" + str(i),
                "lsek_h" + str(i),
                "lsek_m" + str(i),
            )
        )

    num_warps = 4 if B * G * H < 32 or torch.version.hip else 2
    grid = (M, B * G * H, 1)

    kernel_args = {
        "G": G,
        "H": H,
        "num_warps": num_warps,
        **attn_split_strides,
        **lse_split_strides,
    }
    kernel_args.update(_strides(attn_out, "oz", "om", "og", "oh", "ok"))
    kernel_args.update(_strides(lse_out, "lse_z", "lse_g", "lse_h", "lse_m"))
    if grad_attn is not None:
        kernel_args.update(_strides(grad_attn, "doz", "dom", "dog", "doh", "dok"))
        kernel_args.update(_strides(grad_lse, "dlse_z", "dlse_g", "dlse_h", "dlse_m"))
    return kernel_args, grid


FwOp_Map = {
    k: FwOp.get_operator(k) for k in [1, 2, 4, 8, 16, 32, 48, 64, 72, 80, 96, 112, 128]
}
FwOp_S1 = FwOp_Map[1]
FwOp_S2 = FwOp_Map[2]
FwOp_S4 = FwOp_Map[4]
FwOp_S8 = FwOp_Map[8]
FwOp_S16 = FwOp_Map[16]
FwOp_S32 = FwOp_Map[32]
FwOp_S64 = FwOp_Map[64]
FwOp_S128 = FwOp_Map[128]
