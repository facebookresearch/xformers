# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, List, Optional, Tuple

import torch

from xformers.ops.common import get_operator, register_operator
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from xformers.ops.fmha.common import (
    AttentionFwOpBase,
    check_lastdim_alignment_stride1,
    Context,
    Inputs,
)


@register_operator
class FwOp(AttentionFwOpBase):

    OPERATOR = get_operator("xformers", "efficient_attention_forward_decoder_splitk_ck")
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {
        torch.half,
        torch.bfloat16,
        torch.float,
    }  # Those are dtypes of Q. In the quantized case K/V has dtype int32
    SUPPORTED_MAX_K = 256
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "ck_splitKF"

    SPLIT_K: Optional[int] = None
    BLOCK_M = 16
    BLOCK_N = 64

    NUM_GROUPS = 1  # Default quantization is row-wise

    @classmethod
    def shape_not_supported_reasons(
        cls, Mq: int, Mkv: int, K: int, Kv: int
    ) -> List[str]:
        reasons = super().shape_not_supported_reasons(Mq, Mkv, K, Kv)
        # if K not in {16, 32, 64, 128}:
        #     reasons.append(f"Embed dim {K} not supported")
        return reasons

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)
        check_lastdim_alignment_stride1(reasons, "query", d.query, 8)
        if d.key.dtype != torch.int32:
            check_lastdim_alignment_stride1(reasons, "key", d.key, 8)
            check_lastdim_alignment_stride1(reasons, "value", d.value, 8)
        if cls.OPERATOR is None:
            reasons.append("triton is not available")
        if d.device.type == "cuda":
            # Has only been tested on 8.0 / 9.0.
            if torch.cuda.get_device_capability(d.device) < (7, 0):
                reasons.append(
                    "requires GPU with sm80 minimum compute capacity, e.g., A100/H100/L4"
                )

        q_len = d.query.shape[1]
        if isinstance(d.attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            seqinfo = d.attn_bias.q_seqinfo
            if q_len != seqinfo.seqstart_py[-1]:
                reasons.append(
                    f"Expected total {seqinfo.seqstart_py[-1]} queries not {q_len}"
                )
            q_len = seqinfo.min_seqlen
            if q_len != seqinfo.max_seqlen:
                reasons.append(
                    "Variable query len is not supported in the presence of causal mask."
                )

        if d.key.ndim in [4, 5] and d.key.shape[-2] != 1:
            if d.key.stride(-2) == 0 and d.value.stride(-2) == 0 and q_len > 1:
                reasons.append("multiquery is only supported with query seqlen=1")

        if d.attn_bias is not None and q_len > 1:
            reasons.append(
                "query with seqlen > 1 is not supported in the presence of causal mask"
            )
        return reasons

    @classmethod
    def get_split_k(cls, B: int, H: int, Mk: int) -> int:
        """Heuristic for the number of splits"""
        bh = max(B * H, 1)  # NOTE: Handle B*h=0 case
        split_k = max(Mk, 1024) // bh
        max_chunk_size = 64 if Mk <= 512 and bh <= 64 else 128
        while split_k > 0 and Mk / split_k < max_chunk_size:
            split_k = split_k // 2
        split_k = min(split_k, 64)
        split_k = max(split_k, 1)
        return split_k

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        attn_bias = inp.attn_bias
        q, k, v = inp.get_qkv_in_bmghk()

        if attn_bias is not None:
            assert isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
            attn_bias.k_seqinfo.to(k.device)
            attn_bias.q_seqinfo.to(q.device)
            padding = attn_bias.k_seqinfo.padding
            seq_positions_gpu = attn_bias.k_seqinfo.seqlen
        else:
            padding = k.shape[1]
            seq_positions_gpu = None

        if attn_bias is not None:
            # key: (1, B * padding, G, 1 if multiquery else Hkv, D)
            # value: like key
            # query: (1, B * q_seqlen, G, Hq, D)
            multiquery = k.stride(3) == 0
            if multiquery:
                key = k[0, :, :, :1].unflatten(0, (-1, padding))
                value = v[0, :, :, :1].unflatten(0, (-1, padding))
            else:
                key = k[0].unflatten(0, (-1, padding))
                value = v[0].unflatten(0, (-1, padding))
            query = q[0].unflatten(0, (key.shape[0], -1))
        else:
            # key: (B, padding, G, 1 if multiquery else Hkv, D)
            # value: like key
            # query: (B, q_seqlen, G, Hq, D)
            key = k
            query = q
            value = v

        B, _, _, H, _ = query.shape
        _, Mk, _, _, _ = key.shape

        if cls.SPLIT_K is not None:
            split_k = cls.SPLIT_K
        else:
            # Use heuristics
            split_k = cls.get_split_k(B, H, Mk)

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = torch.rsqrt(
                torch.tensor(k.shape[-1], dtype=torch.float32)
            ).item()

        out = cls.OPERATOR(
            query=query,
            key=key,
            value=value,
            seq_positions=seq_positions_gpu,
            scale=qk_scale,
            split_k=split_k,
        )

        return out, None


class FwOp_S1(FwOp):
    SPLIT_K = 1
    NAME = "ck_splitK1"


class FwOp_S2(FwOp):
    SPLIT_K = 2
    NAME = "ck_splitK2"


class FwOp_S4(FwOp):
    SPLIT_K = 4
    NAME = "ck_splitK4"


class FwOp_S8(FwOp):
    SPLIT_K = 8
    NAME = "ck_splitK8"


class FwOp_S16(FwOp):
    SPLIT_K = 16
    NAME = "ck_splitK16"


class FwOp_S32(FwOp):
    SPLIT_K = 32
    NAME = "ck_splitK32"


class FwOp_S64(FwOp):
    SPLIT_K = 64
    NAME = "ck_splitK64"


class FwOp_S128(FwOp):
    SPLIT_K = 128
    NAME = "ck_splitK128"
