# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Set, Tuple

import torch

from ..common import get_xformers_operator, register_operator
from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs


@register_operator
class FwOp(AttentionFwOpBase):
    """
    An operator optimized for K=256 (so the contiguous dim fits into registers).
    Tested to work on MI250x.
    """

    OPERATOR = get_xformers_operator("efficient_attention_forward_decoder_ck")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16, torch.float}
    SUPPORTED_MAX_K: int = 256
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {
        type(None),
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    }
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "ck_decoderF"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)

        attn_bias = d.attn_bias
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            if d.query.shape[0] != 1:
                reasons.append(
                    f"One formal batch element expected; got {d.query.shape[0]}"
                )

            if d.query.shape[-1] > cls.SUPPORTED_MAX_K:
                reasons.append(
                    f"Got head_dim={d.query.shape[-1]}; only head_dim<={cls.SUPPORTED_MAX_K} is supported for now."
                )

            threads_per_warp = 64  # TODO: ideally query the platform here
            required_alignment = 0
            head_dim = d.query.shape[-1]
            for vec_size in (4, 2, 1):
                if head_dim <= vec_size * threads_per_warp:
                    required_alignment = vec_size

            if not required_alignment:
                reasons.append(f"Got head_dim={head_dim} which is too large")

            if head_dim % required_alignment != 0:
                reasons.append(
                    f"Got head_dim={head_dim}; it needs to be divisible by {required_alignment}"
                )

            if d.key.stride(-1) != 1:
                reasons.append("expect keys to have last dim contiguous")

            if d.value.stride(-1) != 1:
                reasons.append("expect values to have last dim contiguous")

            q_starts = attn_bias.q_seqinfo.seqstart_py
            padding = attn_bias.k_seqinfo.padding
            bsz = d.key.shape[1] // padding
            num_queries = d.query.shape[1] // bsz

            if q_starts != list(range(0, 1 + bsz, num_queries)):
                reasons.append("expect to have same num_queries in each batch")
            if bsz != len(q_starts) - 1:
                reasons.append("empty lanes not supported yet")

            if attn_bias.k_seqinfo.padding > 8192:
                reasons.append("key padding exceeds 8192")

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if needs_gradient:
            raise NotImplementedError("backward pass is not supported")
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

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = torch.rsqrt(
                torch.tensor(key.shape[-1], dtype=torch.float32)
            ).item()

        out = cls.OPERATOR(
            query=query,
            key=key,
            value=value,
            seq_positions=seq_positions_gpu,
            scale=qk_scale,
        )
        return out, None
