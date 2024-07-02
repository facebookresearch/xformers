# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
import torch

from ..common import get_xformers_operator, register_operator
from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs


@register_operator
class FwOp(AttentionFwOpBase):
    """An operator optimized for very small values of K (``K <= 32``) \
        and f32 pre-Ampere as it does not use TensorCores.
    Only supports contiguous inputs in BMK format, so an extra reshape \
        or contiguous call might be done.

    :Deprecated:

        This operator is deprecated and should not be used in new code
    """

    OPERATOR = get_xformers_operator("efficient_attention_forward_decoder")
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {torch.bfloat16, torch.half, torch.float32}
    CUDA_MINIMUM_COMPUTE_CAPABILITY = (7, 0)
    SUPPORTED_MAX_K: float = 128
    SUPPORTED_ATTN_BIAS_TYPES: Iterable[Any] = (
        BlockDiagonalCausalWithOffsetPaddedKeysMask,
    )
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_BMGHK = True
    NAME = "decoderF"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)

        attn_bias = d.attn_bias
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            # If we don't get here, we've an error elsewhere
            if d.query.shape[0] != 1:
                reasons.append("One formal batch element expected")

            if d.query.shape[-1] != 128:
                reasons.append("Only head_dim==128 for now.")

            if d.key.stride(-1) != 1:
                reasons.append("expect keys to have last dim contiguous")

            if d.value.stride(-1) != 1:
                reasons.append("expect values to have last dim contiguous")

            q_starts = attn_bias.q_seqinfo.seqstart_py
            if attn_bias.q_seqinfo.max_seqlen != 1:
                reasons.append("decoding expects one query")
            elif d.query.shape[1] != len(q_starts) - 1:
                reasons.append("empty lanes not supported yet")

            if attn_bias.k_seqinfo.padding > 8192:
                reasons.append("key padding exceeds 8192")

        return reasons

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if needs_gradient:
            raise NotImplementedError("gradient")
        attn_bias = inp.attn_bias
        assert isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask)
        assert attn_bias.k_seqinfo.seqlen.device == inp.query.device

        padding = attn_bias.k_seqinfo.padding
        query, key, value = inp.get_qkv_in_bmghk()
        query = query[0, :, None]
        key = key[0].unflatten(0, (-1, padding))
        value = value[0].unflatten(0, (-1, padding))

        seq_positions = attn_bias.k_seqinfo.seqlen

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = 1.0 / np.sqrt(key.shape[-1])

        out = cls.OPERATOR(
            query=query,
            key=key,
            value=value,
            seq_positions=seq_positions,
            scale=qk_scale,
        )
        return out, None
