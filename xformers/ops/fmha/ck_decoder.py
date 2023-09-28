# TODO(max): add a proper copyright header
import math
import torch

from typing import Any, Set, List, Tuple, Optional
from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs
from ..common import get_xformers_operator, register_operator

@register_operator
class FwOp(AttentionFwOpBase):
    OPERATOR = get_xformers_operator("efficient_attention_forward_decoder_ck")
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16, torch.float}
    SUPPORTED_MAX_K: float = 256
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {BlockDiagonalCausalWithOffsetPaddedKeysMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    NAME = "ck_decoderF"

    @classmethod
    def not_supported_reasons(cls, d: Inputs) -> List[str]:
        reasons = super(FwOp, cls).not_supported_reasons(d)

        attn_bias = d.attn_bias
        if isinstance(attn_bias, BlockDiagonalCausalWithOffsetPaddedKeysMask):
            # If we don't get here, we've an error elsewhere
            if d.query.ndim != 4 or d.key.ndim != 4:
                reasons.append("Inputs must be BMHK. BMK not supported")

            if d.query.shape[0] != 1:
                reasons.append("One formal batch element expected")

            if d.query.shape[-1] != 256:
                reasons.append("Only head_dim==256 for now.")

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

        attn_bias.k_seqinfo.to(inp.query.device)
        attn_bias.q_seqinfo.to(inp.query.device)

        padding = attn_bias.k_seqinfo.padding
        multiquery = inp.key.stride(2) == 0
        if multiquery:
            key = inp.key[0, :, :1].unflatten(0, (-1, padding))
            value = inp.value[0, :, :1].unflatten(0, (-1, padding))
        else:
            key = inp.key[0].unflatten(0, (-1, padding))
            value = inp.value[0].unflatten(0, (-1, padding))

        seq_positions = attn_bias.k_seqinfo.seqlen

        query = inp.query[0, :, None]

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = 1.0 / math.sqrt(key.shape[-1])

        out = cls.OPERATOR(
            query=query,
            key=key,
            value=value,
            seq_positions=seq_positions,
            scale=qk_scale,
        )
        return out, None
