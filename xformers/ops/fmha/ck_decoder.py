# TODO(max): add a proper copyright header
import torch

from typing import Any, Set, List, Tuple, Optional
from .attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask
from .common import AttentionFwOpBase, Context, Inputs
from ..common import get_xformers_operator, register_operator

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
                reasons.append(f"One formal batch element expected; got {d.query.shape[0]}")

            if d.query.shape[-1] > cls.SUPPORTED_MAX_K:
                reasons.append(f"Got head_dim={d.query.shape[-1]}; only head_dim<={cls.SUPPORTED_MAX_K} is supported for now.")

            threads_per_warp = 64 # TODO: ideally query the platform here
            required_alignment = 0
            head_dim = d.query.shape[-1]
            for vec_size in (4, 2, 1):
                if head_dim <= vec_size * threads_per_warp:
                    required_alignment = vec_size
            
            if not required_alignment:
                reasons.append(f"Got head_dim={head_dim} which is too large")
            
            if head_dim % required_alignment != 0:
                reasons.append(f"Got head_dim={head_dim}; it needs to be divisible by {required_alignment}")

            if d.key.stride(-1) != 1:
                reasons.append("expect keys to have last dim contiguous")

            if d.value.stride(-1) != 1:
                reasons.append("expect values to have last dim contiguous")

            q_starts = attn_bias.q_seqinfo.seqstart_py
            padding = attn_bias.k_seqinfo.padding
            bsz = d.key.shape[1] // padding
            num_queries = d.query.shape[1] // bsz
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

        query = inp.query[0].unflatten(0, (key.shape[0], -1))

        if inp.scale is not None:
            qk_scale = inp.scale
        else:
            qk_scale = torch.rsqrt(torch.tensor(key.shape[-1], dtype=torch.float32))

        out = cls.OPERATOR(
            query=query,
            key=key,
            value=value,
            seq_positions=seq_positions,
            scale=qk_scale,
        )
        return out, None
