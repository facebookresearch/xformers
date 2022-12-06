from dataclasses import replace
from types import SimpleNamespace

from .common import AttentionOpDispatch
from .cutlass import Op as MemoryEfficientAttentionCutlassOp
from .flash import Op as MemoryEfficientAttentionFlashAttentionOp


class MemoryEfficientAttentionCutlassFwdFlashBwOp(MemoryEfficientAttentionCutlassOp):
    """An operator that uses :attr:`xformers.ops.MemoryEfficientAttentionCutlassOp` for the forward pass \
        and :attr:`xformers.ops.MemoryEfficientAttentionFlashAttentionOp` for the backward.
    """

    FW_OP = MemoryEfficientAttentionCutlassOp
    BW_OP = MemoryEfficientAttentionFlashAttentionOp
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTED_DTYPES = BW_OP.SUPPORTED_DTYPES.intersection(FW_OP.SUPPORTED_DTYPES)

    NAME = "fctls_bflsh"

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if d.requires_grad and not cls.BW_OP.supports(d):
            return False
        return cls.FW_OP.supports(replace(d, requires_grad=False))

    @classmethod
    def backward(cls, ctx, grad):
        query, key, value, lse, out = ctx.saved_tensors
        ctx_flash = SimpleNamespace()

        ctx_flash.causal = ctx.causal
        ctx_flash.dropout_p = 0.0
        query, key, value, cu_seqlens_k, cu_seqlens_q = cls.BW_OP.prepare_inputs(
            ctx_flash, query, key, value
        )
        ctx_flash.kernel_output_shape = (query.shape[0], query.shape[1], value.shape[2])
        ctx_flash.softmax_scale = (
            query.shape[-1] ** (-0.5) if ctx.scale is None else ctx.scale
        )
        rng_state = None

        out = out.reshape(ctx_flash.kernel_output_shape)
        grad = grad.reshape(ctx_flash.kernel_output_shape)
        return cls.BW_OP._backward(
            ctx_flash,
            grad,
            [query, key, value, out, lse, cu_seqlens_q, cu_seqlens_k, rng_state],
        )
