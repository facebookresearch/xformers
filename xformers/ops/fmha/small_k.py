# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Optional, Set, Tuple

import torch

from ..common import get_xformers_operator, register_operator
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    Context,
    Gradients,
    Inputs,
    bmk2bmhk,
)


def _bmhk2bmk_contiguous(tensor) -> torch.Tensor:
    return (
        tensor.permute((0, 2, 1, 3))
        .contiguous()
        .view([tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]])
        .contiguous()
    )


@register_operator
class FwOp(AttentionFwOpBase):
    """An operator optimized for very small values of K (``K <= 32``) \
        and f32 pre-Ampere as it does not use TensorCores.
    Only supports contiguous inputs in BMK format, so an extra reshape \
        or contiguous call might be done.

    :Deprecated:

        This operator is deprecated and should not be used in new code
    """

    OPERATOR = get_xformers_operator("efficient_attention")
    SUPPORTED_DEVICES = {"cuda", "cpu"}
    SUPPORTED_DTYPES = {torch.float}
    SUPPORTED_MAX_K: float = 32
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), torch.Tensor}
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = False
    NAME = "smallkF"

    BACKWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 4e-3,
    }
    # as this kernel is a bit slow, this should make tests run faster
    _TEST_BATCH_SIZES = [1, 3]
    _TEST_K = [2, 3, 8, 16, 32]

    @classmethod
    def supports(cls, d: "Inputs") -> bool:
        if not super(FwOp, cls).supports(d):
            return False
        buffer_size = 8
        k = d.query.shape[-1]
        for pack in [1, 2, 4]:
            if (k % pack) == 0 and (k // pack) <= buffer_size:
                return True
        return False

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        if inp.scale is not None:
            raise NotImplementedError("Unsupport custom scale")
        num_heads = inp.query.shape[2]
        query = _bmhk2bmk_contiguous(inp.query)
        key = _bmhk2bmk_contiguous(inp.key)
        value = _bmhk2bmk_contiguous(inp.value)

        out, lse, rng_seed, rng_offset = cls.OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=needs_gradient,
            attn_bias=inp.attn_bias,
            p=inp.p,
        )
        out = bmk2bmhk(out, num_heads)
        lse = lse.reshape([lse.shape[0] // num_heads, num_heads, lse.shape[1]])
        if not needs_gradient:
            return out, None
        ctx = Context(out=out, lse=lse)
        if inp.p != 0.0:
            ctx.op_bw = BwOp
            ctx.rng_state = torch.tensor(
                [rng_seed, rng_offset], dtype=torch.int64, device="cpu"
            )
        return out, ctx


@register_operator
class BwOp(AttentionBwOpBase):
    OPERATOR = get_xformers_operator("efficient_attention_backward")
    SUPPORTED_DEVICES = FwOp.SUPPORTED_DEVICES
    SUPPORTED_DTYPES = FwOp.SUPPORTED_DTYPES
    SUPPORTED_MAX_K = FwOp.SUPPORTED_MAX_K
    SUPPORTED_ATTN_BIAS_TYPES = FwOp.SUPPORTED_ATTN_BIAS_TYPES
    SUPPORTS_DROPOUT = FwOp.SUPPORTS_DROPOUT
    SUPPORTS_CUSTOM_SCALE = FwOp.SUPPORTS_CUSTOM_SCALE
    SUPPORTS_DIFFERENT_VALUE_EMBED = FwOp.SUPPORTS_DIFFERENT_VALUE_EMBED
    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 4e-3,
    }
    NAME = "smallkB"

    @classmethod
    def supports(cls, d: "Inputs") -> bool:
        return FwOp.supports(d)

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        num_heads = grad.shape[2]
        grad = _bmhk2bmk_contiguous(grad)
        query = _bmhk2bmk_contiguous(inp.query)
        key = _bmhk2bmk_contiguous(inp.key)
        value = _bmhk2bmk_contiguous(inp.value)
        out = _bmhk2bmk_contiguous(ctx.out)

        rng_seed = rng_offset = 0
        if inp.p != 0.0:
            if (
                ctx.rng_state is None
                or ctx.rng_state.dtype != torch.int64
                or ctx.rng_state.device.type != "cpu"
                or ctx.rng_state.shape != (2,)
            ):
                raise NotImplementedError(f"Invalid rng_state: {ctx.rng_state}")
            rng_seed, rng_offset = ctx.rng_state.tolist()
        grad_q, grad_k, grad_v = cls.OPERATOR(
            grad,
            query,
            key,
            value,
            # LSE: BHM -> (BH)M
            ctx.lse.reshape([-1, ctx.lse.shape[-1]]),
            out,
            inp.attn_bias,
            inp.p,
            rng_seed,
            rng_offset,
        )
        return Gradients(
            dq=bmk2bmhk(grad_q, num_heads),
            dk=bmk2bmhk(grad_k, num_heads),
            dv=bmk2bmhk(grad_v, num_heads),
        )
