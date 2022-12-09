# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping, Optional, Set, Union

import torch

from ..common import get_xformers_operator
from .common import AttentionMask, AttentionOpBase, AttentionOpDispatch


class Op(AttentionOpBase):
    """An operator optimized for very small values of K (``K <= 32``) \
        and f32 pre-Ampere as it does not use TensorCores.
    Only supports contiguous inputs in BMK format, so an extra reshape \
        or contiguous call might be done.

    :Deprecated:

        This operator is deprecated and should not be used in new code
    """

    FORWARD_OPERATOR = get_xformers_operator("efficient_attention")
    SUPPORTED_DEVICES = {"cuda", "cpu"}
    SUPPORTED_DTYPES = {torch.float}
    SUPPORTED_MAX_K: float = 32
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), torch.Tensor}
    SUPPORTS_DROPOUT = True
    SUPPORTS_CUSTOM_SCALE = False
    NAME = "small_k"

    BACKWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 4e-3,
    }
    # as this kernel is a bit slow, this should make tests run faster
    _TEST_BATCH_SIZES = [1, 3]
    _TEST_K = [2, 3, 8, 16, 32]

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not super(Op, cls).supports(d):
            return False
        buffer_size = 8
        for pack in [1, 2, 4]:
            if (d.k % pack) == 0 and (d.k // pack) <= buffer_size:
                return True
        return False

    @classmethod
    def bmhk2bmk_contiguous(cls, tensor) -> torch.Tensor:
        return (
            tensor.permute((0, 2, 1, 3))
            .contiguous()
            .view([tensor.shape[0] * tensor.shape[2], tensor.shape[1], tensor.shape[3]])
            .contiguous()
        )

    @classmethod
    def bmk2bmhk(cls, tensor, num_heads: int) -> torch.Tensor:
        return tensor.reshape(
            [-1, num_heads, tensor.shape[1], tensor.shape[2]]
        ).permute((0, 2, 1, 3))

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if scale is not None:
            raise NotImplementedError("Unsupport custom scale")
        num_heads = query.shape[2]
        query = cls.bmhk2bmk_contiguous(query)
        key = cls.bmhk2bmk_contiguous(key)
        value = cls.bmhk2bmk_contiguous(value)
        output = cls._forward_no_grad_bmk(query, key, value, attn_bias=attn_bias, p=p)
        return cls.bmk2bmhk(output, num_heads)

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        if scale is not None:
            raise NotImplementedError("Unsupport custom scale")
        num_heads = query.shape[2]
        query = cls.bmhk2bmk_contiguous(query)
        key = cls.bmhk2bmk_contiguous(key)
        value = cls.bmhk2bmk_contiguous(value)
        output = cls._forward_bmk(ctx, query, key, value, attn_bias=attn_bias, p=p)
        return cls.bmk2bmhk(output, num_heads)

    @classmethod
    def backward(cls, ctx, grad):
        num_heads = grad.shape[2]
        grad = cls.bmhk2bmk_contiguous(grad)
        gq, gk, gv, _, _ = cls._backward_bmk(ctx, grad)
        gq = cls.bmk2bmhk(gq, num_heads)
        gk = cls.bmk2bmhk(gk, num_heads)
        gv = cls.bmk2bmhk(gv, num_heads)
        return gq, gk, gv, None, None, None

    @classmethod
    def _forward_no_grad_bmk(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
    ) -> torch.Tensor:
        return cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=False,
            attn_bias=attn_bias,
            p=p,
        )[0]

    @classmethod
    def _forward_bmk(cls, ctx, query, key, value, attn_bias, p):
        out, lse, rng_seed, rng_offset = cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=True,
            attn_bias=attn_bias,
            p=p,
        )
        ctx.save_for_backward(query, key, value, lse, attn_bias, out)
        ctx.p = p
        ctx.rng_seed = rng_seed
        ctx.rng_offset = rng_offset
        return out

    @staticmethod
    def _backward_bmk(ctx, grad):
        query, key, value, lse, attn_bias, out = ctx.saved_tensors
        p = ctx.p
        rng_seed = ctx.rng_seed
        rng_offset = ctx.rng_offset
        grad_q, grad_k, grad_v = torch.ops.xformers.efficient_attention_backward(
            grad, query, key, value, lse, out, attn_bias, p, rng_seed, rng_offset
        )
        return grad_q, grad_k, grad_v, None, None
