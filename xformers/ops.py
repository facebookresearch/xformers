# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch
import math

def masked_matmul(a, b, mask=None):
    if torch.overrides.has_torch_function((a, b, mask)):
        return torch.overrides.handle_torch_function(
            masked_matmul, (a, b, mask), a, b, mask
        )

    att = a @ b

    if mask is None:
        return att

    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        att += mask
    return att


class MemoryEfficientAttentionOp(torch.autograd.Function):
    FORWARD_OPERATOR = torch.ops.xformers.efficient_attention
    SUPPORTED_DEVICES = {"cuda", "cpu"}
    SUPPORTED_MAX_K = 32
    SUPPORTS_ATTN_BIAS = True
    SUPPORTS_DROPOUT = True

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p):
        out, lse, rng_seed, rng_offset = cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=True,
            attn_bias=attn_bias,
            p=p
        )
        ctx.save_for_backward(query, key, value, lse, attn_bias)
        ctx.p = p
        ctx.rng_seed = rng_seed
        ctx.rng_offset = rng_offset
        return out

    @staticmethod
    def backward(ctx, grad):
        query, key, value, lse, attn_bias = ctx.saved_tensors
        p = ctx.p
        rng_seed = ctx.rng_seed
        rng_offset = ctx.rng_offset
        grad_q, grad_k, grad_v = torch.ops.xformers.efficient_attention_backward(
            grad, query, key, value, lse, attn_bias, p, rng_seed, rng_offset
        )
        return grad_q, grad_k, grad_v, None, None


class MemoryEfficientAttentionGenericForwardOp(MemoryEfficientAttentionOp):
    FORWARD_OPERATOR = torch.ops.xformers.efficient_attention_forward_generic
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_MAX_K = math.inf
    SUPPORTS_ATTN_BIAS = False
    SUPPORTS_DROPOUT = False


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    *,
    op = MemoryEfficientAttentionOp
):
    """
    Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    """
    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [query, key, value]):
        return op.FORWARD_OPERATOR(
            query, key, value, False, attn_bias, p
        )[0]
    return op.apply(query, key, value, attn_bias, p)
