# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch


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


class _MemoryEfficientAttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, attn_bias, p):
        out, lse, rng_seed, rng_offset = torch.ops.xformers.efficient_attention(
            query, key, value, True, attn_bias, p
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


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
):
    """
    Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    """
    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [query, key, value]):
        return torch.ops.xformers.efficient_attention(
            query, key, value, False, attn_bias, p
        )[0]
    return _MemoryEfficientAttentionOp.apply(query, key, value, attn_bias, p)
