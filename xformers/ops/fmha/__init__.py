# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import torch

from .common import (
    AttentionMask,
    AttentionOp,
    AttentionOpBase,
    AttentionOpDispatch,
    LowerTriangularMask,
)
from .cutlass import Op as MemoryEfficientAttentionCutlassOp
from .flash import Op as MemoryEfficientAttentionFlashAttentionOp
from .mixed import (
    MemoryEfficientAttentionCutlassFwdFlashBwOp,
    MemoryEfficientAttentionTritonFwdFlashBwOp,
)
from .small_k import Op as MemoryEfficientAttentionOp
from .triton import Op as TritonFlashAttentionOp


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[AttentionOp] = None,
) -> torch.Tensor:
    """Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    :Inputs shape:

    - Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M \
        the sequence length, H the number of heads, and K the embeding size per head

    - If inputs have dimension 3, it is assumed that the dimensions are ``[B, M, K]`` and ``H=1``

    - Inputs can be non-contiguous - we only require the last dimension's stride to be 1


    :Equivalent pytorch code:

    .. code-block:: python

        scale = 1 / query.shape[-1] ** 0.5
        query = query * scale
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        attn = F.dropout(attn, p)
        return attn @ value

    :Examples:

    .. code-block:: python

        import xformers.ops as xops

        # Compute regular attention
        y = xops.memory_efficient_attention(q, k, v)

        # With a dropout of 0.2
        y = xops.memory_efficient_attention(q, k, v, p=0.2)

        # Causal attention
        y = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=xops.LowerTriangularMask()
        )

    :Supported hardware:

        NVIDIA GPUs with compute capability above 6.0 (P100+), datatype ``f16``, ``bf16`` and ``f32``.

    Raises:
        NotImplementedError: if there is no operator available to compute the MHA

    :parameter query: Tensor of shape ``[B, Mq, H, K]``
    :parameter key: Tensor of shape ``[B, Mkv, H, K]``
    :parameter value: Tensor of shape ``[B, Mkv, H, Kv]``
    :parameter attn_bias: Bias to apply to the attention matrix - defaults to no masking. \
        For causal attention, use :attr:`xformers.ops.LowerTriangularMask`. \
        This can also be a :attr:`torch.Tensor` for an arbitrary mask.
    :parameter p: Dropout probability. Disabled if set to ``0.0``
    :parameter scale: The scale to query_state weights. If set to ``None``, the default \
        scale (q.shape[-1]**-0.5) will be used.
    :parameter op: The operator to use - see :attr:`xformers.ops.AttentionOpBase`. \
        If set to ``None`` (recommended), xFormers \
        will dispatch to the best available operator, depending on the inputs \
        and options.
    :return: multi-head attention Tensor with shape ``[B, Mq, H, Kv]``
    """

    if query.ndim not in [3, 4]:
        raise ValueError(
            f"Invalid shape for query: {query.shape}. "
            "Expected shape [batch, seqlen, num_heads, K], or [batch, seqlen, K]."
        )
    output_shape = tuple(query.shape[:-1]) + (value.shape[-1],)
    # Convert from legacy format
    if query.ndim == 3:
        query = query.unsqueeze(2)
        key = key.unsqueeze(2)
        value = value.unsqueeze(2)

    if op is None:
        op = AttentionOpDispatch.from_arguments(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        ).op

    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [query, key, value]):
        return op.forward_no_grad(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        ).reshape(output_shape)
    return op.apply(query, key, value, attn_bias, p, scale).reshape(output_shape)


__all__ = [
    "AttentionMask",
    "AttentionOp",
    "AttentionOpBase",
    "AttentionOpDispatch",
    "LowerTriangularMask",
    "MemoryEfficientAttentionCutlassFwdFlashBwOp",
    "MemoryEfficientAttentionTritonFwdFlashBwOp",
    "MemoryEfficientAttentionCutlassOp",
    "MemoryEfficientAttentionFlashAttentionOp",
    "MemoryEfficientAttentionOp",
    "TritonFlashAttentionOp",
    "memory_efficient_attention",
]
