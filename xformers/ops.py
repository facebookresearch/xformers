# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from typing import Any, List, Optional, Set, Type, Union

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


def _get_xformers_operator(name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            f"No such operator xformers::{name} - did you forget to build xformers with `python setup.py develop`?"
        )

    try:
        return getattr(torch.ops.xformers, name)
    except (RuntimeError, AttributeError):
        return no_such_operator


def _ref_attention(
    query, key, value, compute_logsumexp: bool, attn_bias=None, p: float = 0.0
):
    query = query * (1.0 / query.shape[-1] ** 0.5)
    if attn_bias is None:
        attn = query @ key.transpose(-2, -1)
    else:
        # equivalent to (query @ key.transpose(-2, -1) + m).softmax(-1) @ v
        # but faster, and is what is used in PyTorch now
        attn = torch.baddbmm(attn_bias, query, key.transpose(-2, -1))
    dtype = attn.dtype
    attn = attn.to(torch.float).softmax(-1).to(dtype)
    if p > 0:
        attn = torch.nn.functional.dropout(attn, p=p)
    rng_seed = 0
    rng_offset = 0
    return (
        attn @ value,
        attn.logsumexp(-1) if compute_logsumexp else None,
        rng_seed,
        rng_offset,
    )


class AttentionOpBase(torch.autograd.Function):
    """
    Manually doing what our efficient kernels do with Pytorch.
    Allows to support forward/backwards when not implemented otherwise
    """

    FORWARD_OPERATOR: Any
    SUPPORTED_DEVICES: Set[str]
    SUPPORTED_DTYPES: Set[torch.dtype]
    SUPPORTED_MAX_K: float
    SUPPORTS_ATTN_BIAS: bool
    SUPPORTS_DROPOUT: bool
    NAME: str

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p):
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

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        device_type = d.device if isinstance(d.device, str) else d.device.type
        if device_type not in cls.SUPPORTED_DEVICES:
            return False
        if d.dtype not in cls.SUPPORTED_DTYPES:
            return False
        if d.k > cls.SUPPORTED_MAX_K:
            return False
        if d.has_attn_bias and not cls.SUPPORTS_ATTN_BIAS:
            return False
        if d.has_dropout and not cls.SUPPORTS_DROPOUT:
            return False
        return True


class MemoryEfficientAttentionOp(AttentionOpBase):
    FORWARD_OPERATOR = _get_xformers_operator("efficient_attention")
    SUPPORTED_DEVICES = {"cuda", "cpu"}
    SUPPORTED_DTYPES = {torch.float}
    SUPPORTED_MAX_K: float = 32
    SUPPORTS_ATTN_BIAS = True
    SUPPORTS_DROPOUT = True
    NAME = "small_k"

    @staticmethod
    def backward(ctx, grad):
        query, key, value, lse, attn_bias, out = ctx.saved_tensors
        p = ctx.p
        rng_seed = ctx.rng_seed
        rng_offset = ctx.rng_offset
        grad_q, grad_k, grad_v = torch.ops.xformers.efficient_attention_backward(
            grad, query, key, value, lse, out, attn_bias, p, rng_seed, rng_offset
        )
        return grad_q, grad_k, grad_v, None, None


class MemoryEfficientAttentionGenericForwardOp(AttentionOpBase):
    FORWARD_OPERATOR = _get_xformers_operator("efficient_attention_forward_generic")
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {torch.float, torch.half}
    SUPPORTED_MAX_K = math.inf
    SUPPORTS_ATTN_BIAS = False
    SUPPORTS_DROPOUT = False
    NAME = "fwd_gen"

    @classmethod
    def backward(cls, ctx, grad):
        query, key, value, lse, attn_bias, out = ctx.saved_tensors
        p = ctx.p
        rng_seed = ctx.rng_seed
        rng_offset = ctx.rng_offset
        grad_q, grad_k, grad_v = torch.ops.xformers.efficient_attention_backward(
            grad.float(),
            query.float(),
            key.float(),
            value.float(),
            lse.float(),
            out.float(),
            attn_bias,
            p,
            rng_seed,
            rng_offset,
        )
        return grad_q, grad_k, grad_v, None, None


@dataclass
class AttentionOpDispatch:
    dtype: torch.dtype
    device: Union[torch.device, str]
    k: int
    has_dropout: bool
    has_attn_bias: bool

    @property
    def op(self) -> Type[AttentionOpBase]:
        priority_list_ops: List[Type[AttentionOpBase]] = [
            MemoryEfficientAttentionOp,
            MemoryEfficientAttentionGenericForwardOp,
        ]
        for op in priority_list_ops:
            if op.supports(self):
                return op
        raise NotImplementedError(f"No operator found for this attention: {self}")


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    *,
    op=None,
):
    """
    Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    """
    if op is None:
        op = AttentionOpDispatch(
            dtype=query.dtype,
            device=query.device,
            k=query.shape[-1],
            has_dropout=p > 0.0,
            has_attn_bias=attn_bias is not None,
        ).op
    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [query, key, value]):
        return op.FORWARD_OPERATOR(query, key, value, False, attn_bias, p)[0]
    return op.apply(query, key, value, attn_bias, p)
