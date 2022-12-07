# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from types import SimpleNamespace
from typing import Any, Optional, Set, Union

import torch

from .common import (
    AttentionMask,
    AttentionOpBase,
    AttentionOpDispatch,
    LowerTriangularMask,
)

try:
    from ... import _C_flashattention  # type: ignore[attr-defined]

    has_flashattention = True
except ImportError:
    has_flashattention = False


class Op(AttentionOpBase):
    """Operator that computes memory-efficient attention using \
        `Flash-Attention <https://github.com/HazyResearch/flash-attention>`_ \
        implementation.


    This is a wrapper to make FlashAttention compatible with xformers's API
    Most of this code was taken from:
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_interface.py
    """

    FORWARD_OPERATOR = None
    SUPPORTED_DEVICES: Set[str] = {"cuda"}
    SUPPORTED_DTYPES: Set[torch.dtype] = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    SUPPORTS_CUSTOM_SCALE = True
    SUPPORTS_DIFFERENT_VALUE_EMBED = False
    NAME = "flshatt"

    @classmethod
    def info(cls):
        if not has_flashattention:
            return "not built"
        return "available - requires GPU with compute capability 7.5+"

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not has_flashattention:
            return False
        if not super(Op, cls).supports(d):
            return False
        # We know `d.device` is cuda now
        # d=128 is only supported on A100 for bw
        device_capability = torch.cuda.get_device_capability(d.device)
        is_sm80 = device_capability[0] == 8 and device_capability[1] == 0
        if d.k not in [16, 32, 64, 128]:
            return False
        if d.requires_grad and d.k == 128 and not is_sm80:
            return False
        return device_capability >= (7, 5)

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
        return cls.forward(
            ctx=None,
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        )

    @classmethod
    def prepare_inputs(
        cls, ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        batch = query.shape[0]
        seqlen_q = query.shape[1]
        seqlen_k = key.shape[1]
        num_heads = query.shape[2]
        head_dim_q = query.shape[3]
        head_dim_v = value.shape[3]
        ctx.max_seqlen_q = seqlen_q
        ctx.max_seqlen_k = seqlen_k

        cu_seqlens_k = torch.arange(
            0,
            (batch + 1) * seqlen_k,
            step=seqlen_k,
            dtype=torch.int32,
            device=query.device,
        )
        if seqlen_q == seqlen_k:
            cu_seqlens_q = cu_seqlens_k
        else:
            cu_seqlens_q = torch.arange(
                0,
                (batch + 1) * seqlen_q,
                step=seqlen_q,
                dtype=torch.int32,
                device=query.device,
            )

        # Initially we have `query.shape = [batch, seqlen, head_dim_q]`
        # We want format `[batch * seqlen, num_heads, head_dim_q]`
        ctx.query_api_input_shape = query.shape
        ctx.key_api_input_shape = key.shape
        ctx.value_api_input_shape = value.shape
        query = query.reshape([batch * seqlen_q, num_heads, head_dim_q])
        key = key.reshape([batch * seqlen_k, num_heads, head_dim_q])
        value = value.reshape([batch * seqlen_k, num_heads, head_dim_v])
        return query, key, value, cu_seqlens_k, cu_seqlens_q

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        if attn_bias is not None and not isinstance(attn_bias, LowerTriangularMask):
            raise NotImplementedError("Unsupported attn_bias type")
        causal = isinstance(attn_bias, LowerTriangularMask)
        return_softmax = False
        ctx_flash = ctx if ctx is not None else SimpleNamespace()
        query, key, value, cu_seqlens_k, cu_seqlens_q = cls.prepare_inputs(
            ctx_flash, query, key, value
        )

        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if p > 0 else None
        softmax_scale = query.shape[-1] ** (-0.5) if scale is None else scale
        out, softmax_lse, S_dmask = cls._flash_attn_forward(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx_flash.max_seqlen_q,
            ctx_flash.max_seqlen_k,
            p,
            softmax_scale,
            causal=causal,
            return_softmax=return_softmax,
        )
        if ctx is not None:
            ctx.save_for_backward(
                query,
                key,
                value,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                rng_state,
            )
            ctx.dropout_p = p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.kernel_output_shape = out.shape
        return out

    @classmethod
    def backward(cls, ctx, grad):
        return cls._backward(ctx, grad, ctx.saved_tensors)

    @classmethod
    def _backward(cls, ctx, grad, saved_tensors):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        # Create dq,dk,dv
        # If Q/K/V come from a single QKV tensor, let's put the gradient in the
        # right strides, so we can avoid a `cat`
        if (
            q.shape[0] == k.shape[0]
            and q.shape[2] == v.shape[2]
            and q.storage().data_ptr() == k.storage().data_ptr()
            and q.storage().data_ptr() == v.storage().data_ptr()
        ):
            # Create one big contiguous chunk
            # This is because q, k and v usually come from a single
            # output of a linear layer that is chunked.
            # Creating the gradients with the right layout saves us
            # a `torch.cat` call in the backward pass
            chunk = torch.empty(
                (q.shape[0], 3, q.shape[1], q.shape[2]), dtype=q.dtype, device=q.device
            )
            dq = chunk.select(1, 0)
            dk = chunk.select(1, 1)
            dv = chunk.select(1, 2)
        else:
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)

        assert grad.dtype in cls.SUPPORTED_DTYPES
        cls._flash_attn_backward(
            grad.reshape(ctx.kernel_output_shape).contiguous(),
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
        )
        if rng_state is not None:
            torch.cuda.set_rng_state(cur_rng_state)
        dq = dq.reshape(ctx.query_api_input_shape)
        dk = dk.reshape(ctx.key_api_input_shape)
        dv = dv.reshape(ctx.value_api_input_shape)
        return dq, dk, dv, None, None, None

    @staticmethod
    def _flash_attn_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        return_softmax,
    ):
        out, softmax_lse, *rest = _C_flashattention.fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            return_softmax,
            None,
        )
        S_dmask = rest[0] if return_softmax else None
        return out, softmax_lse, S_dmask

    @staticmethod
    def _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
    ):
        softmax_d = _C_flashattention.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            False,
            causal,
            None,
        )
        return dq, dk, dv, softmax_d
