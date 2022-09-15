# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Set, Type, Union

import torch

try:
    from . import _C_flashattention  # type: ignore[attr-defined]

    has_flashattention = True
except ImportError:
    has_flashattention = False


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


class AttentionMask:
    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError()


class LowerTriangularMask(AttentionMask):
    def __init__(self, *tensor_args, **tensor_kwargs) -> None:
        self._tensor: Optional[torch.Tensor] = None
        self._tensor_kwargs = tensor_kwargs
        self._tensor_args = tensor_args

    def to_tensor(self) -> torch.Tensor:
        if self._tensor is None:
            # Work around for "triu_tril_cuda_template" not implemented for 'BFloat16'
            dtype = self._tensor_kwargs.pop("dtype", torch.float)
            create_as = dtype if dtype is not torch.bfloat16 else torch.float32
            self._tensor = torch.full(  # type: ignore
                *self._tensor_args,
                **self._tensor_kwargs,
                dtype=create_as,
                fill_value=float("-inf"),
            )
            self._tensor = torch.triu(self._tensor, diagonal=1).to(dtype)  # type: ignore
        return self._tensor


class AttentionOpBase(torch.autograd.Function):
    """
    Manually doing what our efficient kernels do with Pytorch.
    Allows to support forward/backwards when not implemented otherwise
    """

    FORWARD_OPERATOR: Any
    FORWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-4,
        torch.half: 2e-3,
        torch.bfloat16: 2e-3,
    }
    SUPPORTED_DEVICES: Set[str]
    SUPPORTED_DTYPES: Set[torch.dtype]
    SUPPORTED_MAX_K: float
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None)}
    SUPPORTS_DROPOUT: bool
    NAME: str

    @classmethod
    def forward_no_grad(
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
        if d.attn_bias_type not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            return False
        if d.has_dropout and not cls.SUPPORTS_DROPOUT:
            return False
        return True


class MemoryEfficientAttentionOp(AttentionOpBase):
    FORWARD_OPERATOR = _get_xformers_operator("efficient_attention")
    SUPPORTED_DEVICES = {"cuda", "cpu"}
    SUPPORTED_DTYPES = {torch.float}
    SUPPORTED_MAX_K: float = 32
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), torch.Tensor}
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
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    NAME = "fwd_gen"

    @classmethod
    def forward_no_grad(
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
            attn_bias=None,
            p=p,
            causal=isinstance(attn_bias, LowerTriangularMask),
        )[0]

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p):
        causal = isinstance(attn_bias, LowerTriangularMask)
        out, lse, rng_seed, rng_offset = cls.FORWARD_OPERATOR(
            query=query,
            key=key,
            value=value,
            compute_logsumexp=True,
            attn_bias=None,
            p=p,
            causal=causal,
        )
        ctx.save_for_backward(query, key, value, lse, out)
        ctx.p = p
        ctx.rng_seed = rng_seed
        ctx.rng_offset = rng_offset
        ctx.causal = causal
        return out

    @classmethod
    def uses_tensorcores(cls, d: "AttentionOpDispatch", is_half: bool) -> bool:
        sm_major = torch.cuda.get_device_capability(d.device)[0]
        if sm_major >= 8:
            return True
        if sm_major >= 7:
            return is_half
        return False

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not super(MemoryEfficientAttentionGenericForwardOp, cls).supports(d):
            return False
        cap = torch.cuda.get_device_capability(d.device)
        sm = cap[0] * 10 + cap[1]
        bits_per_scalar = {torch.float: 32, torch.half: 16, torch.bfloat16: 16}[d.dtype]
        uses_tensorcores = cls.uses_tensorcores(d, bits_per_scalar == 16)
        matmul_alignment_mn = 1
        if sm >= 80:
            matmul_alignment_mn = 4
        if uses_tensorcores:
            matmul_alignment_mn = max(matmul_alignment_mn, 64 // bits_per_scalar)
        if d.k % matmul_alignment_mn != 0:
            return False
        return True

    @classmethod
    def backward(cls, ctx, grad):
        query, key, value, lse, out = ctx.saved_tensors
        p = ctx.p
        rng_seed = ctx.rng_seed
        rng_offset = ctx.rng_offset
        dtype = query.dtype
        (
            grad_q,
            grad_k,
            grad_v,
        ) = torch.ops.xformers.efficient_attention_backward_generic(
            grad.to(dtype),
            query,
            key,
            value,
            lse,
            out.to(dtype),
            None,
            p,
            rng_seed,
            rng_offset,
            causal=ctx.causal,
        )
        return grad_q, grad_k, grad_v, None, None


class MemoryEfficientAttentionFlashAttentionOp(AttentionOpBase):
    """
    This is a wrapper to make FlashAttention compatible with xformers's API
    Most of this code was taken from:
    https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_interface.py
    """

    FORWARD_OPERATOR = None
    FORWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.half: 5e-2,
        torch.bfloat16: 5e-2,
    }
    SUPPORTED_DEVICES = {"cuda"}
    SUPPORTED_DTYPES = {torch.half, torch.bfloat16}
    SUPPORTED_MAX_K = 128
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None), LowerTriangularMask}
    SUPPORTS_DROPOUT = False
    NAME = "flshatt"

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        if not has_flashattention:
            return False
        if not super(MemoryEfficientAttentionFlashAttentionOp, cls).supports(d):
            return False
        # We know `d.device` is cuda now
        # d=128 is only supported on A100
        device_capability = torch.cuda.get_device_capability(d.device)
        is_sm80 = device_capability[0] >= 8
        if d.k not in [16, 32, 64, 128] or (d.k == 128 and not is_sm80):
            return False
        if d.dtype is torch.bfloat16 and not is_sm80:
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
    ) -> torch.Tensor:
        return cls.forward(
            ctx=None, query=query, key=key, value=value, attn_bias=attn_bias, p=p
        )

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p):
        causal = isinstance(attn_bias, LowerTriangularMask)
        return_softmax = False

        batch = query.shape[0]
        seqlen_q = query.shape[1]
        seqlen_k = key.shape[1]
        head_dim_q = query.shape[2]
        head_dim_v = query.shape[2]

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
        query_api_input_shape = query.shape
        key_api_input_shape = key.shape
        value_api_input_shape = value.shape
        query = query.reshape([batch * seqlen_q, 1, head_dim_q])
        key = key.reshape([batch * seqlen_k, 1, head_dim_q])
        value = value.reshape([batch * seqlen_k, 1, head_dim_v])

        # Save rng_state because the backward pass will regenerate the dropout mask
        rng_state = torch.cuda.get_rng_state() if p > 0 else None
        softmax_scale = query.shape[-1] ** (-0.5)
        out, softmax_lse, S_dmask = cls._flash_attn_forward(
            query,
            key,
            value,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
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
            ctx.max_seqlen_q = seqlen_q
            ctx.max_seqlen_k = seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.kernel_output_shape = out.shape
            ctx.query_api_input_shape = query_api_input_shape
            ctx.key_api_input_shape = key_api_input_shape
            ctx.value_api_input_shape = value_api_input_shape
        return out.reshape([batch, seqlen_q, head_dim_v])

    @classmethod
    def backward(cls, ctx, grad):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        if rng_state is not None:
            cur_rng_state = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(rng_state)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        assert grad.dtype in cls.SUPPORTED_DTYPES
        cls._flash_attn_backward(
            grad.reshape(ctx.kernel_output_shape),
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
        return dq, dk, dv, None, None

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


@dataclass
class AttentionOpDispatch:
    dtype: torch.dtype
    device: Union[torch.device, str]
    k: int
    has_dropout: bool
    attn_bias_type: Any
    kv_len: int
    q_len: int

    @property
    def op(self) -> Type[AttentionOpBase]:
        priority_list_ops: List[Type[AttentionOpBase]] = [
            MemoryEfficientAttentionFlashAttentionOp,
            MemoryEfficientAttentionGenericForwardOp,
            MemoryEfficientAttentionOp,
        ]
        for op in priority_list_ops:
            if op.supports(self):
                return op
        raise NotImplementedError(f"No operator found for this attention: {self}")

    @classmethod
    def from_arguments(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
        p: float = 0.0,
    ) -> "AttentionOpDispatch":
        return AttentionOpDispatch(
            dtype=query.dtype,
            device=query.device,
            k=query.shape[-1],
            has_dropout=p > 0.0,
            attn_bias_type=type(attn_bias),
            kv_len=value.shape[-2],
            q_len=query.shape[-2],
        )


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
    p: float = 0.0,
    *,
    op=None,
):
    """
    Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    """
    if op is None:
        op = AttentionOpDispatch.from_arguments(
            query=query, key=key, value=value, attn_bias=attn_bias, p=p
        ).op

    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [query, key, value]):
        return op.forward_no_grad(
            query=query, key=key, value=value, attn_bias=attn_bias, p=p
        )
    return op.apply(query, key, value, attn_bias, p)
