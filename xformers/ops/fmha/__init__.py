# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Optional, Sequence, Tuple, Type, Union

import torch

from . import (
    attn_bias,
    ck,
    ck_decoder,
    ck_splitk,
    cutlass,
    decoder,
    flash,
    small_k,
    triton_splitk,
)
from .attn_bias import (
    AttentionBias,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
)
from .common import (
    AttentionBwOpBase,
    AttentionFwOpBase,
    AttentionOp,
    AttentionOpBase,
    AttentionOpDispatch,
    Context,
    Gradients,
    Inputs,
    bmk2bmhk,
)
from .dispatch import _dispatch_bw, _dispatch_fw, _ensure_op_supports_or_raise

MemoryEfficientAttentionCutlassOp = (cutlass.FwOp, cutlass.BwOp)
MemoryEfficientAttentionCutlassFwdFlashBwOp = (cutlass.FwOp, flash.BwOp)
MemoryEfficientAttentionDecoderOp = (decoder.FwOp, cutlass.BwOp)
MemoryEfficientAttentionFlashAttentionOp = (flash.FwOp, flash.BwOp)
MemoryEfficientAttentionOp = (small_k.FwOp, small_k.BwOp)
MemoryEfficientAttentionCkOp = (ck.FwOp, ck.BwOp)
MemoryEfficientAttentionCkDecoderOp = (ck_decoder.FwOp, ck.BwOp)
MemoryEfficientAttentionSplitKCkOp = (ck_splitk.FwOp, ck.BwOp)


class _fMHA(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, op: AttentionOp, *args: Any) -> Any:
        inp = Inputs(*args)
        op_fw = op[0] if op is not None else None
        op_bw = op[1] if op is not None else None

        out, op_ctx = _memory_efficient_attention_forward_requires_grad(
            inp=inp, op=op_fw
        )

        # Saving attn_bias is a bit complicated, as the
        # torch part should go in `save_for_backward`
        if isinstance(inp.attn_bias, torch.Tensor):
            attn_bias_tensor = inp.attn_bias
            attn_bias_ctx = None
        else:
            attn_bias_tensor = None
            attn_bias_ctx = inp.attn_bias

        ctx.save_for_backward(
            inp.query,
            inp.key,
            inp.value,
            op_ctx.out,
            op_ctx.lse,
        )
        ctx.rng_state = op_ctx.rng_state
        ctx.attn_bias_tensor = attn_bias_tensor
        if op_ctx.op_bw is not None:
            if op_bw is not None and op_bw is not op_ctx.op_bw:
                raise ValueError(
                    f"Specified op_bw={op_bw.NAME}, but forward op "
                    f"can only run with op_bw={op_ctx.op_bw.NAME}. Please set op_bw=None."
                )
            op_bw = op_ctx.op_bw
        ctx.op_fw = op_fw
        ctx.op_bw = op_bw
        ctx.p = inp.p

        ctx.scale = inp.scale
        ctx.attn_bias_ctx = attn_bias_ctx
        ctx.n_args = len(args)
        return out

    @staticmethod
    def deserialize_bias(
        attn_bias_ctx, attn_bias_tensor: Optional[torch.Tensor]
    ) -> Any:
        if attn_bias_tensor is None:
            return attn_bias_ctx
        return attn_bias_tensor

    @classmethod
    @torch.autograd.function.once_differentiable
    def backward(cls, ctx, grad):
        # Re-create context
        query, key, value, out, lse = ctx.saved_tensors
        attn_bias_tensor = ctx.attn_bias_tensor
        rng_state = ctx.rng_state
        inp = Inputs(
            query=query,
            key=key,
            value=value,
            attn_bias=cls.deserialize_bias(ctx.attn_bias_ctx, attn_bias_tensor),
            p=ctx.p,
            scale=ctx.scale,
        )
        op_ctx = Context(
            lse=lse,
            out=out,
            rng_state=rng_state,
        )
        grads = _memory_efficient_attention_backward(
            ctx=op_ctx, inp=inp, grad=grad, op=ctx.op_bw
        )
        return (None, grads.dq, grads.dk, grads.dv, grads.db) + (None,) * (
            ctx.n_args - 2
        )


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[AttentionOp] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Implements the memory-efficient attention mechanism following
    `"Self-Attention Does Not Need O(n^2) Memory" <http://arxiv.org/abs/2112.05682>`_.

    :Inputs shape:

    - Input tensors must be in format ``[B, M, H, K]``, where B is the batch size, M \
        the sequence length, H the number of heads, and K the embeding size per head

    - If inputs have dimension 3, it is assumed that the dimensions are ``[B, M, K]`` and ``H=1``

    - Inputs can also be of dimension 5 with GQA - see note below

    - Inputs can be non-contiguous - we only require the last dimension's stride to be 1


    :Equivalent pytorch code:

    .. code-block:: python

        scale = 1.0 / query.shape[-1] ** 0.5
        query = query * scale
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn = query @ key.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(-1)
        attn = F.dropout(attn, p)
        attn = attn @ value
        return attn.transpose(1, 2)

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

    :EXPERIMENTAL: Using with Multi Query Attention (MQA) and Grouped Query Attention (GQA):

        MQA/GQA is an experimental feature supported only for the forward pass.
        If you have 16 heads in query, and 2 in key/value, you can provide 5-dim tensors
        in the ``[B, M, G, H, K]`` format, where ``G`` is the number of head groups (here 2), and
        ``H`` is the number of heads per group (8 in the example).

        Please note that xFormers will not automatically broadcast the inputs, so you will need
        to broadcast it manually before calling `memory_efficient_attention`.

    :GQA/MQA example:

    .. code-block:: python

        import torch
        import xformers.ops as xops

        B, M, K = 3, 32, 128
        kwargs = dict(device="cuda", dtype=torch.float16)
        q = torch.randn([B, M, 8, K], **kwargs)
        k = torch.randn([B, M, 2, K], **kwargs)
        v = torch.randn([B, M, 2, K], **kwargs)
        out_gqa = xops.memory_efficient_attention(
            q.reshape([B, M, 2, 4, K]),
            k.reshape([B, M, 2, 1, K]).expand([B, M, 2, 4, K]),
            v.reshape([B, M, 2, 1, K]).expand([B, M, 2, 4, K]),
        )

    Raises:
        NotImplementedError: if there is no operator available to compute the MHA
        ValueError: if inputs are invalid

    :parameter query: Tensor of shape ``[B, Mq, H, K]``
    :parameter key: Tensor of shape ``[B, Mkv, H, K]``
    :parameter value: Tensor of shape ``[B, Mkv, H, Kv]``
    :parameter attn_bias: Bias to apply to the attention matrix - defaults to no masking. \
        For common biases implemented efficiently in xFormers, see :attr:`xformers.ops.fmha.attn_bias.AttentionBias`. \
        This can also be a :attr:`torch.Tensor` for an arbitrary mask (slower).
    :parameter p: Dropout probability. Disabled if set to ``0.0``
    :parameter scale: Scaling factor for ``Q @ K.transpose()``. If set to ``None``, the default \
        scale (q.shape[-1]**-0.5) will be used.
    :parameter op: The operators to use - see :attr:`xformers.ops.AttentionOpBase`. \
        If set to ``None`` (recommended), xFormers \
        will dispatch to the best available operator, depending on the inputs \
        and options.
    :return: multi-head attention Tensor with shape ``[B, Mq, H, Kv]``
    """
    return _memory_efficient_attention(
        Inputs(
            query=query,
            key=key,
            value=value,
            p=p,
            attn_bias=attn_bias,
            scale=scale,
            output_dtype=output_dtype,
        ),
        op=op,
    )


def memory_efficient_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Type[AttentionFwOpBase]] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Calculates the forward pass of :attr:`xformers.ops.memory_efficient_attention`.
    """
    return _memory_efficient_attention_forward(
        Inputs(
            query=query,
            key=key,
            value=value,
            p=p,
            attn_bias=attn_bias,
            scale=scale,
            output_dtype=output_dtype,
        ),
        op=op,
    )


def memory_efficient_attention_forward_requires_grad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Type[AttentionFwOpBase]] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple (output, lse), where `lse` can be used to compute the backward pass later.
    See :attr:`xformers.ops.memory_efficient_attention` for an explanation of the arguments
    See :attr:`xformers.ops.memory_efficient_attention_backward` for running the backward pass
    """
    if p != 0.0:
        raise NotImplementedError(
            "dropout is not supported on the non-autograd API."
            " If you want to use dropout, please call `memory_efficient_attention` directly"
        )
    out, ctx = _memory_efficient_attention_forward_requires_grad(
        Inputs(
            query=query,
            key=key,
            value=value,
            p=p,
            attn_bias=attn_bias,
            scale=scale,
            output_dtype=output_dtype,
        ),
        op=op,
    )
    return out, ctx.lse


def memory_efficient_attention_backward(
    grad: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Type[AttentionBwOpBase]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the gradient of the attention.
    Returns a tuple (dq, dk, dv)
    See :attr:`xformers.ops.memory_efficient_attention` for an explanation of the arguments.
    `lse` is the tensor returned by :attr:`xformers.ops.memory_efficient_attention_forward_requires_grad`
    """
    if p != 0.0:
        raise NotImplementedError(
            "dropout is not supported on the non-autograd API."
            " If you want to use dropout, please call `memory_efficient_attention` directly"
        )
    gradients = _memory_efficient_attention_backward(
        Context(out=output, lse=lse),
        Inputs(
            query=query, key=key, value=value, p=p, attn_bias=attn_bias, scale=scale
        ),
        grad,
        op=op,
    )
    return (gradients.dq, gradients.dk, gradients.dv)


def _memory_efficient_attention(
    inp: Inputs, op: Optional[AttentionOp] = None
) -> torch.Tensor:
    # fast-path that doesn't require computing the logsumexp for backward computation
    if all(x.requires_grad is False for x in [inp.query, inp.key, inp.value]):
        return _memory_efficient_attention_forward(
            inp, op=op[0] if op is not None else None
        )

    output_shape = inp.normalize_bmhk()
    return _fMHA.apply(
        op, inp.query, inp.key, inp.value, inp.attn_bias, inp.p, inp.scale
    ).reshape(output_shape)


def _memory_efficient_attention_forward(
    inp: Inputs, op: Optional[Type[AttentionFwOpBase]]
) -> torch.Tensor:
    inp.validate_inputs()
    output_shape = inp.normalize_bmhk()
    if op is None:
        op = _dispatch_fw(inp, False)
    else:
        _ensure_op_supports_or_raise(ValueError, "memory_efficient_attention", op, inp)

    out, *_ = op.apply(inp, needs_gradient=False)
    return out.reshape(output_shape)


def _memory_efficient_attention_forward_requires_grad(
    inp: Inputs, op: Optional[Type[AttentionFwOpBase]]
) -> Tuple[torch.Tensor, Context]:
    inp.validate_inputs()
    output_shape = inp.normalize_bmhk()
    if op is None:
        op = _dispatch_fw(inp, True)
    else:
        _ensure_op_supports_or_raise(ValueError, "memory_efficient_attention", op, inp)
    out = op.apply(inp, needs_gradient=True)
    assert out[1] is not None
    return (out[0].reshape(output_shape), out[1])


def _memory_efficient_attention_backward(
    ctx: Context, inp: Inputs, grad: torch.Tensor, op: Optional[Type[AttentionBwOpBase]]
) -> Gradients:
    """Warning: grad/ctx.out is potentially in BMK format"""
    inp.validate_inputs()
    if grad.ndim != inp.query.ndim or grad.ndim != ctx.out.ndim:
        raise ValueError(
            "All tensors should be either in BMK (ndim=3) or BMHK (ndim=4) format. \n"
            f"grad.shape : {grad.shape} \n"
            f"out.shape  : {ctx.out.shape} \n"
            f"query.shape: {inp.query.shape}"
        )
    shape_dq, shape_dk, shape_dv = tuple(
        x.shape for x in (inp.query, inp.key, inp.value)
    )
    inp.normalize_bmhk()
    # LSE has shape [B, H, M] while query has shape [B, M, H, K]
    if (
        ctx.lse.ndim != 3
        # Dim 0
        or (
            not isinstance(inp.attn_bias, BlockDiagonalMask)
            and ctx.lse.shape[0] != inp.query.shape[0]
        )
        or (
            isinstance(inp.attn_bias, BlockDiagonalMask)
            and ctx.lse.shape[0] != inp.attn_bias.q_seqinfo.seqstart.shape[0] - 1
        )
        # Dim 1
        or ctx.lse.shape[1] != inp.query.shape[2]
        # Dim 2
        or (
            not isinstance(inp.attn_bias, BlockDiagonalMask)
            and ctx.lse.shape[2] < inp.query.shape[1]
        )
    ):
        raise ValueError(
            "Input tensors have incompatible shapes."
            f"lse.shape    : {ctx.lse.shape} \n"
            f"query.shape  : {inp.query.shape}"
        )
    grad = bmk2bmhk(grad, 1)
    ctx.out = bmk2bmhk(ctx.out, 1)

    if op is None:
        op = _dispatch_bw(inp)
    else:
        _ensure_op_supports_or_raise(
            ValueError, "memory_efficient_attention_backward", op, inp
        )

    grads = op.apply(ctx, inp, grad)
    grads.dq = grads.dq.reshape(shape_dq)
    grads.dk = grads.dk.reshape(shape_dk)
    grads.dv = grads.dv.reshape(shape_dv)
    return grads


def memory_efficient_attention_partial(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Type[AttentionFwOpBase]] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple (output, lse), where `output` is the attention and  `lse`
    is a least squared error. The cat'ed outputs of calls to this with the same query
    and separate keys and values can be merged with merge_attentions to obtain
    the attention of the queries against the disjoint union of the keys and values.
    """
    if p != 0.0:
        raise NotImplementedError("dropout is not supported.")
    if not isinstance(
        attn_bias,
        (
            type(None),
            BlockDiagonalCausalWithOffsetPaddedKeysMask,
            LowerTriangularFromBottomRightMask,
            LowerTriangularMask,
        ),
    ):
        raise ValueError(
            f"{type(attn_bias)} is not supported in memory_efficient_attention_partial."
        )
    out, ctx = _memory_efficient_attention_forward_requires_grad(
        Inputs(
            query=query,
            key=key,
            value=value,
            p=p,
            attn_bias=attn_bias,
            scale=scale,
            output_dtype=output_dtype,
            is_partial=True,
        ),
        op=op,
    )
    return out, ctx.lse


def merge_attentions(
    attn_split: torch.Tensor,
    lse_split: torch.Tensor,
    write_lse: bool = True,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Combine attention output computed on different parts of K/V for the same
    query to get attention on the whole K/V. See https://arxiv.org/abs/2402.05099
    The result is equal to
        Out_full = (Out1 * exp(LSE1) + Out2 * exp(LSE2) + ...) / (exp(LSE1) + exp(LSE2) + ...)
        LSE_full = log(exp(LSE1) + exp(LSE2) + ...)
    Attention inputs are in BH(G)MK format, stacked along dim 0. Attention output also is in BH(G)MK.

    Args:
        attn_split: [split_k, B, M, G, H, Kq] or [split_k, B, M, H, Kq]
        lse_split: [split_k, B, G, H, M] or [split_k, B, H, M]
        out_dype: dtype of attn_out

    Returns:
        attn_out: [B, M, G, H, Kq] or [B, M, H, Kq]
        lse_out: [B, G, H, M] or [B, H, M] if write_lse
                 or None otherwise
    """

    assert (
        attn_split.ndim == lse_split.ndim + 1
    ), f"{attn_split.shape=} {lse_split.shape=}"

    is_bmhk = attn_split.ndim == 5
    if is_bmhk:
        attn_split = attn_split.unsqueeze(3)
        lse_split = lse_split.unsqueeze(2)

    split_k, B, M, G, H, Kq = attn_split.shape
    split_k1, B1, G1, H1, M1 = lse_split.shape
    assert B == B1 and G == G1 and H == H1 and split_k == split_k1 and M == M, (
        f"{attn_split.shape=} {lse_split.shape=} "
        f"{B}/{B1}, {G}/{G1}, {H}/{H1}, {split_k}/{split_k1}, {M}/{M}"
    )

    attn_split = attn_split.permute(1, 3, 4, 0, 2, 5).reshape(B, G * H, split_k, M, Kq)
    lse_split = lse_split.permute(1, 2, 3, 0, 4).reshape(B, G * H, split_k, M)

    attn_out = torch.empty(
        B,
        M,
        G,
        H,
        Kq,
        device=attn_split.device,
        dtype=attn_split.dtype if output_dtype is None else output_dtype,
    )
    if write_lse:
        lse_out = torch.empty(
            B * H * G, M, device=attn_split.device, dtype=lse_split.dtype
        )
    else:
        lse_out = None

    triton_splitk.merge_attentions(
        attn_out.permute(0, 1, 3, 2, 4), lse_out, attn_split, lse_split
    )
    if lse_out is not None:
        lse_out = lse_out.view(B, G, H, M)

    if is_bmhk:
        attn_out = attn_out[:, :, 0]
        if lse_out is not None:
            lse_out = lse_out[:, 0]

    return attn_out, lse_out


ALL_FW_OPS: Sequence[Type[AttentionFwOpBase]] = [
    cutlass.FwOp if torch.version.cuda else ck.FwOp,
    flash.FwOp,
    small_k.FwOp,
    triton_splitk.FwOp,
]

ALL_BW_OPS: Sequence[Type[AttentionBwOpBase]] = [
    cutlass.BwOp if torch.version.cuda else ck.BwOp,
    flash.BwOp,
    small_k.BwOp,
]

__all__ = [
    "AttentionBias",
    "AttentionOp",
    "AttentionOpBase",
    "AttentionOpDispatch",
    "LowerTriangularMask",
    "MemoryEfficientAttentionCutlassFwdFlashBwOp",
    "MemoryEfficientAttentionCutlassOp",
    "MemoryEfficientAttentionFlashAttentionOp",
    "MemoryEfficientAttentionOp",
    "memory_efficient_attention",
    "MemoryEfficientAttentionCkOp",
    "MemoryEfficientAttentionCkDecoderOp",
    "ALL_FW_OPS",
    "ALL_BW_OPS",
    "attn_bias",
]
