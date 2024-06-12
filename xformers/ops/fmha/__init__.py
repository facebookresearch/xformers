# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional, Sequence, Tuple, Type, Union, cast

import torch

from . import attn_bias
from . import attn_bias as _attn_bias
from . import ck, ck_decoder, ck_splitk, cutlass, decoder, flash, small_k, triton_splitk
from .attn_bias import AttentionBias, BlockDiagonalMask, LowerTriangularMask
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


def _deserialize_bias(attn_bias_ctx, attn_bias_tensor: Optional[torch.Tensor]) -> Any:
    if attn_bias_tensor is None:
        return attn_bias_ctx
    return attn_bias_tensor


# Note: `torch.compile` only allows custom autograd functions
# to accept a subset of types. Therefore we serialize `op` objects
# to `str` before entering the function, and unserialize them inside.
# See also: https://github.com/pytorch/pytorch/issues/118395
_OPS_LOOKUP = {
    flash.FwOp.NAME: flash.FwOp,
    flash.BwOp.NAME: flash.BwOp,
}


def _serialize_op(op):
    if op is not None and op.NAME in _OPS_LOOKUP:
        return op.NAME
    return op


def _unserialize_op(op):
    if isinstance(op, str):
        return _OPS_LOOKUP[op]
    return op


class _fMHA(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(ctx, op_fw, op_bw, *args: Any) -> Any:
        inp = Inputs(*args)

        op_fw = _unserialize_op(op_fw)
        op_bw = _unserialize_op(op_bw)

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
        if op_bw is None and (
            inp.query.requires_grad or inp.key.requires_grad or inp.value.requires_grad
        ):
            is_valid_unpadded_lse = _valid_unpadded_lse_shape(op_ctx.lse, inp)
            # NOTE: We need to check tensor strides to decide which operator we run in the BW pass.
            # Unfortunately, PyTorch only allows to call this function during the FW pass, so
            # we decide the operator to use now.
            op_bw = _dispatch_bw(inp, is_valid_unpadded_lse)
        ctx.op_fw = op_fw
        ctx.op_bw = op_bw
        ctx.p = inp.p
        # This allows to create gradients from a single storage,
        # to avoid a "cat" in the BW pass.
        # The heuristic is approximative, but:
        # (1) It's not a big issue to create a shared storage
        # (2) The heuristic needs to pass `torch.compile`
        #  (this is also why we run it in the FW pass, the BW pass is stricter)
        ctx.qkv_share_storage = (
            inp.query.shape[0] == inp.key.shape[0]
            and inp.query.shape[-1] == inp.value.shape[-1]
            and inp.query.stride(-2)
            == (inp.key.shape[-1] + inp.query.shape[-1] + inp.value.shape[-1])
        )

        ctx.scale = inp.scale
        ctx.attn_bias_ctx = attn_bias_ctx
        ctx.n_args = len(args)
        return out, op_ctx.lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad, grad_lse):
        # Re-create context
        query, key, value, out, lse = ctx.saved_tensors
        attn_bias_tensor = ctx.attn_bias_tensor
        rng_state = ctx.rng_state
        inp = Inputs(
            query=query,
            key=key,
            value=value,
            attn_bias=_deserialize_bias(ctx.attn_bias_ctx, attn_bias_tensor),
            p=ctx.p,
            scale=ctx.scale,
        )
        op_ctx = Context(
            lse=lse,
            out=out,
            rng_state=rng_state,
        )
        grads = _memory_efficient_attention_backward(
            ctx=op_ctx,
            inp=inp,
            grad=grad,
            op=ctx.op_bw,
            _skip_op_checks=True,
        )
        return (None, None, grads.dq, grads.dk, grads.dv, grads.db) + (None,) * (
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
    `lse` is the tensor returned by
    :attr:`xformers.ops.memory_efficient_attention_forward_requires_grad`
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

    op_fw = _serialize_op(op[0] if op is not None else None)
    op_bw = _serialize_op(op[1] if op is not None else None)
    return _fMHA.apply(
        op_fw, op_bw, inp.query, inp.key, inp.value, inp.attn_bias, inp.p, inp.scale
    )[0].reshape(output_shape)


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


def _valid_padded_lse_shape(lse, inp):
    invalid_shape = (
        lse.ndim != 3
        # Dim 0
        or (
            not isinstance(inp.attn_bias, BlockDiagonalMask)
            and lse.shape[0] != inp.query.shape[0]
        )
        or (
            isinstance(inp.attn_bias, BlockDiagonalMask)
            and lse.shape[0] != inp.attn_bias.q_seqinfo.seqstart.shape[0] - 1
        )
        # Dim 1
        or lse.shape[1] != inp.query.shape[2]
        # Dim 2
        or (
            not isinstance(inp.attn_bias, BlockDiagonalMask)
            and lse.shape[2] < inp.query.shape[1]
        )
    )
    return not invalid_shape


def _valid_unpadded_lse_shape(lse, inp):
    return (
        inp.query.ndim == 4
        and lse.ndim == 2
        # Dim 0
        and lse.shape[0] == inp.query.shape[2]
        # Dim 1
        and lse.shape[1] == inp.attn_bias.q_seqinfo.seqstart_py[-1]
        and isinstance(
            inp.attn_bias,
            (
                _attn_bias.BlockDiagonalMask,
                _attn_bias.BlockDiagonalGappyKeysMask,
                _attn_bias.PagedBlockDiagonalPaddedKeysMask,
                _attn_bias.BlockDiagonalPaddedKeysMask,
            ),
        )
    )


def _memory_efficient_attention_backward(
    ctx: Context,
    inp: Inputs,
    grad: torch.Tensor,
    op: Optional[Type[AttentionBwOpBase]],
    *,
    _skip_op_checks: bool = False,
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
    # LSE has shape [B, H, M] or [H, total_q_len], while query has shape [B, M, H, K] or [1, total_q_len, H, K]
    support_unpadded_lse = op is None or op.SUPPORTS_UNPADDED_LSE
    is_valid_unpadded_lse = support_unpadded_lse and _valid_unpadded_lse_shape(
        ctx.lse, inp
    )
    is_valid_padded_lse = _valid_padded_lse_shape(ctx.lse, inp)
    if not (is_valid_padded_lse or is_valid_unpadded_lse):
        raise ValueError(
            "Input tensors have incompatible shapes."
            f"lse.shape    : {ctx.lse.shape} \n"
            f"query.shape  : {inp.query.shape}"
        )
    grad = bmk2bmhk(grad, 1)
    ctx.out = bmk2bmhk(ctx.out, 1)

    if op is None:
        op = _dispatch_bw(inp, is_valid_unpadded_lse)
    elif not _skip_op_checks:
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
    op: Optional[Union[AttentionOp, Type[AttentionFwOpBase]]] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns a tuple (output, lse), where `output` is the attention in the style of
    memory_efficient_attention, and  `lse` is extra data, a log-sum-exp.
    The outputs of calls to this with the same query and separate keys and values
    can be merged with merge_attentions to obtain the attention of the queries
    against the disjoint union of the keys and values.

    Warning: The backward pass of this function is quite restricted. In particular
    we assume that in the forward pass the outputs were only used in merge_attention
    calculations, and that LSEs weren't used anywhere except in merge attentions.
    """
    if p != 0.0:
        raise NotImplementedError("dropout is not supported.")
    fwop: Optional[Type[AttentionFwOpBase]] = op[0] if isinstance(op, tuple) else op
    if not (
        isinstance(
            attn_bias,
            (
                type(None),
                _attn_bias.BlockDiagonalGappyKeysMask,
                _attn_bias.BlockDiagonalPaddedKeysMask,
                _attn_bias.PagedBlockDiagonalGappyKeysMask,
                _attn_bias.PagedBlockDiagonalPaddedKeysMask,
                _attn_bias.LowerTriangularFromBottomRightMask,
                _attn_bias.LowerTriangularMask,
            ),
        )
        or fwop is None
        or fwop.UNPADDED_LSE
    ):
        raise ValueError(
            f"{type(attn_bias)} is not supported in memory_efficient_attention_partial."
        )
    inp = Inputs(
        query=query,
        key=key,
        value=value,
        p=p,
        attn_bias=attn_bias,
        scale=scale,
        output_dtype=output_dtype,
        is_partial=True,
    )

    is_grad = torch.is_grad_enabled() and any(
        x.requires_grad for x in [query, key, value]
    )

    if not is_grad:
        out, ctx = _memory_efficient_attention_forward_requires_grad(
            inp,
            op=fwop,
        )
        return out, ctx.lse

    if query.ndim == 5:
        raise ValueError("gradients not supported for 5D tensors")
    if isinstance(op, tuple):
        op_fw = _serialize_op(op[0])
        op_bw = _serialize_op(op[1])
    elif op is None:
        op_fw = op_bw = None
    else:
        op_fw = _serialize_op(op)
        op_bw = None
    return _fMHA.apply(
        op_fw,
        op_bw,
        inp.query,
        inp.key,
        inp.value,
        inp.attn_bias,
        inp.p,
        inp.scale,
        inp.output_dtype,
        inp.is_partial,
    )


def merge_attentions(
    attn_split: Union[torch.Tensor, Sequence[torch.Tensor]],
    lse_split: Union[torch.Tensor, Sequence[torch.Tensor]],
    write_lse: bool = True,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Combine attention output computed on different parts of K/V for the same
    query to get attention on the whole K/V. See https://arxiv.org/abs/2402.05099
    The result is equal to
        Out_full = (Out1 * exp(LSE1) + Out2 * exp(LSE2) + ...) / (exp(LSE1) + exp(LSE2) + ...)
        LSE_full = log(exp(LSE1) + exp(LSE2) + ...)

    Args:
        attn_split: attention outputs for chunks,
            either as a list of tensors of shapes [B, M, G, H, Kq] or [B, M, H, Kq]
            or as a single tensor of shape [num_chunks, B, M, G, H, Kq]
            or [num_chunks, B, M, H, Kq]
        lse_split: LSE for chunks,
            either as a list of tensors of shapes [B, G, H, M] or [B, H, M]
            or as a single tensor of shape [num_chunks, B, G, H, M] or [num_chunks, B, H, M]
        write_lse: whether to output LSE
        out_dype: dtype of attn_out

    Returns:
        attn_out: [B, M, G, H, Kq] or [B, M, H, Kq]
        lse_out: [B, G, H, M] or [B, H, M] if write_lse
                 or None otherwise
    """

    attn_is_concat = isinstance(attn_split, torch.Tensor)
    lse_is_concat = isinstance(lse_split, torch.Tensor)

    attn_requires_grad = (
        attn_split.requires_grad  # type: ignore
        if attn_is_concat
        else any(x.requires_grad for x in attn_split)
    )
    lse_requires_grad = (
        lse_split.requires_grad  # type: ignore
        if lse_is_concat
        else any(x.requires_grad for x in lse_split)
    )
    requires_grad = torch.is_grad_enabled() and (
        attn_requires_grad or lse_requires_grad
    )
    if requires_grad and not write_lse:
        raise ValueError("write_lse should be true if inputs require gradients.")

    concat_path = attn_is_concat and lse_is_concat and not requires_grad
    if concat_path:
        attn_split = cast(torch.Tensor, attn_split)
        lse_split = cast(torch.Tensor, lse_split)
        if attn_split.ndim != lse_split.ndim + 1:
            raise ValueError(
                f"Incompatible input shapes: {attn_split.shape=}, {lse_split.shape=}"
            )

        is_bmhk = attn_split.ndim == 5
        if is_bmhk:
            attn_split = attn_split.unsqueeze(3)
            lse_split = lse_split.unsqueeze(2)

        num_chunks, B, M, G, H, Kq = attn_split.shape
        num_chunks1, B1, G1, H1, M1 = lse_split.shape
        if B != B1 or G != G1 or H != H1 or num_chunks != num_chunks1 or M != M:
            raise ValueError(
                f"Incompatible input shapes: {attn_split.shape=} {lse_split.shape=} "
                f"{B}/{B1}, {G}/{G1}, {H}/{H1}, {num_chunks}/{num_chunks1}, {M}/{M}"
            )

        attn_split = attn_split.permute(1, 3, 4, 0, 2, 5)
        lse_split = lse_split.permute(1, 2, 3, 0, 4)

        device = attn_split.device
        attn_dtype = attn_split.dtype
        lse_dtype = lse_split.dtype
    else:
        if attn_is_concat:
            attn_split = attn_split.unbind(0)  # type: ignore
        if lse_is_concat:
            lse_split = lse_split.unbind(0)  # type: ignore
        num_chunks = len(attn_split)
        if len(lse_split) != num_chunks:
            raise ValueError(
                f"Incompatible number of LSE and attention chunks: {len(attn_split)=}, {len(lse_split)=}"
            )

        attn_unsqueezed = []
        lse_unsqueezed = []
        is_bmhk = False
        for i in range(num_chunks):
            if attn_split[i].ndim != lse_split[i].ndim + 1:
                raise ValueError(
                    f"Incompatible input shapes for chunk {i}: {attn_split[i].shape=}, {lse_split[i].shape=}"
                )

            is_bmhk = attn_split[i].ndim == 4
            if is_bmhk:
                attn_unsqueezed.append(attn_split[i].unsqueeze(2))
                lse_unsqueezed.append(lse_split[i].unsqueeze(1))
            else:
                attn_unsqueezed.append(attn_split[i])
                lse_unsqueezed.append(lse_split[i])
        attn_split, lse_split = attn_unsqueezed, lse_unsqueezed

        B, M, G, H, Kq = attn_split[0].shape
        B1, G1, H1, M1 = lse_split[0].shape
        if B != B1 or G != G1 or H != H1 or M != M:
            raise ValueError(
                f"Incompatible input shapes: {attn_split[0].shape=}, {lse_split[0].shape=} "
                f"{B}/{B1}, {G}/{G1}, {H}/{H1}, {M}/{M}"
            )

        for i in range(num_chunks):
            if attn_split[i].shape != (B, M, G, H, Kq):
                raise ValueError(
                    f"Incompatible input shapes for attention chunk {i}: "
                    f"{attn_split[i].shape=}, {(B, M, G, H, Kq)=}"
                )
            if lse_split[i].shape != (B, G, H, M):
                raise ValueError(
                    f"Incompatible input shapes for LSE chunk {i}: "
                    f"{lse_split[i].shape=}, {(B, G, H, M)=}"
                )

            attn_split[i] = attn_split[i].permute(0, 2, 3, 1, 4)  # to (B, G, H, M, Kq)

        device = attn_split[0].device
        attn_dtype = attn_split[0].dtype
        lse_dtype = lse_split[0].dtype

    attn_out = torch.empty(
        B,
        M,
        G,
        H,
        Kq,
        device=device,
        dtype=output_dtype or attn_dtype,
        requires_grad=requires_grad,
    )
    if write_lse:
        lse_out = torch.empty(
            B, G, H, M, device=device, dtype=lse_dtype, requires_grad=requires_grad
        )
    else:
        lse_out = None

    if concat_path:
        triton_splitk.merge_attentions(attn_out, lse_out, attn_split, lse_split)  # type: ignore
    else:
        attn_out, lse_out = _MergeAttentions.apply(attn_out, lse_out, *attn_split, *lse_split)  # type: ignore

    if is_bmhk:
        attn_out = attn_out[:, :, 0]
        if lse_out is not None:
            lse_out = lse_out[:, 0]

    return attn_out, lse_out


class _MergeAttentions(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(
        ctx, attn_out: torch.Tensor, lse_out: torch.Tensor, *inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_chunks = len(inputs) // 2
        attn_split, lse_split = inputs[:num_chunks], inputs[num_chunks:]

        triton_splitk.merge_attentions_varargs(attn_out, lse_out, attn_split, lse_split)

        ctx.save_for_backward(
            attn_out,
            lse_out,
            *inputs,
        )
        return attn_out, lse_out

    @staticmethod
    # type: ignore
    def backward(
        ctx, grad_attn: torch.Tensor, grad_lse: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        out, lse, *inputs = ctx.saved_tensors
        num_chunks = len(inputs) // 2
        attn_split, lse_split = inputs[:num_chunks], inputs[num_chunks:]
        dattn, dlse = triton_splitk.merge_attentions_varargs_backward(
            attn_split,
            lse_split,
            out,
            lse,
            grad_attn,
            grad_lse,
        )
        ret = [None, None] + dattn + dlse
        return tuple(ret)


ALL_FW_OPS: List[Type[AttentionFwOpBase]] = [
    cutlass.FwOp if torch.version.cuda else ck.FwOp,
    flash.FwOp,
    small_k.FwOp,
    triton_splitk.FwOp,
]

ALL_BW_OPS: List[Type[AttentionBwOpBase]] = [
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
