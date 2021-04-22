import math
from typing import Optional

import torch


def _matmul_with_mask(
    a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if mask is None:
        return a @ b
    return torch.ops.xformers.matmul_with_mask(a, b, mask)


def _softmax(a: torch.Tensor) -> torch.Tensor:
    if a.is_sparse:
        return torch.sparse.softmax(a, dim=a.ndim - 1)
    return torch.softmax(a, dim=a.ndim - 1)


class SparseBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a = a.coalesce()
        r = torch.bmm(a, b)
        ctx.save_for_backward(a, b)
        return r

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors

        # gradients w.r.t. a
        ga = None
        if ctx.needs_input_grad[0]:
            ga = torch.ops.xformers.matmul_with_mask(grad, b.transpose(-2, -1), a)

        # gradients w.r.t. b
        gb = None
        if ctx.needs_input_grad[1]:
            gb = a.transpose(1, 2).bmm(grad)

        return ga, gb


def _sparse_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix multiply between a sparse matrix and a dense matrix
    """
    assert a.ndim == b.ndim == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    return SparseBMM.apply(a, b)


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.is_sparse:
        return _sparse_bmm(a, b)
    return a @ b


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[torch.Tensor],
    dropout: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    # TODO assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
    # this is needed due to limitations in sparse_bmm for now

    # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
    att = _matmul_with_mask(q, k.transpose(-2, -1), att_mask) * (
        1.0 / math.sqrt(k.size(-1))
    )

    # Softmax to get the attention probabilities
    att = _softmax(att)

    #  Optional dropout, could be part of the masking in the future
    if dropout is not None:
        # Dropout chokes on sparse tensors
        if att.is_sparse:
            att = att.coalesce()
            values = att.values().clone()  # protect against in-place droupout
            values = dropout(values)
            att = torch.sparse_coo_tensor(att.indices(), values, att.shape)
        else:
            att = dropout(att)

    # Get to the predicted values, for all heads
    # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    y = bmm(att, v)
    return y
