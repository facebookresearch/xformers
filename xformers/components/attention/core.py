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


def _sparse_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # need to use torch.sparse.mm to get gradients wrt sparse matrix a
    # TODO implement this in C++ / CUDA as this is slow!
    out = []
    for ai, bi in zip(a, b):
        out.append(torch.sparse.mm(ai, bi))
    return torch.stack(out, dim=0)


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
            att = att.to_dense()
        att = dropout(att)

    # Get to the predicted values, for all heads
    # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    y = bmm(att, v)
    return y
