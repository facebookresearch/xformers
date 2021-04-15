import math
from typing import Optional

import torch


def _matmul_with_sparse_mask(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    # TODO implement this in C++ / CUDA
    # to save a bit more memory as we don't need to materialize some temporaries
    assert a.ndim == b.ndim
    assert mask.ndim == a.ndim
    assert a.shape[-1] == b.shape[-2]
    assert a.shape[-2] == mask.shape[-2], f"{a.shape}, {mask.shape}"
    assert b.shape[-1] == mask.shape[-1], f"{b.shape}, {mask.shape}"
    assert a.shape[:-2] == b.shape[:-2], f"{a.shape}, {b.shape}"
    assert a.shape[:-2] == mask.shape[:-2], f"{a.shape}, {mask.shape}"
    idxs = mask.indices().unbind()
    b = b.transpose(-2, -1)

    # compute matmul for elements within the mask
    val = (a[idxs[:-2] + (idxs[-2], slice(None))] * b[idxs[:-2] + (idxs[-1], slice(None))]).sum(-1)  # type: ignore

    out_shape = a.shape[:-1] + (b.shape[-2],)
    res = torch.sparse_coo_tensor(torch.stack(idxs), val, out_shape)
    return res


def _matmul_with_mask(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if mask.is_sparse:
        return _matmul_with_sparse_mask(a, b, mask)

    res = a @ b
    res[~mask] = float("-inf")
    return res


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
    att_mask: torch.Tensor,
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
        att = dropout(att)

    # Get to the predicted values, for all heads
    # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    y = bmm(att, v)
    return y
