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
    """
    Batch matrix multiply between a sparse matrix and a dense matrix
    """
    # approach: convert a batch of 2d sparse matrices A, B, C, ... into
    # a large 2d block-diagonal matrix composed of A, B, C, ...
    # as follows
    #                A  0  0
    # [A, B, C] - >  0  B  0
    #                0  0  C
    # and multiply it by the dense matrix flattened over first 2 dimensions
    # reshaping the result back to the original format
    assert a.ndim == b.ndim == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    B, M, N = a.shape
    K = b.shape[-1]
    a = a.coalesce()
    idxs = a.indices()
    # create indices corresponding to the larger 2d matrix
    i = idxs[1] + idxs[0] * M
    j = idxs[2] + idxs[0] * N
    new_idxs = torch.stack([i, j], dim=0)
    aa = torch.sparse_coo_tensor(new_idxs, a.values(), size=(B * M, B * N))
    bb = b.flatten(0, 1)
    res = torch.sparse.mm(aa, bb)
    res = res.reshape(B, M, K)
    return res


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
