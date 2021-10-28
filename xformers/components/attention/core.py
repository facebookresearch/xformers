# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from contextlib import nullcontext
from typing import Optional

import torch

from xformers import _is_sparse_available

if _is_sparse_available:
    from ._sputnik_sparse import SparseCS

# NOTE: Could do with a better option on when to use triton and not
_use_triton = True
if _use_triton:
    try:
        from xformers.triton.softmax import softmax as triton_softmax
    except ImportError:
        logging.warning(
            "Triton is not available, some optimizations will not be enabled."
        )
        _use_triton = False


def _create_random_sparsity(matrix, sparsity, divisible_by=4):
    assert matrix.ndim == 3
    keep = torch.rand_like(matrix[0], dtype=torch.float32) > sparsity
    nonzero = torch.nonzero(keep)
    nnz = nonzero.shape[0]
    # NOTE: need to make it a multiple of 4 for sputnik
    nonzero = nonzero[: (nnz - nnz % divisible_by)]
    i, j = nonzero.unbind(1)
    output = torch.zeros_like(matrix)
    bdim = torch.arange(matrix.shape[0], device=matrix.device)[:, None]
    output[bdim, i, j] = matrix[bdim, i, j]
    return output


def _broadcast_batch(mask, batch_size):
    if mask.ndim == 3:
        return mask
    assert mask.ndim == 2

    mask = mask.coalesce()
    values = mask.values()
    indices = mask.indices()
    nnz = len(values)
    # strategy: repeat the indices and append the extra batch dimension to the indices
    indices = indices.repeat(1, batch_size)
    # now create the batch indices
    batch_indices = torch.arange(batch_size, device=indices.device)
    batch_indices = batch_indices[:, None].expand(batch_size, nnz).flatten()

    # put them together
    indices = torch.cat([batch_indices[None, :], indices], dim=0)

    # now repeat the values
    values = values.repeat(batch_size)

    size = (batch_size,) + mask.shape

    return torch.sparse_coo_tensor(indices, values, size)


def _matmul_with_mask(
    a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if mask is None:
        return a @ b

    if _is_sparse_available:
        if isinstance(mask, SparseCS):
            return mask.matmul_with_mask(a, b)
        if mask.is_sparse:
            # perform broadcasting if needed
            mask = _broadcast_batch(mask, a.shape[0])
            # coalesced is not implemented for bool tensors, so need to cast
            mask = mask.to(dtype=a.dtype)  # type: ignore  # mypy is missing the catch above
        return torch.ops.xformers.matmul_with_mask(a, b, mask)

    # Non optimized codepath
    att = a @ b
    att[~mask] = float("-inf")
    return att


def _softmax(a: torch.Tensor, causal: bool = False) -> torch.Tensor:
    if _is_sparse_available and isinstance(a, SparseCS):
        return a.softmax()

    if a.is_sparse:
        return torch.sparse.softmax(a, dim=a.ndim - 1)

    if _use_triton:
        return triton_softmax(a, mask=None, causal=causal)
    else:
        return torch.softmax(a, dim=a.ndim - 1)


if _is_sparse_available:

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
        # pyre-fixme[16]: The `apply` method is not visible to Pyre.
        return SparseBMM.apply(a, b)


def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if _is_sparse_available:
        if isinstance(a, SparseCS):
            return a.spmm(b)
        if a.is_sparse:
            return _sparse_bmm(a, b)
    return a @ b


def _apply_dropout(att, dropout):
    if dropout is None:
        return att
    # Dropout chokes on sparse tensors
    if _is_sparse_available:
        if isinstance(att, SparseCS):
            values = att.values.clone()
            values = dropout(values)
            att = SparseCS.wrap(
                att.shape,
                values,
                att.row_indices,
                att.row_offsets,
                att.column_indices,
                att._transp_info,
            )
        elif att.is_sparse:
            att = att.coalesce()
            values = att.values().clone()  # protect against in-place dropout
            values = dropout(values)
            att = torch.sparse_coo_tensor(att.indices(), values, att.shape)

        return att

    # Non optimized vanilla dropout
    att = dropout(att)
    return att


def scaled_query_key_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask: Optional[torch.Tensor],
    causal: bool = False,
) -> torch.Tensor:
    # TODO assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
    # this is needed due to limitations in sparse_bmm for now

    # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
    q = q / math.sqrt(k.size(-1))
    att = _matmul_with_mask(q, k.transpose(-2, -1), att_mask)

    if att_mask is not None and att_mask.dtype != torch.bool:
        att = att + att_mask

    # Softmax to get the attention probabilities
    att = _softmax(att, causal=causal)
    return att


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[torch.Tensor],
    dropout: Optional[torch.nn.Module] = None,
    causal: bool = False,
) -> torch.Tensor:
    autocast_disabled = (
        _is_sparse_available
        and isinstance(att_mask, SparseCS)
        or (att_mask is not None and att_mask.is_sparse)
    )

    with torch.cuda.amp.autocast(enabled=False) if autocast_disabled else nullcontext():
        if autocast_disabled:
            q, k, v = q.float(), k.float(), v.float()

        att = scaled_query_key_softmax(q, k, att_mask=att_mask, causal=causal)

        #  Optional dropout, could be part of the masking in the future
        att = _apply_dropout(att, dropout)

        # Get to the predicted values, for all heads
        # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
        y = bmm(att, v)
    return y
