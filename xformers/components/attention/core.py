# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from contextlib import nullcontext
from functools import lru_cache
from typing import Optional, Union

import torch

from xformers import _has_cpp_library, _is_triton_available
from xformers.components.attention.attention_mask import AttentionMask

if _has_cpp_library:
    from ._sputnik_sparse import SparseCS

_is_blocksparse_available = _is_triton_available()
if _is_blocksparse_available:
    from xformers.components.attention.blocksparse import BlockSparseAttention
    from xformers.triton.softmax import softmax as triton_softmax


logger = logging.getLogger("xformers")


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
    a: torch.Tensor,
    b: torch.Tensor,
    mask: Optional[Union[torch.Tensor, "SparseCS"]],
) -> torch.Tensor:
    if mask is None:
        return a @ b

    if _has_cpp_library and mask.dtype == torch.bool:
        if isinstance(mask, SparseCS):
            return mask.matmul_with_mask(a, b)
        if mask.is_sparse:
            # perform broadcasting if needed
            mask = _broadcast_batch(mask, a.shape[0])

            # coalesced is not implemented for bool tensors, so need to cast
            mask = mask.to(dtype=a.dtype)  # type: ignore  # mypy is missing the catch above

        return torch.ops.xformers.matmul_with_mask(a, b, mask)

    # Non optimized codepath
    if _has_cpp_library:
        assert not isinstance(mask, SparseCS)

    att = a @ b
    if mask.dtype == torch.bool:
        assert not isinstance(mask, SparseCS)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        # repeat if batch sizes don't match
        if (
            not isinstance(mask, SparseCS)
            and mask.ndim == 3
            and mask.shape[0] != att.shape[0]
            and (att.shape[0] % mask.shape[0]) == 0
        ):
            repeat_factor = att.shape[0] // mask.shape[0]
            mask = mask.repeat([repeat_factor, 1, 1])
            logger.info("Mismatched batch dimensions for mask, repeating mask.")
        att += mask
    return att


def _softmax(a: torch.Tensor, causal: bool = False) -> torch.Tensor:
    if _has_cpp_library and isinstance(a, SparseCS):
        return a.softmax()

    if a.is_sparse:
        return torch.sparse.softmax(a, dim=a.ndim - 1)

    if _is_triton_available():
        return triton_softmax(a, mask=None, causal=causal)
    else:
        return torch.softmax(a, dim=a.ndim - 1)


if _has_cpp_library:

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
    if _has_cpp_library:
        if isinstance(a, SparseCS):
            return a.spmm(b)
        if a.is_sparse:
            return _sparse_bmm(a, b)
    return a @ b


def _apply_dropout(att, dropout):
    if dropout is None:
        return att

    # Dropout chokes on sparse tensors
    if _has_cpp_library:
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
        else:
            # Simple dense case
            att = dropout(att)

        return att

    # Non optimized vanilla dropout
    att = dropout(att)
    return att


def scaled_query_key_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask: Optional[Union[AttentionMask, "SparseCS", torch.Tensor]],
) -> torch.Tensor:
    # TODO assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
    # this is needed due to limitations in sparse_bmm for now

    # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
    q = q / math.sqrt(k.size(-1))

    # Matmul with mask
    if att_mask is not None and isinstance(att_mask, AttentionMask):
        # Additive mask
        mask: Optional[Union[SparseCS, torch.Tensor]] = att_mask.values
    else:
        mask = att_mask

    att = _matmul_with_mask(q, k.transpose(-2, -1), mask)

    # Softmax to get the attention probabilities
    is_causal = isinstance(att_mask, AttentionMask) and att_mask.is_causal
    att = _softmax(att, causal=is_causal)
    return att


if _is_blocksparse_available:
    # 128 is default maxsize
    @lru_cache(maxsize=128)
    def _retrieve_blocksparse(
        num_heads: int, seq_len: int, block_size: int
    ) -> BlockSparseAttention:
        # Checks if blocksparse object exists in cache

        blocks = seq_len // block_size
        layout_fill = torch.ones((num_heads, blocks, blocks), dtype=torch.long)
        return BlockSparseAttention(
            layout=layout_fill, block_size=block_size, causal=True
        )


def blocksparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout: Optional[torch.nn.Module] = None,
    block_size: int = 128,
) -> torch.Tensor:

    orig_dim = q.dim()
    seq_len = q.shape[-2]
    # Layout head dimension: 1 or batch size (q.shape[0])
    layout_heads = 1

    # TODO perhaps add functionality to pad qkv if sequence length is not divisible by block size?
    assert seq_len % block_size == 0, "Sequence length must be divisible by block size"

    if orig_dim == 3:
        # Reshape from (N, S, hs) to (B, nh, S, hs) where N = B x nh, hs = D / nh
        # Assuming num_heads = 1, (N, S, hs) to (B, 1, S, hs)
        if layout_heads == 1:
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        else:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

    blocksparse_attention = _retrieve_blocksparse(layout_heads, seq_len, block_size)
    # Dropout is a no-op in evaluation mode
    if isinstance(dropout, torch.nn.Dropout):
        blocksparse_attention.attn_drop = dropout
    else:
        blocksparse_attention.attn_drop = torch.nn.Dropout(0.0)
    att = blocksparse_attention(q, k, v)

    # Reshape attention (B, nh, S, hs) back to (N, S, hs)
    if orig_dim == 3:
        return att.flatten(0, 1)
    return att


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[Union[AttentionMask, "SparseCS", torch.Tensor]],
    dropout: Optional[torch.nn.Module] = None,
    block_size: int = 128,
) -> torch.Tensor:
    autocast_disabled = (
        _has_cpp_library
        and isinstance(att_mask, SparseCS)
        or (att_mask is not None and att_mask.is_sparse)
    )
    seq_len = q.shape[-2]

    # switch if:
    #   causal is required but mask is not sparse
    #   fp16 or under amp context
    #   sequence length is divisible by block size
    #   same seq len for K and Q
    switch_to_blocksparse = (
        _is_blocksparse_available
        and (att_mask is not None and not att_mask.is_sparse)
        and (isinstance(att_mask, AttentionMask) and att_mask.is_causal)
        and (q.dtype == torch.float16 or torch.is_autocast_enabled())
        and not seq_len % block_size
        and q.shape[-2] == k.shape[-2]
    )

    if switch_to_blocksparse:
        logger.info("Switching causal attention to Triton blocksparse...")
        return blocksparse_attention(q, k, v, dropout, block_size)

    with torch.cuda.amp.autocast(enabled=False) if autocast_disabled else nullcontext():  # type: ignore
        if autocast_disabled:
            q, k, v = q.float(), k.float(), v.float()

        att = scaled_query_key_softmax(q, k, att_mask=att_mask)

        #  Optional dropout, could be part of the masking in the future
        att = _apply_dropout(att, dropout)

        # Get to the predicted values, for all heads
        # y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
        y = bmm(att, v)
    return y
