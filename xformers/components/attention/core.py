# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from contextlib import nullcontext
from typing import Optional, Union

import torch

from xformers.components.attention.attention_mask import AttentionMask
from xformers.ops import masked_matmul, softmax
from xformers.sparse import (
    BlockSparseTensor,
    CausalTensor,
    SparseCOOTensor,
    SparseCSRTensor,
)

MaskType = Union[
    torch.Tensor, BlockSparseTensor, SparseCSRTensor, SparseCOOTensor, CausalTensor
]


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


def scaled_query_key_softmax(
    q: torch.Tensor,
    k: torch.Tensor,
    att_mask: Optional[MaskType],
) -> torch.Tensor:
    # TODO assume we have (N, S, hs) instead of (B, nh, S, hs), with N = B x nh
    # this is needed due to limitations in sparse_bmm for now

    # Self-attend: (N, S, hs) x (N, hs, S) -> (N, S, S)
    q = q / math.sqrt(k.size(-1))

    # Matmul with mask
    if att_mask is not None and isinstance(att_mask, AttentionMask):
        # Additive mask
        mask: Optional[MaskType] = att_mask.values
    else:
        mask = att_mask

    att = masked_matmul(q, k.transpose(-2, -1), mask)

    att = softmax(att)
    return att


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    att_mask: Optional[MaskType],
    dropout: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    autocast_disabled = type(att_mask) in [SparseCSRTensor, SparseCOOTensor]

    with torch.cuda.amp.autocast(enabled=False) if autocast_disabled else nullcontext():
        if autocast_disabled:
            q, k, v = q.float(), k.float(), v.float()

        att = scaled_query_key_softmax(q, k, att_mask=att_mask)

        #  Optional dropout, could be part of the masking in the future
        if dropout is not None:
            att = dropout(att)

        # Get to the predicted values, for all heads
        y = att @ v  # (N, S, S) x (N, S, hs) -> (N, S, hs)
    return y
