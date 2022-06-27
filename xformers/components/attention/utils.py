# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional

import torch


# Reshapes key padding mask from (batch_size, src_len) -> (batch_size * num_heads 1, src_len)
def reshape_key_padding_mask(
    key_padding_mask: torch.Tensor, batched_dim: int
) -> torch.Tensor:
    assert key_padding_mask.ndim == 2
    batch_size, src_len = key_padding_mask.size()
    num_heads = batched_dim // batch_size
    return _reshape_key_padding_mask(key_padding_mask, batch_size, src_len, num_heads)


def _reshape_key_padding_mask(
    key_padding_mask: torch.Tensor, batch_size: int, src_len: int, num_heads: int
) -> torch.Tensor:
    assert key_padding_mask.shape == (batch_size, src_len)
    key_padding_mask = (
        key_padding_mask.view(batch_size, 1, 1, src_len)
        .expand(-1, num_heads, -1, -1)
        .reshape(batch_size * num_heads, 1, src_len)
    )
    return key_padding_mask


# Combine the attention mask and key padding mask into a single mask
# Taken from https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py
# Additive masking not yet supported
def maybe_merge_masks(
    att_mask: Optional[torch.Tensor],
    key_padding_mask: Optional[torch.Tensor],
    batch_size: int,
    src_len: int,
    num_heads: int,
    tgt_len: Optional[int] = None,
) -> Optional[torch.Tensor]:
    if tgt_len is None:
        tgt_len = src_len
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch_size, src_len)
        key_padding_mask = _reshape_key_padding_mask(
            key_padding_mask, batch_size, src_len, num_heads
        )
        if att_mask is None:
            # make sure dimensions of key padding mask are the same as those expected for att_mask
            att_mask = key_padding_mask.expand(-1, tgt_len, -1)
        # Assumption is that False means to mask.
        elif att_mask.dtype == torch.bool:
            att_mask = att_mask.logical_and(key_padding_mask)
        else:
            att_mask = att_mask.masked_fill(~key_padding_mask, float("-inf"))

    return att_mask


# Assumes that matrix passed in has had softmax applied to it.
def iterative_pinv(softmax_mat: torch.Tensor, n_iter=6, pinverse_original_init=False):
    """
    Computing the Moore-Penrose inverse.
    Use an iterative method from (Razavi et al. 2014) to approximate the Moore-Penrose inverse via efficient
    matrix-matrix multiplications.
    """

    i = torch.eye(
        softmax_mat.size(-1), device=softmax_mat.device, dtype=softmax_mat.dtype
    )
    k = softmax_mat

    # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
    if pinverse_original_init:
        # This original implementation is more conservative to compute coefficient of Z_0.
        v = 1 / torch.max(torch.sum(k, dim=-2)) * k.transpose(-1, -2)
    else:
        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster
        # convergence.
        v = (
            1
            / torch.max(torch.sum(k, dim=-2), dim=-1).values[:, None, None]
            * k.transpose(-1, -2)
        )

    for _ in range(n_iter):
        kv = torch.matmul(k, v)
        v = torch.matmul(
            0.25 * v,
            13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)),
        )
    return v


def bool_mask_to_additive(
    mask: torch.Tensor, dtype: Optional[torch.dtype] = torch.float32
) -> torch.Tensor:
    assert (
        mask.dtype == torch.bool
    ), "This util is meant to convert in between bool masks and additive ones"

    mask_ = torch.zeros_like(mask, dtype=dtype)
    mask_[~mask] = float("-inf")
    return mask_
