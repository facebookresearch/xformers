# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import List

import numpy as np
import torch

from xformers.components.attention.sparsity_config import (
    BigBirdSparsityConfig,
    BSLongformerSparsityConfig,
    FixedSparsityConfig,
    VariableSparsityConfig,
)


# generic nd cases
def _generate_nd_grid(*sizes):
    coords = [torch.arange(s) for s in sizes]
    return torch.meshgrid(*coords)


def local_nd_distance(*sizes, p=2.0, weights=None):
    if weights is None:
        weights = (1,) * len(sizes)
    assert len(sizes) == len(weights)
    grid = _generate_nd_grid(*sizes)
    grid = [i.flatten() * w for i, w in zip(grid, weights)]
    grid = torch.stack(grid, dim=1).float()
    d = torch.cdist(grid, grid, p=p)
    return d


def local_nd_gaussian_distribution(*sizes, sigma=1):
    d = local_nd_distance(*sizes, p=2.0) ** 2
    d = torch.exp(-0.5 * sigma ** (-2.0) * d)
    return d


def local_nd_pattern(*sizes, distance, p=2.0):
    d = local_nd_distance(*sizes, p=p)
    return d < distance


def axial_nd_pattern(*sizes):
    # axial is a special case with p=0 and distance=2
    d = local_nd_distance(*sizes, p=0)
    return d < 2


def random_pattern_from_probability_matrix(dist_matrix, nnz):
    att = torch.zeros_like(dist_matrix, dtype=torch.bool)
    # PyTorch multinomial wrongly doesn't support sampling when number of categories
    # is > 2^24, arguing that it's because it's the max representable consecutive element
    # in fp32 and that the kernels use float32. This is actually not true, and the kernels
    # should work fine if double tensor is passed on CPU. This is a bug that was introduced
    # in https://github.com/pytorch/pytorch/commit/bf04c2ca2f591d98ce57816f0ef0cd20a21bbf66
    # when unifying the checks between CPU and CUDA. For now, just fall-back to numpy
    if dist_matrix.numel() > 2**24:
        dist_matrix = dist_matrix.double()
        dist_matrix /= dist_matrix.sum()
        idxs = np.random.choice(
            dist_matrix.numel(), nnz, p=dist_matrix.flatten(), replace=False
        )
        idxs = torch.as_tensor(idxs)
    else:
        idxs = torch.multinomial(dist_matrix.flatten(), nnz, replacement=False)
    att.view(-1)[idxs] = True
    return att


def global_token_pattern(attention_query_mask: torch.Tensor) -> torch.Tensor:
    assert attention_query_mask.ndim == 1
    assert attention_query_mask.dtype == torch.bool
    attention_query_mask = attention_query_mask[None, :]
    mask = attention_query_mask | attention_query_mask.transpose(1, 0)
    return mask


def random_pattern(attn_size: int, sparsity: float) -> torch.Tensor:
    assert 0 < sparsity < 1
    mask = torch.rand(attn_size, attn_size) > sparsity
    return mask


# 1d-specific cases
def local_1d_pattern(attn_size: int, window_size: int) -> torch.Tensor:
    assert (
        window_size % 2 == 1
    ), "The window size is assumed to be odd (counts self-attention + 2 wings)"
    h_win_size = window_size // 2 + 1
    return local_nd_pattern(attn_size, distance=h_win_size, p=1.0)


def causal_1d_pattern(attn_size: int) -> torch.Tensor:
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.bool))
    return mask


# 2d-specific cases
def horizontal_axial_2d_distance(H, W, p=2.0):
    d = local_nd_distance(H, W, p=p, weights=(1, 0))
    return d


def vertical_axial_2d_distance(H, W, p=2.0):
    d = local_nd_distance(H, W, p=p, weights=(0, 1))
    return d


def local_2d_distance(H, W, p=2.0):
    return local_nd_distance(H, W, p=p)


def local_2d_gausian_distribution(H, W, sigma=1):
    return local_nd_gaussian_distribution(H, W, sigma=sigma)


def local_2d_pattern(H, W, distance, p=2.0):
    return local_nd_pattern(H, W, distance=distance, p=p)


def axial_2d_pattern(H, W):
    return axial_nd_pattern(H, W)


def swin_attention_pattern(H, W, window_size, shift_size=0):
    assert H % window_size == 0
    assert W % window_size == 0
    assert 0 <= shift_size < window_size, "shift_size must in 0-window_size"

    # input grid
    i, j = _generate_nd_grid(H, W)
    i, j = i + 0.5, j + 0.5

    # anchors grid
    # if shift is present, add extra element to the grid
    # to account for the uneven partitioning
    extra = int(shift_size % window_size != 0)
    grid_h = H // window_size + extra
    grid_w = W // window_size + extra

    ii, jj = _generate_nd_grid(grid_h, grid_w)
    # convert shift to be compatible with the paper representation
    s = (-shift_size) % window_size
    offset = window_size / 2 - s
    ii = ii * window_size + offset
    jj = jj * window_size + offset

    input_coords = torch.stack([i.flatten(), j.flatten()], 1).float()
    anchors_coords = torch.stack([ii.flatten(), jj.flatten()], 1).float()

    anchor_id = torch.cdist(input_coords, anchors_coords, p=2).argmin(1)
    mask = anchor_id[:, None] == anchor_id[None, :]
    return mask


def dilated_2d_pattern(H, W, k=2):
    """
    Returns a 2d pattern that samples 1 every k elements in the attention mask.
    Can be seen as a form of downsampling, where every pixel attends to a downsampled
    version of the input.
    """
    d_h = local_nd_distance(H, W, p=1, weights=(1, 0))
    d_w = local_nd_distance(H, W, p=1, weights=(0, 1))
    d = (d_h.floor() % k == 0) & (d_w.floor() % k == 0)
    return d


# Block sparse utils
def block_sparsify_tensor(x, mask, block_size):
    """
    Block sparsify a tensor, given a mask and block size
    """
    ret = torch.empty(
        (x.size(0), mask.sum(), block_size, block_size), dtype=x.dtype, device=x.device
    )

    for idx, (h, i, j) in enumerate(zip(*mask.nonzero(as_tuple=True))):
        ret[:, idx, :, :] = x[
            :,
            h,
            i * block_size : (i + 1) * block_size,
            j * block_size : (j + 1) * block_size,
        ]
    return ret


def pattern_to_layout(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    r"""
    Given a mask pattern and blocksize, return the corresponding layout
    which makes sure that all the positives in the mask are covered
    """
    assert mask.ndim >= 2, "We're expecting [Heads, Seq, Seq] or [Seq, Seq]"
    _should_squeeze = False

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        _should_squeeze = True

    assert (
        mask.shape[1] % block_size == 0 and mask.shape[2] % block_size == 0
    ), "We're only handling masks divisible by block_size"

    # Now mark the mask
    layout = torch.nn.functional.max_pool2d(
        mask.to(torch.float), kernel_size=block_size, stride=block_size
    )
    layout = layout.to(torch.long)

    if _should_squeeze:
        layout.squeeze_(0)

    return layout


def alibi_pattern(threshold: float, mask_shape: torch.Size) -> torch.Tensor:
    r"""
    Use the additive bias computation from ALiBi_ to generate a mask.
    Note that this mask can in turn be used to generate a blocksparse attention computation layout

    .. note: mask_shape is expected to hold the [heads, seq, seq] dimensions

    .. _ALiBi: https://arxiv.org/pdf/2108.12409.pdf
    """

    # CREDITS: code snippet from Ofir Press, one of the authors

    def get_slopes(n: int):
        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some a. This function has
        # some good properties that only occur when the input is a power of 2. To maintain that even
        # when the number of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = mask_shape[1]
    attn_heads = mask_shape[0]
    slopes = torch.Tensor(get_slopes(attn_heads))

    # In the next line, the part after the * is what constructs the diagonal matrix
    # (right matrix in Figure 3 in the paper).
    # If you run it you'll see that it doesn't exactly print out the same matrix as we have in Figure 3,
    # but one where all rows are identical.
    # This works because the softmax operation is invariant to translation,
    # and our bias functions are always linear.
    alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(maxpos).unsqueeze(
        0
    ).unsqueeze(0).expand(attn_heads, -1, -1)
    alibi = alibi.view(attn_heads, 1, maxpos)

    # Now threshold arbitrarily, report the mask
    return alibi < threshold


def quick_fixed_layout(num_heads: int, block_size: int, seq_len: int):
    config = FixedSparsityConfig(num_heads=num_heads, block_size=block_size)
    return config.make_layout(seq_len)


def quick_variable_layout(num_heads: int, block_size: int, seq_len: int):
    config = VariableSparsityConfig(num_heads=num_heads, block_size=block_size)
    return config.make_layout(seq_len)


def quick_bigbird_layout(num_heads: int, block_size: int, seq_len: int):
    config = BigBirdSparsityConfig(num_heads=num_heads, block_size=block_size)
    return config.make_layout(seq_len)


def quick_bslongformer_layout(num_heads: int, block_size: int, seq_len: int):
    config = BSLongformerSparsityConfig(num_heads=num_heads, block_size=block_size)
    return config.make_layout(seq_len)


def layout_to_pattern(layout: torch.Tensor, block_size: int):
    r"""
    create a pattern of shape [heads, seq, seq] out of a blocksparse
    layout of shape [heads, seq/block_size, seq/block_size]
    """
    return torch.kron(layout, torch.ones(block_size, block_size))
