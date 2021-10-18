# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

from typing import Optional

import torch
import torch.nn as nn

from xformers.triton.k_layer_norm import _LayerNorm


class FusedLayerNorm(nn.Module):
    """
    Handle a layer normalization, like torch.nn.LayerNorm_.

    This implementation should be measurably faster than the default PyTorch layernorm (as of PyTorch 1.9),
    both for training and inference worloads.

    .. NOTE: Computations under Torch AMP are kept as float32 by default, one can change this to be float16
    by setting the flag `xformers.triton.k_layer_norm._triton_layernorm_fp16_enabled = True`

    .. _torch.nn.LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, normalized_shape, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.epsilon = eps

    def forward(self, x):
        return _LayerNorm.apply(x, self.weight, self.bias, self.epsilon)


def layer_norm(
    x: torch.Tensor,
    normalized_shape,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:
    r"""Applies normalization over a mini batch of inputs"""

    if (
        torch.cuda.is_available()
        and x.is_cuda
        and weight is not None
        and bias is not None
    ):
        # pyre-ignore[16]: Pyre is unable to find the `apply` method.
        return _LayerNorm.apply(x, normalized_shape, weight, bias, eps)

    return torch.nn.functional.layer_norm(
        x, normalized_shape, weight=weight, bias=bias, eps=eps
    )
