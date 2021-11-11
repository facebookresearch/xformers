# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import triton

from xformers.triton.k_layer_norm import _LayerNorm

_triton_registered_warnings = False


class FusedLayerNorm(nn.Module):
    """
    Handle a layer normalization, like torch.nn.LayerNorm_.

    This implementation should be measurably faster than the default PyTorch layernorm (as of PyTorch 1.9),
    both for training and inference worloads.

    .. NOTE: Computations under Torch AMP are kept as float32 by default, one can change this to be float16
        by setting the flag `xformers.triton.k_layer_norm._triton_layernorm_fp16_enabled = True`

    .. _torch.nn.LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    """

    def __init__(self, normalized_shape, affine=True, eps=1e-05):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = self.bias = None
        self.epsilon = eps

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.epsilon)


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
) -> torch.Tensor:

    global _triton_registered_warnings

    r"""Applies normalization over a mini batch of inputs"""

    try:
        if (
            not _triton_registered_warnings
            and torch.cuda.is_available()
            and x.is_cuda
            and weight is not None
            and bias is not None
        ):
            return _LayerNorm.apply(x, weight, bias, eps)
    except (triton.code_gen.OutOfResources, RuntimeError) as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_warnings = True
        logging.warning(
            "Triton layernorm kernel register spillover or invalid image caught. "
            "Deactivating this kernel, please file an issue int the xFormers repository"
        )
        logging.warning(e)

    return torch.nn.functional.layer_norm(
        x, [x.shape[-1]], weight=weight, bias=bias, eps=eps
    )
