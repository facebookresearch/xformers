# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
from torch import nn

from .. import _is_triton_available


def rms_norm(x, weight: Optional[torch.Tensor], eps: float = 1e-6):
    """
    RMS Normalization along the last dimension.

    This is similar to torch.nn.functional.normalize but with eps being added
    instead of max.

    Expects x contiguous of shape (..., dim), and returns normalized data
    of the same shape. For each dim-length vector x, the result has

        x / sqrt( x*x.sum() + eps)

    If weights are included, they are a contiguous parameter of length dim
    which multiplies the result.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """
    assert _is_triton_available()
    from ._triton.rmsnorm_kernels import _rms_norm_forward

    if torch.is_grad_enabled() and (
        x.requires_grad or (weight is not None and weight.requires_grad)
    ):
        raise ValueError("Gradients not supported.")

    return _rms_norm_forward(x, weight, eps)


def rms_norm_add(
    x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor], eps: float = 1e-6
):
    """
    An addition fused with rms_norm.

        z = rms_norm_add(x, y, weight, eps)

    is equivalent to

        x += y
        z = rms_norm(x, weight, eps)

    where x, y and z are all contiguous.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """
    if torch.is_grad_enabled() and (
        x.requires_grad
        or y.requires_grad
        or (weight is not None and weight.requires_grad)
    ):
        raise ValueError("Gradients not supported.")
    assert _is_triton_available()
    from ._triton.rmsnorm_kernels import _rms_norm_add_forward

    return _rms_norm_add_forward(x, y, weight, eps)


class RMSNorm(torch.nn.Module):
    """
    RMS Normalization layer along the last dimension.

    This is similar to torch.nn.functional.normalize but with eps being added
    instead of max.

    Expects contiguous input of shape (..., dim), and returns normalized data
    of the same shape. For each dim-length vector x, the result has

        x / sqrt( x*x.sum() + eps)

    If weights are included, they are a parameter of length dim which multiplies
    the result.

    This functionality is experimental. Its API might be changed without warnings.
    Use it at your own risk.
    """

    def __init__(self, dim: int, include_weight: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if include_weight:
            self.weight: Optional[nn.Parameter] = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x: torch.Tensor):
        return rms_norm(x, self.weight, self.eps)  # type: ignore

    def increment_and_forward_(self, x: torch.Tensor, y: torch.Tensor):
        """
        An addition fused with forward.

            z = layer.increment_and_forward_(x, y)

        is equivalent to

            x += y
            z = layer(x)
        """
        return rms_norm_add(x, y, self.weight, self.eps)  # type: ignore
