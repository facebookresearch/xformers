import typing

import torch
import torch.nn
import torch.nn.functional

from xformers.triton.dropout import dropout
from xformers.triton.fused_linear_layer import FusedLinear
from xformers.triton.layer_norm import FusedLayerNorm
from xformers.triton.softmax import log_softmax, softmax


def _softmax(input: torch.Tensor, dim: int = -1, dtype=torch.float32):
    if dim == -1:
        return softmax(input)
    return softmax(input.transpose(-1, dim)).transpose(-1, dim).to(dtype=dtype)


def _log_softmax(input: torch.Tensor, dim: int = -1, dtype=torch.float32):
    if dim == -1:
        return log_softmax(input)
    return log_softmax(input.transpose(-1, dim)).transpose(-1, dim).to(dtype=dtype)


def _dropout(input: torch.Tensor, p: float = 0.5, training: bool = True, inplace: bool = False):
    if training:
        return dropout(input, p)
    return input * p


def _layer_norm_wrapper(normalized_shape: typing.List[int], eps: float = 1e-5, elementwise_affine: bool = True,
                        device=None, dtype=None):
    return FusedLayerNorm(normalized_shape, elementwise_affine, eps).to(device=device, dtype=dtype)


def patch_torch():
    torch.nn.functional.softmax = _softmax
    torch.nn.functional.log_softmax = _log_softmax
    torch.nn.functional.dropout = dropout
    torch.nn.Linear = FusedLinear
    torch.nn.LayerNorm = FusedLayerNorm
