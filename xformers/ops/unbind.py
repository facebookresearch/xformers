# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Sequence, Tuple, Union

import torch


def get_stack_strides(
    tensors: Sequence[torch.Tensor], dim: int
) -> Optional[Tuple[int, ...]]:
    """
    If the tensors are already stacked, returns the strides of the stacked
    tensors. Otherwise returns None.
    """
    if len(tensors) <= 1 or dim > tensors[0].ndim:
        return None

    final_stride = []
    for i in range(tensors[0].ndim + 1):
        if i == dim:
            final_stride.append(
                tensors[1].storage_offset() - tensors[0].storage_offset()
            )
            continue
        if i > dim:
            i -= 1
        final_stride.append(tensors[0].stride(i))

    for i, x in enumerate(tensors):
        # Sanity checks
        if x.shape != tensors[0].shape:
            return None
        # Actual storage check
        if x.storage().data_ptr() != tensors[0].storage().data_ptr():
            return None
        if x.stride() != tensors[0].stride():
            return None
        if x.storage_offset() != tensors[0].storage_offset() + i * final_stride[dim]:
            return None
    return tuple(final_stride)


def efficient_stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int
) -> torch.Tensor:
    strides = get_stack_strides(tensors, dim)
    if strides is not None:
        input_shape = list(tensors[0].shape)
        input_shape.insert(dim, len(tensors))
        return tensors[0].as_strided(input_shape, strides)
    return torch.stack(tensors, dim=dim)


class _Unbind(torch.autograd.Function):
    """
    Splits a packed `qkv` tensor into query, key and values.
    The magic happens in the backward. We want to `torch.stack` the tensors
    together, but we don't need to if the gradients have already the same storage
    (and that is something that our attention operators support)
    """

    @staticmethod
    # type: ignore
    def forward(ctx, x: torch.Tensor, dim: int):
        ctx.dim = dim
        return x.unbind(dim)

    @classmethod
    # type: ignore
    def backward(cls, ctx, *tensors: torch.Tensor):
        return efficient_stack(tensors, ctx.dim), None


def unbind(x: torch.Tensor, dim: int) -> Tuple[torch.Tensor, ...]:
    return _Unbind.apply(x, dim)
