# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Sequence

import torch

from xformers.ops._triton import (
    index_select_cat_bwd,
    index_select_cat_fwd,
    scaled_index_add_bwd,
    scaled_index_add_fwd,
)

from .common import BaseOperator, register_operator


# Keeping these operator registry here so that
# it's easy to check if they are available
@register_operator
class ScaledIndexAddFw(BaseOperator):
    OPERATOR = scaled_index_add_fwd
    OPERATOR_CATEGORY = "indexing"
    NAME = "scaled_index_addF"


@register_operator
class ScaledIndexAddBw(BaseOperator):
    OPERATOR = scaled_index_add_bwd
    OPERATOR_CATEGORY = "indexing"
    NAME = "scaled_index_addB"


@register_operator
class IndexSelect(BaseOperator):
    OPERATOR = index_select_cat_fwd
    OPERATOR_CATEGORY = "indexing"
    NAME = "index_select"


class _ScaledIndexAdd(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(
        ctx,
        x: torch.Tensor,
        index: torch.Tensor,
        source: torch.Tensor,
        scaling: Optional[torch.Tensor],
        alpha: float,
    ) -> torch.Tensor:
        if scaled_index_add_fwd is not None:
            scaled_index_add_fwd(x, index, source, scaling, alpha)
        else:
            raise RuntimeError(
                "Triton is needed for forward pass but it is not available!"
            )

        ctx.mark_dirty(x)
        ctx.save_for_backward(index, scaling, source)
        ctx.source_shape = source.shape
        ctx.alpha = alpha
        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        index, scaling, source = ctx.saved_tensors
        grad_source = torch.empty_like(source)
        grad_scaling = (
            None
            if scaling is None
            else torch.empty(
                ctx.source_shape, dtype=scaling.dtype, device=scaling.device
            )
        )

        if scaled_index_add_bwd is not None:
            scaled_index_add_bwd(
                grad_output,
                grad_source,
                grad_scaling,
                source,
                scaling,
                index,
                ctx.alpha,
            )
        else:
            raise RuntimeError(
                "Triton is needed for backward pass but it is not available!"
            )

        return (
            grad_output,  # gradient of input
            None,  # gradient of index
            grad_source,  # gradient of source
            grad_scaling,  # gradient of scaling
            None,  # gradient of alpha
        )


def scaled_index_add(
    input: torch.Tensor,  # [B, M, D]
    index: torch.Tensor,  # [Bi] - int64
    source: torch.Tensor,  # [Bi, M, D]
    scaling: Optional[torch.Tensor] = None,  # [D]
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    In-place scaling+index_add

    Indices in ``index`` are assumed to be unique

    The max index in ``index`` is assumed to be less than the size of dim0 of ``input``.

    :Note:

        The FW pass is done in-place (``input`` is modified)

    :Equivalent pytorch code:

    .. code-block:: python

        return torch.index_add(input, dim=0, source=scaling * src, index=indices, alpha=alpha)
    """

    return _ScaledIndexAdd.apply(input, index, source, scaling, alpha)


class _IndexSelectCat(torch.autograd.Function):
    @staticmethod
    # type: ignore
    def forward(
        ctx,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        assert len(args) % 2 == 0
        sources = args[: len(args) // 2]
        indices = args[len(args) // 2 :]

        output_numel = 0
        for source, index in zip(sources, indices):
            num_rows, num_cols = source.shape
            num_indices = index.shape[0]
            output_numel += num_indices * num_cols

        output = torch.empty(
            [output_numel], dtype=sources[0].dtype, device=sources[0].device
        )

        processed_numel = 0
        for source, index in zip(sources, indices):
            num_indices = index.shape[0]
            num_cols = source.shape[1]

            if index_select_cat_fwd is not None:
                index_select_cat_fwd(
                    output[
                        processed_numel : processed_numel + num_indices * num_cols
                    ].view([num_indices, num_cols]),
                    source,
                    index,
                )
            else:
                raise RuntimeError(
                    "Triton is needed for forward pass but it is not available!"
                )

            processed_numel += num_indices * num_cols

        ctx.save_for_backward(*indices)
        ctx.source_shapes = [source.shape for source in sources]

        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        indices = ctx.saved_tensors

        gradients = []
        processed_numel = 0
        for source_shape, index in zip(ctx.source_shapes, indices):
            num_rows, num_cols = source_shape
            num_indices = index.shape[0]

            grad_output_slice = grad_output[
                processed_numel : processed_numel + num_indices * num_cols
            ].reshape([num_indices, num_cols])
            processed_numel += num_indices * num_cols

            grad_source_slice = torch.zeros(
                [num_rows, num_cols],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            if index_select_cat_bwd is not None:
                index_select_cat_bwd(
                    grad_source_slice,
                    index,
                    grad_output_slice,
                )
            else:
                raise RuntimeError(
                    "Triton is needed for backward pass but it is not available!"
                )
            gradients.append(grad_source_slice)

        return (*gradients, *([None] * len(gradients)))


def index_select_cat(
    sources: Sequence[torch.Tensor], indices: Sequence[torch.Tensor]
) -> torch.Tensor:
    """
    Indices in ``index`` are assumed to be unique
    In each (index, source) pair, the max index in ``index`` is assumed to be less than the size of dim0 of ``source``

    :Example:

    Given:
    - ``sources[0]`` of shape ``[S0, D0]``
    - ``indices[0]`` of shape ``[I0]``
    - ``sources[1]`` of shape ``[S1, D1]``
    - ``indices[1]`` of shape ``[I1]``
    returns a ``torch.Tensor`` of shape ``[I0 * D0 + I1 * D1]``

    :Equivalent pytorch code:

    .. code-block:: python

        return torch.cat([s[i.long()].flatten() for s, i in zip(sources, indices)], dim=0)
    """
    return _IndexSelectCat.apply(*sources, *indices)
