# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton
import triton.language as tl


@triton.jit
def index_select_cat_fwd_kernel(
    output_ptr,  # *Pointer* to output tensor.
    source_ptr,  # *Pointer* to source tensor.
    index_ptr,  # *Pointer* to index tensor.
    num_indices,
    num_cols,
    stride0,  # Stride information of source tensor.
    stride1,
    BLOCK_SIZE_INDEX: tl.constexpr,  # Number of indices each program should process.
    BLOCK_SIZE_COL: tl.constexpr,  # Number of cols each program should process.
):
    pid0 = tl.program_id(axis=0)  # We use 2D launch grid
    pid1 = tl.program_id(axis=1)

    indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    rows = tl.load(index_ptr + indices, mask=(indices < num_indices))
    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    source_offsets = source_ptr + rows[:, None] * stride0 + cols[None, :] * stride1
    mask = (indices[:, None] < num_indices) & (cols[None, :] < num_cols)
    output = tl.load(source_offsets, mask=mask)

    output_offsets = output_ptr + indices[:, None] * stride0 + cols[None, :] * stride1
    tl.store(output_offsets, output, mask=mask)


def index_select_cat_fwd(
    output: torch.Tensor,
    source: torch.Tensor,
    index: torch.Tensor,
):
    if not (source.is_cuda and index.is_cuda):
        raise ValueError("The index tensor and the source tensor must be of type CUDA!")

    if not source.ndim == 2:
        raise ValueError(f"Expected 2-dimensional tensor, got {source.ndim}.")
    if not index.ndim == 1:
        raise ValueError(f"Expected 1-dimensional tensor, got {index.ndim}.")

    num_rows, num_cols = source.shape
    num_indices = index.shape[0]

    if not num_indices < num_rows:
        raise ValueError(
            "The number of indices cannot exceed the number of rows in the source matrix."
        )

    stride0, stride1 = source.stride(0), source.stride(1)

    def grid(meta):
        return (
            triton.cdiv(num_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    index_select_cat_fwd_kernel[grid](
        output,
        source,
        index,
        num_indices,
        num_cols,
        stride0,
        stride1,
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_COL=512,
    )

    return output


@triton.jit
def index_select_cat_bwd_kernel(
    grad_source_ptr,  # *Pointer* to grad_source tensor.
    index_ptr,  # *Pointer* to index tensor.
    grad_output_ptr,  # *Pointer* to grad_output tensor.
    num_rows,
    num_indices,
    num_cols,
    stride0,  # Stride information of input and source tensor.
    stride1,
    BLOCK_SIZE_INDEX: tl.constexpr,  # Number of indices each program should process.
    BLOCK_SIZE_COL: tl.constexpr,  # Number of cols each program should process.
):
    pid0 = tl.program_id(axis=0)  # We use 3D launch grid
    pid1 = tl.program_id(axis=1)

    cols = pid1 * BLOCK_SIZE_COL + tl.arange(0, BLOCK_SIZE_COL)

    # load grad_output
    grad_output_indices = pid0 * BLOCK_SIZE_INDEX + tl.arange(0, BLOCK_SIZE_INDEX)
    grad_output_offsets = (
        grad_output_ptr
        + grad_output_indices[:, None] * stride0
        + cols[None, :] * stride1
    )
    grad_output_mask = (grad_output_indices[:, None] < num_indices) & (
        cols[None, :] < num_cols
    )
    grad_output = tl.load(grad_output_offsets, mask=grad_output_mask).to(tl.float32)

    # select indices from grad_source
    grad_source_indices = tl.load(
        index_ptr + grad_output_indices, mask=(grad_output_indices < num_indices)
    )
    grad_source_offsets = (
        grad_source_ptr
        + grad_source_indices[:, None] * stride0
        + cols[None, :] * stride1
    )

    # compute scaled index add and save
    tl.store(grad_source_offsets, grad_output, mask=grad_output_mask)


def index_select_cat_bwd(
    grad_source: torch.Tensor,
    index: torch.Tensor,
    grad_output: torch.Tensor,
):
    if not (grad_source.is_cuda and grad_output.is_cuda):
        raise ValueError("The grad_source and grad_output tensor must be of type CUDA!")

    if not (grad_source.ndim == 2 and grad_output.ndim == 2):
        raise ValueError(
            f"The grad_source and grad_output must be three-dimensional "
            f"(got {grad_source.ndim} and {grad_output.ndim})!"
        )
    if not grad_source.shape[1] == grad_output.shape[1]:
        raise ValueError(
            f"The number of elements along dimension 1 of grad_source and grad_output must be the same "
            f"(got {grad_source.shape[1]} and {grad_output.shape[1]})"
        )

    num_rows, num_cols = grad_source.shape
    num_indices, num_cols = grad_output.shape
    if not num_rows >= num_indices:
        raise ValueError(
            f"The number of elements along dimension 0 of grad_source must be larger than that of grad_output "
            f"(got {num_rows} and {num_indices})!"
        )
    if not index.shape[0] == num_indices:
        raise ValueError(
            f"The number of indices and the number of elements along dimension 0 of grad_output must match "
            f"(got {index.shape[0]} and {num_indices})!"
        )

    stride0, stride1 = grad_source.stride(0), grad_source.stride(1)
    if not (grad_output.stride(0) == stride0 and grad_output.stride(1) == stride1):
        raise ValueError(
            f"The strides of the grad_source and grad_output tensors must match "
            f"(got {stride0} vs. {grad_output.stride(0)}, {stride1} vs. {grad_output.stride(1)})!"
        )

    def grid(meta):
        return (
            triton.cdiv(num_indices, meta["BLOCK_SIZE_INDEX"]),
            triton.cdiv(num_cols, meta["BLOCK_SIZE_COL"]),
        )

    index_select_cat_bwd_kernel[grid](
        grad_source,
        index,
        grad_output,
        num_rows,
        num_indices,
        num_cols,
        grad_source.stride(0),
        grad_source.stride(1),
        BLOCK_SIZE_INDEX=1,
        BLOCK_SIZE_COL=512,
    )

    return
