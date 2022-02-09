# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.sparse import (
    BlockSparseTensor,
    CausalTensor,
    SparseCOOTensor,
    SparseCSRTensor,
)


def _create_blocksparse_tensor(
    device, block_size=32, Z=8, C=2, H=64, W=64, dtype=torch.float32
):
    layout = torch.randint(2, (C, H // block_size, W // block_size))
    layout[:, :, 0] = 1
    layout[:, 0, :] = 1
    values = torch.randn(Z, layout.sum(), block_size, block_size, device=device).to(
        dtype
    )

    return BlockSparseTensor(values, layout)


def _create_csr_tensor(device, dtype, shape, sparsity, divisible_by=4):
    matrix = torch.rand(shape, dtype=torch.float32, device=device).to(dtype)
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
    return SparseCSRTensor.from_dense(output)


def _create_coo_tensor(device, dtype, shape, sparsity):
    matrix = torch.rand(shape, dtype=torch.float32, device=device).to(dtype)
    zeros = torch.rand(shape, device=device) > sparsity
    matrix[zeros] = 0
    return SparseCOOTensor(matrix.to_sparse())


def _create_causal_tensor(device, dtype, shape):
    matrix = torch.rand(shape, dtype=torch.float32, device=device).to(dtype)
    return CausalTensor(torch.tril(matrix))


def _create_tensor(tensor_type, device, dtype, shape, sparsity):
    if tensor_type == BlockSparseTensor:
        block_size = 16
        return _create_blocksparse_tensor(
            device=device, dtype=dtype, block_size=block_size
        )
    elif tensor_type == SparseCSRTensor:
        return _create_csr_tensor(
            device=device, dtype=dtype, shape=shape, sparsity=sparsity
        )
    elif tensor_type == SparseCOOTensor:
        return _create_coo_tensor(
            device=device, dtype=dtype, shape=shape, sparsity=sparsity
        )
    elif tensor_type == CausalTensor:
        return _create_causal_tensor(device=device, dtype=dtype, shape=shape)
    elif tensor_type == torch.Tensor:
        matrix = torch.rand(shape, dtype=torch.float32, device=device).to(dtype)
        zeros = torch.rand(shape, device=device) > sparsity
        matrix[zeros] = 0
        return matrix
