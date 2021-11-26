# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# needed to register custom ops
import xformers  # noqa: F401

from xformers.sparse.csr_tensor import SparseCSRTensor


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _create_random_sparsity(matrix, sparsity, divisible_by=4):
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
    return output


@pytest.mark.parametrize("device", _devices)
def test_sparse_softmax(device):
    B, L = 8, 30
    prob = 0.5
    a = _create_random_sparsity(torch.rand(B, L, L, device=device), prob)

    a_csr = SparseCSRTensor.from_dense(a)

    fn = xformers.components.attention.core._softmax
    fn2 = lambda x: torch.nn.functional.softmax(x, -1)

    a = a.to_sparse()

    res = fn2(a_csr)
    res_gt = fn(a)

    res = res.to_dense()
    res_gt = res_gt.to_dense()

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res, res_gt)


@pytest.mark.parametrize("device", _devices)
def test_sparse_softmax_backward(device):
    B, L = 8, 30
    prob = 0.5
    a = _create_random_sparsity(torch.rand(B, L, L, device=device), prob)

    a_csr = SparseCSRTensor.from_dense(a)

    fn = xformers.components.attention.core._softmax
    fn2 = lambda x: torch.nn.functional.softmax(x, -1)

    a = a.to_sparse()

    a_csr.requires_grad_(True)
    fn2(a_csr)._csr_values.sum().backward()
    grad_a = a_csr._csr_values.grad.clone()
    a.requires_grad_(True)
    fn(a).coalesce().values().sum().backward()
    assert torch.allclose(
        grad_a, a.grad.coalesce().values().reshape_as(grad_a), atol=1e-7
    )
