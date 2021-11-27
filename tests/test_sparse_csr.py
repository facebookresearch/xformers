# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# needed to register custom ops
import xformers  # noqa: F401
import xformers.components.attention
from xformers.sparse import SparseCSRTensor

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
@pytest.mark.parametrize("func", [torch.add, torch.mul])
def test_sparse_binary_ops(func, device):
    B, L = 8, 30
    prob = 0.5
    a = _create_random_sparsity(torch.rand(B, L, L, device=device), prob)

    a_csr = SparseCSRTensor.from_dense(a)

    b = 5

    res = func(a_csr, b).to_dense()
    res_gt = func(a, b)
    res_gt[a == 0] = 0

    assert torch.allclose(res, res_gt)


@pytest.mark.parametrize("device", _devices)
def test_sparse_softmax(device):
    B, L = 8, 30
    prob = 0.5
    a = _create_random_sparsity(torch.rand(B, L, L, device=device), prob)

    a_csr = SparseCSRTensor.from_dense(a)

    fn = xformers.components.attention.core._softmax

    def fn2(x):
        return torch.nn.functional.softmax(x, -1)

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

    def fn2(x):
        return torch.nn.functional.softmax(x, -1)

    a = a.to_sparse()

    a_csr.requires_grad_(True)
    fn2(a_csr)._csr_values.sum().backward()
    grad_a = a_csr._csr_values.grad.clone()
    a.requires_grad_(True)
    fn(a).coalesce().values().sum().backward()
    assert torch.allclose(
        grad_a, a.grad.coalesce().values().reshape_as(grad_a), atol=1e-7
    )


@pytest.mark.parametrize("device", _devices)
def test_deepcopy(device):
    import copy

    B, L = 8, 30
    prob = 0.3
    a = _create_random_sparsity(torch.rand(B, L, L), prob)
    a_csr = SparseCSRTensor.from_dense(a)

    b_csr = copy.deepcopy(a_csr)
    assert torch.equal(a_csr, b_csr)


@pytest.mark.parametrize("device", _devices)
def test_module_buffer(device):
    B, L = 8, 30
    prob = 0.3
    a = _create_random_sparsity(torch.rand(B, L, L), prob)
    a_csr = SparseCSRTensor.from_dense(a)

    prob = 0.5
    b = _create_random_sparsity(torch.rand(B, L, L), prob)
    b_csr = SparseCSRTensor.from_dense(a)

    module = torch.nn.Module()
    # test that register_buffer works
    module.register_buffer("a_csr", a_csr)

    assert module.a_csr is a_csr

    module.to(device)
    assert module.a_csr.device == torch.device(device)

    state_dict = module.state_dict()
    assert "a_csr" in state_dict
    assert torch.equal(a_csr, state_dict["a_csr"])

    module.load_state_dict(state_dict)

    module.load_state_dict({"a_csr": b_csr})
    assert torch.equal(module.a_csr, b_csr)
