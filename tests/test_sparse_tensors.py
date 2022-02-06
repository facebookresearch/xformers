# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# needed to register custom ops
import xformers  # noqa: F401
from xformers.ops import masked_matmul
from xformers.sparse import BlockSparseTensor, SparseCSRTensor

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
_tensor_types = [BlockSparseTensor, SparseCSRTensor]


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


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("func", [torch.add, torch.mul])
def test_sparse_binary_ops(func, device):
    # TODO: add for BlockSparseTensor as well
    N, H, W = 8, 64, 64
    sparsity = 0.5
    shape = (N, H, W)

    a_sparse = _create_tensor(
        SparseCSRTensor, device, dtype=torch.float32, shape=shape, sparsity=sparsity
    )
    a = a_sparse.to_dense()

    b = a
    b_sparse = a_sparse

    res = func(a_sparse, b_sparse).to_dense()
    res_gt = func(a, b)

    assert torch.allclose(res, res_gt)


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_masked_matmul(tensor_type, device):
    N, C, H, W, L = 8, 2, 64, 64, 32
    sparsity = 0.7

    shape0 = (N, C, H, W)
    shape1 = (N, C, H, L)
    shape2 = (N, C, W, L)

    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]
        shape1 = shape1[1:]
        shape2 = shape2[1:]

    mask_sparse = _create_tensor(
        tensor_type, device, dtype=torch.bool, shape=shape0, sparsity=sparsity
    )
    mask = mask_sparse.to_dense()

    a = torch.randn(shape1, device=device)
    b = torch.randn(shape2, device=device)

    aa = a.clone()
    bb = b.clone()

    a.requires_grad_(True)
    b.requires_grad_(True)
    aa.requires_grad_(True)
    bb.requires_grad_(True)

    bt = b.transpose(-2, -1)
    bbt = bb.transpose(-2, -1)

    res_gt = masked_matmul(a, bt, mask)
    res = masked_matmul(aa, bbt, mask_sparse)

    res_dense = res.to_dense()
    res_dense = torch.where(mask, res_dense, torch.full_like(res_dense, float("-inf")))

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res_dense, res_gt, atol=5e-6)

    # try to workaround non-contiguous issues with triton for now
    res_gt.backward(torch.ones_like(res_gt))
    res.values().backward(torch.ones_like(res.values()))
    # TODO: this is not passing for BlockSparse!!!
    if tensor_type != BlockSparseTensor:
        assert torch.allclose(a.grad, aa.grad, atol=5e-6)
        assert torch.allclose(b.grad, bb.grad, atol=5e-6)


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_bmm(tensor_type, device):
    N, C, H, W, L = 8, 2, 64, 64, 32
    dtype = torch.float32
    sparsity = 0.8

    shape0 = (N, C, H, W)
    shape1 = (N, C, W, L)

    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]
        shape1 = shape1[1:]

    a_sparse = _create_tensor(
        tensor_type, device, dtype=dtype, shape=shape0, sparsity=sparsity
    )
    a = a_sparse.to_dense()
    mask = a != 0

    a_sparse.requires_grad_(True)
    a.requires_grad_(True)

    b = torch.randn(shape1, device=device)
    b2 = b.clone()

    b.requires_grad_(True)
    b2.requires_grad_(True)

    res_gt = a @ b
    res = a_sparse @ b2

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res, res_gt, atol=1e-5)

    res_gt.sum().backward()
    res.sum().backward()

    a_grad = a.grad.clone().detach()
    a_grad[~mask] = 0

    assert torch.allclose(b.grad, b2.grad, atol=1e-5)
    assert torch.allclose(a_grad, a_sparse.grad.to_dense(), atol=1e-5)


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_sparse_softmax(tensor_type, device):
    N, C, H, W = 8, 2, 64, 64
    dtype = torch.float32
    sparsity = 0.8

    shape0 = (N, C, H, W)
    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]

    a_sparse = _create_tensor(
        tensor_type, device, dtype=dtype, shape=shape0, sparsity=sparsity
    )
    a = a_sparse.to_dense()
    mask = a != 0

    a[~mask] = float("-inf")

    a_sparse.requires_grad_(True)
    a.requires_grad_(True)

    res_gt = torch.softmax(a, dim=-1)
    res_sparse = torch.softmax(a_sparse, dim=-1)

    res = res_sparse.to_dense()

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res, res_gt)

    # WARNING: gradients are modified in-place!
    res_sparse.values().backward(torch.ones_like(res_sparse.values()))
    res_gt.backward(torch.ones_like(res_gt))

    assert torch.allclose(a.grad, a_sparse.grad.to_dense(), atol=2e-7)


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_deepcopy(tensor_type, device):
    import copy

    N, C, H, W = 8, 2, 64, 64
    dtype = torch.float32
    sparsity = 0.8

    shape0 = (N, C, H, W)
    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]

    a_sparse = _create_tensor(
        tensor_type, device, dtype=dtype, shape=shape0, sparsity=sparsity
    )

    b_sparse = copy.deepcopy(a_sparse)
    assert torch.equal(a_sparse, b_sparse)


@pytest.mark.parametrize("tensor_type", _tensor_types)
@pytest.mark.parametrize("device", _devices)
def test_module_buffer(tensor_type, device):
    N, C, H, W = 8, 2, 64, 64
    dtype = torch.float32
    sparsity = 0.8

    shape0 = (N, C, H, W)
    if tensor_type != BlockSparseTensor:
        shape0 = shape0[1:]

    a_sparse = _create_tensor(
        tensor_type, device, dtype=dtype, shape=shape0, sparsity=sparsity
    )
    b_sparse = _create_tensor(
        tensor_type, device, dtype=dtype, shape=shape0, sparsity=sparsity
    )

    module = torch.nn.Module()
    # test that register_buffer works
    module.register_buffer("a_sparse", a_sparse)

    assert module.a_sparse is a_sparse

    module.to(device)
    assert module.a_sparse.device == torch.device(device)

    state_dict = module.state_dict()
    assert "a_sparse" in state_dict
    assert torch.equal(a_sparse.to(device), state_dict["a_sparse"])

    module.load_state_dict(state_dict)

    module.load_state_dict({"a_sparse": b_sparse})
    assert torch.equal(module.a_sparse, b_sparse.to(device))
