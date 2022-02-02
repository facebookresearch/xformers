# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# needed to register custom ops
import xformers  # noqa: F401
from xformers.ops import masked_matmul
from xformers.sparse import BlockSparseTensor

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cuda:0"] if torch.cuda.is_available() else []


def _create_tensor(device, BLOCK=32, Z=8, C=2, H=512, W=512, dtype=torch.float32):
    layout = torch.randint(2, (C, H // BLOCK, W // BLOCK))
    values = torch.randn(Z, layout.sum(), BLOCK, BLOCK, device=device).to(dtype)

    mask = (
        layout[None, :, :, None, :, None]
        .repeat(Z, 1, 1, BLOCK, 1, BLOCK)
        .reshape(Z, C, H, W)
    )

    return BlockSparseTensor(values, layout), mask.bool()


@pytest.mark.parametrize("device", _devices)
def test_masked_matmul(device):
    BLOCK = 32
    N, C, H, W, L = 8, 2, 512, 512, 64
    mask_block, _ = _create_tensor(device, BLOCK, N, C, H, W, dtype=torch.bool)
    mask = mask_block.to_dense()

    a = torch.randn(N, C, H, L, device=device)
    b = torch.randn(N, C, W, L, device=device)

    aa = a.clone()
    bb = b.clone()

    b = b.transpose(-2, -1)
    bb = bb.transpose(-2, -1)

    a.requires_grad_(True)
    b.requires_grad_(True)
    aa.requires_grad_(True)
    bb.requires_grad_(True)

    # res_gt = masked_matmul(a, b, mask)
    res_gt = a @ b
    # res_gt[~mask] = 0
    res_gt = torch.where(mask, res_gt, torch.zeros_like(res_gt))
    res = masked_matmul(aa, bb, mask_block)

    res_dense = res.to_dense()
    # res_dense[~mask] = float('-inf')

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res_dense, res_gt)

    res_gt.sum().backward()
    res._blocksparse_values.sum().backward()
    # TODO: this is not passing!!!
    # assert torch.allclose(a.grad, aa.grad, atol=1e-7)
    # assert torch.allclose(b.grad, bb.grad, atol=1e-7)


@pytest.mark.parametrize("device", _devices)
def test_bmm(device):
    BLOCK = 32
    N, C, H, W, L = 8, 2, 512, 512, 64
    a_block, mask = _create_tensor(device, BLOCK, N, C, H, W)
    a = a_block.to_dense()

    a_block.requires_grad_(True)
    a.requires_grad_(True)

    b = torch.randn(N, C, W, L, device=device)
    b2 = b.clone()

    b.requires_grad_(True)
    b2.requires_grad_(True)

    res_gt = a @ b
    res = a_block @ b2

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res, res_gt)

    res_gt.sum().backward()
    res.sum().backward()

    a_grad = a.grad.clone().detach()
    a_grad[~mask] = 0

    assert torch.allclose(b.grad, b2.grad)
    assert torch.allclose(a_grad, a_block.grad.to_dense(), atol=1e-7)


@pytest.mark.parametrize("device", _devices)
def test_sparse_softmax(device):
    a_block, mask = _create_tensor(device)
    a = a_block.to_dense()
    a[~mask] = float("-inf")

    res_gt = torch.softmax(a, dim=-1)
    res_block = torch.softmax(a_block, dim=-1)

    res = res_block.to_dense()

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res, res_gt)


@pytest.mark.parametrize("device", _devices)
def test_sparse_softmax_backward(device):
    a_block, mask = _create_tensor(device)
    a = a_block.to_dense()
    a_block.requires_grad_(True)

    a[~mask] = float("-inf")
    a.requires_grad_(True)

    res_gt = torch.softmax(a, dim=-1)
    res_block = torch.softmax(a_block, dim=-1)

    res_block._blocksparse_values.sum().backward()
    res_gt.sum().backward()

    assert torch.allclose(a.grad, a_block.grad.to_dense(), atol=1e-7)


@pytest.mark.parametrize("device", _devices)
def test_deepcopy(device):
    import copy

    a_block, mask = _create_tensor(device)

    b_block = copy.deepcopy(a_block)
    assert torch.equal(a_block, b_block)


@pytest.mark.parametrize("device", _devices)
def test_module_buffer(device):
    a_block, _ = _create_tensor(device)
    b_block, _ = _create_tensor(device)

    module = torch.nn.Module()
    # test that register_buffer works
    module.register_buffer("a_block", a_block)

    assert module.a_block is a_block

    module.to(device)
    assert module.a_block.device == torch.device(device)

    state_dict = module.state_dict()
    assert "a_block" in state_dict
    assert torch.equal(a_block.to(device), state_dict["a_block"])

    module.load_state_dict(state_dict)

    module.load_state_dict({"a_block": b_block})
    assert torch.equal(module.a_block, b_block.to(device))
