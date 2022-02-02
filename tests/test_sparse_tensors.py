# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# needed to register custom ops
import xformers  # noqa: F401
from xformers.sparse import BlockSparseTensor

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
_devices = ["cuda:0"]


def _create_tensor(device):
    BLOCK = 32
    Z = 8
    H = 2
    shape = (512, 512)

    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    values = torch.randn(Z, layout.sum(), BLOCK, BLOCK, device=device)

    mask = (
        layout[None, :, :, None, :, None]
        .repeat(Z, 1, 1, BLOCK, 1, BLOCK)
        .reshape(Z, H, shape[0], shape[1])
    )

    return BlockSparseTensor(values, layout), mask.bool()


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
