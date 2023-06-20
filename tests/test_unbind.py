# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

import xformers.ops
from xformers.ops.common import _get_storage_base


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_unbind(dim: int, contiguous: bool):
    x = torch.randn([10, 20, 4, 10, 3])
    x2 = x.clone()

    if not contiguous:
        perm = list(range(x.ndim))
        random.Random(dim).shuffle(perm)
        # Let's hope we didn't pick identity
        x = x.permute(perm)
        x2 = x2.permute(perm)
    assert contiguous == x.is_contiguous()
    x.requires_grad_(True)
    x2.requires_grad_(True)

    # FW
    tensors = xformers.ops.unbind(x, dim)
    tensors2 = torch.unbind(x2, dim)
    assert len(tensors) == len(tensors2)
    for t1, t2 in zip(tensors, tensors2):
        assert torch.allclose(t1, t2)

    # BW
    grads = torch.unbind(torch.randn(x.shape), dim)
    zero = torch.zeros_like(tensors[0])
    loss1 = sum(((g * t) for (g, t) in zip(grads, tensors)), zero)
    loss2 = sum(((g * t) for (g, t) in zip(grads, tensors2)), zero)
    assert torch.allclose(loss1, loss2)
    g = torch.randn_like(loss1)
    loss1.backward(g)
    loss2.backward(g)
    assert x.grad is not None
    assert x2.grad is not None
    assert torch.allclose(x.grad, x2.grad)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
def test_unbind_get_stack_strides(dim: int, contiguous: bool):
    def not_stacked(t, d):
        return xformers.ops.get_stack_strides(t, d) is None

    x = torch.randn([10, 20, 4, 4, 3])
    ndim = x.ndim

    # Non-contiguous tensors
    if not contiguous:
        x = x.transpose(dim, (dim + 1) % ndim)
    assert contiguous == x.is_contiguous()

    tensors = xformers.ops.unbind(x, dim)
    tensors2 = torch.unbind(x.clone(), dim)

    for cat_dim in range(ndim):
        permute = list(range(ndim))
        permute.pop(dim)
        permute.insert(cat_dim, dim)
        x_permuted = x.permute(permute)
        assert not_stacked([tensors2[0], tensors[1]], cat_dim), "different storage"
        assert not_stacked(
            [tensors[0], tensors[1].clone()], cat_dim
        ), "different storage"

        def test_slice(s):
            slices = [slice(None) for _ in range(ndim)]
            slices[cat_dim] = s
            reference = x_permuted[tuple(slices)]
            stacked = xformers.ops.stack_or_none(tensors[s], cat_dim)
            assert stacked is not None
            assert (
                xformers.ops.get_stack_strides(tensors[s], cat_dim)
                == reference.stride()
            )
            assert torch.allclose(stacked, torch.stack(tensors2[s], cat_dim))
            assert _get_storage_base(stacked) == _get_storage_base(tensors[0])

        # tensors
        test_slice(slice(None))

        # tensors[1:]
        test_slice(slice(1, None))

        # tensors[:2]
        test_slice(slice(None, 2))

        # tensors[::2]
        test_slice(slice(None, None, 2))
