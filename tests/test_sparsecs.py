# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components.attention import maybe_sparsify
from xformers.components.attention._sputnik_sparse import _dense_to_sparse
from xformers.components.attention.core import _create_random_sparsity, SparseCS

B = 2
M = 16  # not a nice round number, on purpose

_devices_list = ["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]
_devices = [torch.device(d) for d in _devices_list]


@pytest.mark.parametrize("device", _devices)
def test_logical_and(device):
    mask = _create_random_sparsity(torch.ones(B, M, M, dtype=torch.bool), 0.1)
    mask_cs = SparseCS(mask, device)

    # Check that we cannot & two sparse matrices (for now)
    with pytest.raises(Exception):
        _ = mask_cs & mask_cs

    # Check that & ones returns the same values
    mask_ones = mask_cs & torch.ones_like(mask, dtype=torch.bool, device=device)
    assert torch.allclose(mask_cs.to_dense().long(), mask_ones.to_dense().long())

    # Check that & the inverse returns 0 all around
    mask_not = ~mask.to(device)
    assert (mask_cs & mask_not).values.numel() == 0


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("seq", [12, 32, 128])
def test_dense_sparse(seq, device):
    # Check that we can .to_dense() without crashing
    mask = torch.rand(seq, seq, device=device) > 0.1
    mask_cs = SparseCS(mask, device)

    mask_back_forth = SparseCS(mask_cs.to_dense(), device)

    assert torch.allclose(mask_cs.to_dense().long(), mask_back_forth.to_dense().long())


@pytest.mark.parametrize("device", _devices)
def test_device(device):
    mask = _create_random_sparsity(
        torch.ones(B, M, M, dtype=torch.bool, device=device), 0.1
    )
    assert mask.device.type == device.type

    sparse_mask = maybe_sparsify(mask)
    assert sparse_mask.device.type == device.type


def _baseline_dense_to_sparse(matrix):
    import numpy as np

    # Extract the nonzero values.
    values = matrix.compress((matrix != 0).flatten())

    # Calculate the offset of each row.
    mask = (matrix != 0).astype(np.int32)
    row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))), axis=0)

    # Create the row indices and sort them.
    # note: use torch.argsort to make it compatible as sorting is not stable in PyTorch
    row_indices = torch.argsort(-1 * torch.as_tensor(np.diff(row_offsets))).numpy()

    # Extract the column indices for the nonzero values.
    x = mask * (np.arange(matrix.shape[1]) + 1)
    column_indices = x.compress((x != 0).flatten())
    column_indices = column_indices - 1

    # Cast the desired precision.
    values = torch.as_tensor(values.astype(np.float32))
    row_indices, row_offsets, column_indices = [
        torch.as_tensor(x.astype(np.int32))
        for x in [row_indices, row_offsets, column_indices]
    ]
    return values, row_indices, row_offsets, column_indices


@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("seq", [12, 32, 128])
def test_dense_to_sparse(seq, device):
    matrix = torch.rand(seq, seq, device=device)
    matrix[matrix > 0.9] = 0
    baseline_res = _baseline_dense_to_sparse(matrix.cpu().numpy())
    res = _dense_to_sparse(matrix, device=device)

    _idx_to_name = ["values", "row_indices", "row_offsets", "column_indices"]

    for idx, (bi, i) in enumerate(zip(baseline_res, res)):
        if idx != 1:
            # row_indices is the result of an argsort, which is not stable
            # for same number of elements
            assert torch.allclose(bi.to(device), i), f"error in {_idx_to_name[idx]}"
        assert bi.dtype == i.dtype
        assert i.device == device
