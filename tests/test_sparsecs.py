import pytest
import torch

from xformers.components.attention import maybe_sparsify
from xformers.components.attention.core import SparseCS, _create_random_sparsity

B = 2
M = 16  # not a nice round number, on purpose


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_logical_and():
    device = torch.device("cuda")
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


@pytest.mark.parametrize("seq", [12, 32, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_dense_sparse(seq):
    # Check that we can .to_dense() without crashing
    device = torch.device("cuda")
    mask = torch.rand(seq, seq, device=device) > 0.1
    mask_cs = SparseCS(mask, device)
    mask_back_forth = SparseCS(mask_cs.to_dense(), device)

    assert torch.allclose(mask_cs.to_dense().long(), mask_back_forth.to_dense().long())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
def test_device(device):
    mask = _create_random_sparsity(
        torch.ones(B, M, M, dtype=torch.bool, device=device), 0.1
    )
    assert mask.device.type == device.type

    sparse_mask = maybe_sparsify(mask)
    assert sparse_mask.device.type == device.type
