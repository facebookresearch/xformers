import torch

from xformers.components.attention.core import _sparse_bmm, scaled_dot_product_attention


def test_core_attention():
    b, s, d = 8, 900, 32
    prob = 0.95

    a = torch.rand(b, s, d)
    m = torch.rand(b, s, s) > prob
    m = m.to_sparse()

    # Check that the sparse and dense computations are equivalent
    r_sparse = scaled_dot_product_attention(a, a, a, m)
    r_dense = scaled_dot_product_attention(a, a, a, m.to_dense())

    assert torch.allclose(r_sparse, r_dense)


def _baseline_sparse_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # need to use torch.sparse.mm to get gradients wrt sparse matrix a
    # TODO implement this in C++ / CUDA as this is slow!
    out = []
    for ai, bi in zip(a, b):
        out.append(torch.sparse.mm(ai, bi))
    return torch.stack(out, dim=0)


def test_sparse_bmm():
    B, M, N = 8, 64, 32
    prob = 0.95
    a = torch.rand(B, M, N)
    a[a < prob] = 0
    a = a.to_sparse()
    b = torch.rand(B, N, M)

    res = _sparse_bmm(a, b)
    res_gt = _baseline_sparse_bmm(a, b)

    assert torch.allclose(res, res_gt)
