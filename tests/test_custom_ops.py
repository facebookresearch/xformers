import pytest
import torch

# needed to register custom ops
import xformers  # noqa: F401
from xformers.components.attention.core import _sparse_bmm

_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def _baseline_matmul_with_sparse_mask(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    assert a.ndim == b.ndim
    assert mask.ndim == a.ndim
    assert a.shape[-1] == b.shape[-2]
    assert a.shape[-2] == mask.shape[-2], f"{a.shape}, {mask.shape}"
    assert b.shape[-1] == mask.shape[-1], f"{b.shape}, {mask.shape}"
    assert a.shape[:-2] == b.shape[:-2], f"{a.shape}, {b.shape}"
    assert a.shape[:-2] == mask.shape[:-2], f"{a.shape}, {mask.shape}"
    idxs = mask.indices().unbind()
    b = b.transpose(-2, -1)

    # compute matmul for elements within the mask
    val = (a[idxs[:-2] + (idxs[-2], slice(None))] * b[idxs[:-2] + (idxs[-1], slice(None))]).sum(-1)  # type: ignore

    out_shape = a.shape[:-1] + (b.shape[-2],)
    res = torch.sparse_coo_tensor(torch.stack(idxs), val, out_shape)
    return res


def _baseline_matmul_with_dense_mask(
    a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    res = a @ b
    res[~mask] = float("-inf")
    return res


def _baseline_sparse_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # need to use torch.sparse.mm to get gradients wrt sparse matrix a
    # TODO implement this in C++ / CUDA as this is slow!
    out = []
    for ai, bi in zip(a, b):
        out.append(torch.sparse.mm(ai, bi))
    return torch.stack(out, dim=0)


@pytest.mark.parametrize("is_sparse", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", _devices)
def test_matmul_with_mask(device, contiguous, is_sparse):
    B, L, K = 8, 30, 32
    prob = 0.5
    a = torch.rand(B, L, K, device=device)
    b = torch.rand(B, K, L, device=device)
    if not contiguous:
        a = a.transpose(-2, -1).contiguous().transpose(-2, -1)
        b = b.transpose(-2, -1).contiguous().transpose(-2, -1)
    mask = torch.rand(B, L, L, device=device) > prob

    fn = torch.ops.xformers.matmul_with_mask
    fn_gt = _baseline_matmul_with_dense_mask

    if is_sparse:
        mask = mask.to_sparse()
        fn_gt = _baseline_matmul_with_sparse_mask

    res = fn(a, b, mask)
    res_gt = fn_gt(a, b, mask)

    if is_sparse:
        res = res.to_dense()
        res_gt = res_gt.to_dense()

    assert res.dtype == res_gt.dtype
    assert torch.allclose(res, res_gt)


@pytest.mark.parametrize("is_sparse", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", _devices)
def test_matmul_with_mask_backward(device, contiguous, is_sparse):
    if device == "cuda" and is_sparse is False:
        # Skip test for now due to bug in torch 1.8
        # See https://github.com/pytorch/pytorch/issues/54975
        # Broken CUDA / torch 1.8 combination, awaiting an update
        return

    B, L, K = 8, 10, 16
    prob = 0.5
    a = torch.rand(B, L, K, device=device, requires_grad=True)
    b = torch.rand(B, K, L, device=device, requires_grad=True)
    if not contiguous:
        a = a.detach().transpose(-2, -1).contiguous().transpose(-2, -1).requires_grad_()
        b = b.detach().transpose(-2, -1).contiguous().transpose(-2, -1).requires_grad_()
    mask = torch.rand(B, L, L, device=device) > prob

    fn = torch.ops.xformers.matmul_with_mask
    fn_gt = _baseline_matmul_with_dense_mask

    if is_sparse:
        mask = mask.to_sparse()
        fn_gt = _baseline_matmul_with_sparse_mask

    def compute_grads(f):
        out = f(a, b, mask)
        if is_sparse:
            out = out.to_dense()
        out.sum().backward()

    compute_grads(fn)
    grad_a = a.grad.clone()
    grad_b = b.grad.clone()
    a.grad = None
    b.grad = None
    compute_grads(fn_gt)
    assert torch.allclose(grad_a, a.grad)
    assert torch.allclose(grad_b, b.grad)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", _devices)
def test_sparse_bmm(device, contiguous):
    B, M, N = 8, 64, 32
    prob = 0.95
    a = torch.rand(B, M, N, device=device)
    a[a < prob] = 0
    a = a.to_sparse()
    b = torch.rand(B, N, M, device=device)
    if not contiguous:
        a = a + a
        b = b.transpose(-2, -1).contiguous().transpose(-2, -1)

    res = _sparse_bmm(a, b)
    res_gt = _baseline_sparse_bmm(a, b)

    assert torch.allclose(res, res_gt)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", _devices)
def test_sparse_bmm_backward(device, contiguous):
    if device == "cuda":
        # Skip test for now due to bug in torch 1.8
        # See https://github.com/pytorch/pytorch/issues/54975
        # Broken CUDA / torch 1.8 combination, awaiting an update
        return

    B, L, K = 8, 10, 16
    prob = 0.5
    a = torch.rand(B, L, K, device=device)
    a[a < prob] = 0
    a = a.to_sparse()
    b = torch.rand(B, K, L, device=device, requires_grad=True)
    if not contiguous:
        a = a + a
        b = b.detach().transpose(-2, -1).contiguous().transpose(-2, -1).requires_grad_()
    a.requires_grad_(True)

    def compute_grads(f):
        out = f(a, b)
        out.sum().backward()

    compute_grads(_sparse_bmm)
    grad_a = a.grad.clone().coalesce()
    grad_b = b.grad.clone()
    a.grad = None
    b.grad = None
    compute_grads(_baseline_sparse_bmm)
    new_grad_a = a.grad.coalesce()
    assert torch.allclose(grad_a.indices(), new_grad_a.indices())
    assert torch.allclose(grad_a.values(), new_grad_a.values())
    assert torch.allclose(grad_b, b.grad)
