import pytest
import torch
import triton

from xformers.triton.flash_attention import _flash_attention

attention = _flash_attention.apply


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(3, 2, 2048, 64)])
def test_op(Z, H, N_CTX, D_HEAD, causal=True, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.3
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    for z in range(Z):
        for h in range(H):
            p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, sm_scale, causal)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    triton.testing.assert_almost_equal(ref_out, tri_out)
    triton.testing.assert_almost_equal(ref_dv, tri_dv)
    triton.testing.assert_almost_equal(ref_dk, tri_dk)
    triton.testing.assert_almost_equal(ref_dq, tri_dq)
