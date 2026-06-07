# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.ops.rotary import apply_rotary_emb


def _make_freqs(seq_len, rot_dim, theta=10000.0, device="cpu"):
    inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, device=device).float() / rot_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _ref_apply_rotary(t, cos, sin):
    rot_dim = cos.shape[-1]
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    x1, x2 = t_rot.chunk(2, dim=-1)
    t_rot_out = t_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    return t_rot_out if t_pass.shape[-1] == 0 else torch.cat([t_rot_out, t_pass], dim=-1)


@pytest.mark.parametrize("seq_len", [1, 17, 128])
@pytest.mark.parametrize("dim", [32, 64, 128])
@pytest.mark.parametrize("rotary_dim", [None, 16, 32])
def test_apply_rotary_emb_cpu(seq_len, dim, rotary_dim):
    rot_dim = rotary_dim if rotary_dim is not None else dim
    if rot_dim > dim or rot_dim % 2 != 0:
        pytest.skip("rotary_dim must be <= dim and even")

    torch.manual_seed(42)
    q = torch.randn(seq_len, 4, dim)
    k = torch.randn(seq_len, 4, dim)
    cos, sin = _make_freqs(seq_len, rot_dim * 2)
    cos = cos[:, :rot_dim]
    sin = sin[:, :rot_dim]

    q_out, k_out = apply_rotary_emb(q, k, cos, sin)
    q_ref = _ref_apply_rotary(q, cos, sin)
    k_ref = _ref_apply_rotary(k, cos, sin)

    assert torch.allclose(q_out, q_ref, atol=1e-6), (
        f"seq_len={seq_len}, dim={dim}, rot_dim={rot_dim}: q max diff={((q_out - q_ref).abs().max()).item()}"
    )
    assert torch.allclose(k_out, k_ref, atol=1e-6), (
        f"k max diff={((k_out - k_ref).abs().max()).item()}"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_apply_rotary_emb_grad_flow(dtype):
    seq_len, n_heads, dim = 8, 4, 64
    rot_dim = 64
    torch.manual_seed(42)
    q = torch.randn(seq_len, n_heads, dim, dtype=dtype, requires_grad=True)
    k = torch.randn(seq_len, n_heads, dim, dtype=dtype, requires_grad=True)
    cos, sin = _make_freqs(seq_len, rot_dim)

    q_out, k_out = apply_rotary_emb(q, k, cos, sin)
    loss = q_out.sum() + k_out.sum()
    loss.backward()

    assert q.grad is not None
    assert k.grad is not None
    assert not torch.isnan(q.grad).any(), "NaNs in q grad"
    assert not torch.isnan(k.grad).any(), "NaNs in k grad"
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape


def test_apply_rotary_emb_partial_rotary():
    seq_len, n_heads, dim, rot_dim = 8, 4, 128, 64
    torch.manual_seed(42)
    q = torch.randn(seq_len, n_heads, dim)
    k = torch.randn(seq_len, n_heads, dim)
    cos, sin = _make_freqs(seq_len, rot_dim)

    q_out, k_out = apply_rotary_emb(q, k, cos, sin)
    q_ref = _ref_apply_rotary(q, cos, sin)
    k_ref = _ref_apply_rotary(k, cos, sin)

    assert torch.allclose(q_out, q_ref, atol=1e-6)
    assert torch.allclose(k_out, k_ref, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_apply_rotary_emb_cuda():
    seq_len, n_heads, dim, rot_dim = 16, 4, 64, 64
    torch.manual_seed(42)
    q = torch.randn(seq_len, n_heads, dim, device="cuda")
    k = torch.randn(seq_len, n_heads, dim, device="cuda")
    cos, sin = _make_freqs(seq_len, rot_dim, device="cuda")

    q_out, k_out = apply_rotary_emb(q, k, cos, sin)
    q_ref = _ref_apply_rotary(q, cos, sin)
    k_ref = _ref_apply_rotary(k, cos, sin)

    assert torch.allclose(q_out.cpu(), q_ref.cpu(), atol=1e-5), (
        f"q max diff={((q_out - q_ref).abs().max()).item()}"
    )
    assert torch.allclose(k_out.cpu(), k_ref.cpu(), atol=1e-5)
