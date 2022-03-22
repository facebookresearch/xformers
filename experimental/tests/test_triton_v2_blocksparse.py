# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch

import xformers
#turn off xformers triton 1.X access to avoid triton version class 
xformers.triton.softmax.softmax = lambda x: torch.softmax(x,dim=a.ndim-1)#_is_triton_available=False

from xformers.components import MultiHeadDispatch
from xformers.components.attention import build_attention
from xformers.components.attention.attention_patterns import block_sparsify_tensor
from xformers.triton.utils import get_current_cuda_device

# CREDITS:
# Tests inherited from both xformers 0.10.0 triton tests which were in turn copied from 
# https://github.com/openai/triton/blob/master/python/test/unit/operators/test_blocksparse.py
# and Triton 2.0 tests copied from
# https://github.com/openai/triton/blob/v2.0/python/test/unit/operators/test_blocksparse.py


_triton_available = torch.cuda.is_available()
_matmul_types = []

if _triton_available:
    try:
        import triton
        from triton.ops.blocksparse import matmul as blocksparse_matmul
        from triton.ops.blocksparse import softmax as blocksparse_softmax
        
        #FIXME: This should just import standard xFormers attention after porting
        from triton_v2_blocksparse import BlockSparseAttentionV2 as BlockSparseAttention
        from xformers.triton.utils import (
            assert_almost_equal,
            gpu_capabilities_older_than_70,
        )

        _triton_available = not gpu_capabilities_older_than_70()
        _matmul_types = ["sdd", "dsd", "dds"]
    except (ImportError, ModuleNotFoundError) as e:
        import logging

        logging.warning(f"Triton is not available: {e}. Some tests will be skipped")
        _triton_available = False


@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.skipif(
    not _triton_available or get_current_cuda_device() == "T4",
    reason="FIXME - blocksparse matmuls are slightly off on T4s",
)
@pytest.mark.parametrize("MODE", _matmul_types)
@pytest.mark.parametrize("TRANS_A", [False, True])
@pytest.mark.parametrize("TRANS_B", [False, True])
@pytest.mark.parametrize("BLOCK", [16, 32, 64, 128])
@pytest.mark.parametrize("DTYPE", [torch.float16,torch.bfloat16,torch.float32,torch.float])
def test_matmul(MODE, TRANS_A, TRANS_B, BLOCK, DTYPE, Z=32, H=2, M=512, N=384, K=384):
    seed = 0
    torch.manual_seed(seed)
    is_sdd = MODE == "sdd"
    is_dsd = MODE == "dsd"
    is_dds = MODE == "dds"
    do_sparsify = lambda x: triton.testing.sparsify_tensor(x, layout, BLOCK)
    do_mask = lambda x: triton.testing.mask_tensor(x, layout, BLOCK)
    # create inputs
    # create op
    a_shape = (Z, H, K, M) if TRANS_A else (Z, H, M, K)
    b_shape = (Z, H, N, K) if TRANS_B else (Z, H, K, N)
    c_shape = (Z, H, M, N)
    shape = {
        "sdd": (M, N),
        "dsd": (a_shape[2], a_shape[3]),
        "dds": (b_shape[2], b_shape[3]),
    }[MODE]
    layout = torch.randint(2, (H, shape[0] // BLOCK, shape[1] // BLOCK))
    layout[1, 2, :] = 0
    layout[1, :, 1] = 0
    # create data
    a_ref, a_tri = triton.testing.make_pair(a_shape, alpha=.1)
    b_ref, b_tri = triton.testing.make_pair(b_shape, alpha=.1)
    dc_ref, dc_tri = triton.testing.make_pair(c_shape)
    # compute [torch]
    dc_ref = do_mask(dc_ref) if is_sdd else dc_ref
    a_ref = do_mask(a_ref) if is_dsd else a_ref
    b_ref = do_mask(b_ref) if is_dds else b_ref
    a_ref.retain_grad()
    b_ref.retain_grad()
    c_ref = torch.matmul(a_ref.transpose(2, 3) if TRANS_A else a_ref,
                         b_ref.transpose(2, 3) if TRANS_B else b_ref)
    c_ref.backward(dc_ref)
    c_ref = do_sparsify(c_ref) if is_sdd else c_ref
    da_ref = do_sparsify(a_ref.grad) if is_dsd else a_ref.grad
    db_ref = do_sparsify(b_ref.grad) if is_dds else b_ref.grad
    # triton result
    dc_tri = do_sparsify(dc_tri) if is_sdd else dc_tri
    a_tri = do_sparsify(a_tri) if is_dsd else a_tri
    b_tri = do_sparsify(b_tri) if is_dds else b_tri
    a_tri.retain_grad()
    b_tri.retain_grad()
    op = triton.ops.blocksparse.matmul(layout, BLOCK, MODE, trans_a=TRANS_A, trans_b=TRANS_B, device="cuda")
    c_tri = triton.testing.catch_oor(lambda: op(a_tri, b_tri), pytest)
    triton.testing.catch_oor(lambda: c_tri.backward(dc_tri), pytest)
    da_tri = a_tri.grad
    db_tri = b_tri.grad
    # compare
    triton.testing.assert_almost_equal(c_ref, c_tri)
    triton.testing.assert_almost_equal(da_ref, da_tri)
    triton.testing.assert_almost_equal(db_ref, db_tri)

configs = [
    (16, 256),
    (32, 576),
    (64, 1871),
    (128, 2511),
]

@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.parametrize("is_dense", [False, True])
@pytest.mark.parametrize("BLOCK, WIDTH", configs)
def test_softmax(BLOCK, WIDTH, is_dense, Z=2, H=2, is_causal=True, scale=0.4):
    # set seed
    torch.random.manual_seed(0)
    Z, H, M, N = 2, 3, WIDTH, WIDTH
    # initialize layout
    # make sure each row has at least one non-zero element
    layout = torch.randint(2, (H, M // BLOCK, N // BLOCK))
    if is_dense:
        layout[:] = 1
    else:
        layout[1, 2, :] = 0
        layout[1, :, 1] = 0
    # initialize data
    a_shape = (Z, H, M, N)
    a_ref, a_tri = triton.testing.make_pair(a_shape)
    dout_ref, dout_tri = triton.testing.make_pair(a_shape)
    # compute [torch]
    a_ref = triton.testing.mask_tensor(a_ref, layout, BLOCK, value=float("-inf"))
    a_ref.retain_grad()
    at_mask = torch.ones((M, N), device="cuda")
    if is_causal:
        at_mask = torch.tril(at_mask)
    M = at_mask[None, None, :, :] + torch.zeros_like(a_ref)
    a_ref[M == 0] = float("-inf")
    out_ref = torch.softmax(a_ref * scale, -1)
    out_ref.backward(dout_ref)
    out_ref = triton.testing.sparsify_tensor(out_ref, layout, BLOCK)
    da_ref = triton.testing.sparsify_tensor(a_ref.grad, layout, BLOCK)
    # compute [triton]
    a_tri = triton.testing.sparsify_tensor(a_tri, layout, BLOCK)
    a_tri.retain_grad()
    dout_tri = triton.testing.sparsify_tensor(dout_tri, layout, BLOCK)
    op = triton.ops.blocksparse.softmax(layout, BLOCK, device="cuda", is_dense=is_dense)
    out_tri = op(a_tri, scale=scale, is_causal=is_causal)
    out_tri.backward(dout_tri)
    da_tri = a_tri.grad
    # compare
    triton.testing.assert_almost_equal(out_tri, out_ref)
    triton.testing.assert_almost_equal(da_tri, da_ref)


@pytest.mark.skipif(not _triton_available, reason="Triton requires a recent CUDA gpu")
@pytest.mark.parametrize("block", [32, 43, 64, 128])  # 16, 32, 64
@pytest.mark.parametrize("is_causal", [True,False])  # 16, 32, 64
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float])
def test_attention_fwd_bwd(
    block,
    is_causal,
    dtype,
    input_scale=0.4,#1.0,
    scale=1 / 8.0,
    n_ctx=256,
    batch_size=2,
    n_heads=2,
):
    # inputs
    head_dim = 64
    qkv_shape = (batch_size, n_heads, n_ctx, head_dim)
    qkvs = [
        torch.nn.Parameter(input_scale * torch.randn(qkv_shape), requires_grad=True)
        .to(dtype)
        .cuda()
        for _ in range(3)
    ]
    if is_causal:
        attn_mask = torch.tril(
            torch.ones(
                [n_ctx, n_ctx],
                device="cuda",
            ),
            diagonal=0,
        ).to(dtype)
    else:
        attn_mask=torch.ones([n_ctx, n_ctx],device="cuda").to(dtype)

    def loss_fn(x):
        return (x ** 2).mean()

    # Triton:
    n_blocks = n_ctx // block
    layout = torch.tril(torch.ones([n_heads, n_blocks, n_blocks], dtype=torch.long))
    query, key, value = [x.clone() for x in qkvs]
    query.retain_grad()
    key.retain_grad()
    value.retain_grad()
    if block not in [16, 32, 64, 128]:
        # Check that unsupported dimensions are caught
        with pytest.raises(AssertionError):
            _ = BlockSparseAttention(layout, block)
    else:
        block_sparse_attention = BlockSparseAttention(layout, block)
        attn_out = block_sparse_attention(
            is_causal=is_causal, q=query, k=key, v=value, scale=scale
        )

        # ad hoc loss
        loss = loss_fn(attn_out)
        loss.backward()
        grads = [query.grad, key.grad, value.grad]

        # Torch version:
        torch_q, torch_k, torch_v = [x.clone() for x in qkvs]
        torch_q = torch_q / math.sqrt(head_dim)
        attn_mask = 1e6 * (-1 + (attn_mask.reshape((1, 1, n_ctx, n_ctx)).cuda()))
        torch_q.retain_grad()
        torch_k.retain_grad()
        torch_v.retain_grad()
        scores = scale * torch.einsum("bhsd,bhtd->bhst", torch_q, torch_k)
        scores = scores + attn_mask
        probs = torch.softmax(scores, dim=-1)
        torch_attn_out = torch.einsum("bhst,bhtd->bhsd", probs, torch_v)

        # ad hoc loss
        torch_loss = loss_fn(torch_attn_out)
        torch_loss.backward()
        torch_grads = [torch_q.grad, torch_k.grad, torch_v.grad]

        # comparison
        assert_almost_equal(
            loss.float(), torch_loss.float(), err_msg=f"Triton loss {loss} and torch loss {torch_loss}"
        )

        #need to convert back to float, bfloat causes norm issues
        for g1, g2 in zip(grads, torch_grads):
            assert_almost_equal(
                torch.norm(g1).float(),
                torch.norm(g2).float(),
                err_msg=f"Triton grad {torch.norm(g1).item()} and torch grad {torch.norm(g2).item()}",
            )
