# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Mapping

import pytest
import torch

_triton_available = torch.cuda.is_available()
if _triton_available:
    try:
        from xformers.triton.flash_attention import _flash_attention
        from xformers.triton.utils import assert_allclose
    except (ImportError, ModuleNotFoundError):
        _triton_available = False

ERROR_ATOL: Mapping[torch.dtype, float] = {
    torch.half: 4e-3,
    torch.bfloat16: 2e-2,
}
ERROR_RTOL: Mapping[torch.dtype, float] = {
    torch.half: 4e-4,
    torch.bfloat16: 5e-3,
}

if _triton_available:

    attention = _flash_attention.apply

    @pytest.mark.parametrize("causal", [True, False])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(3, 2, 2048, 64)])
    def test_op(Z, H, N_CTX, D_HEAD, causal, dtype):
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
        if causal:
            for z in range(Z):
                for h in range(H):
                    p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).to(dtype)
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

        assert_allclose(
            ref_out, tri_out, rtol=ERROR_RTOL[dtype], atol=ERROR_ATOL[dtype]
        )
        # when bf16 is used, max diff is .016, .0006% of elements are not within .015 of ref
        assert_allclose(ref_dv, tri_dv, rtol=ERROR_RTOL[dtype], atol=ERROR_ATOL[dtype])
        assert_allclose(ref_dk, tri_dk, rtol=ERROR_RTOL[dtype], atol=ERROR_ATOL[dtype])
        assert_allclose(ref_dq, tri_dq, rtol=ERROR_RTOL[dtype], atol=ERROR_ATOL[dtype])
