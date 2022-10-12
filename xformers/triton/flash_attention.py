# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch
import triton

from xformers.triton.k_flash_attn import _bwd_kernel, _bwd_preprocess, _fwd_kernel


class _flash_attention(torch.autograd.Function):
    @staticmethod
    def _flash_attn_forward(
        ctx, q, k, v, sm_scale, causal: bool = False, no_grad: bool = False
    ):
        BLOCK = 128
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
        tmp = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        L = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        m = torch.empty(
            (q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
            causal,
            tmp,
            L,
            m,
            o,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            BLOCK_M=BLOCK,
            BLOCK_N=BLOCK,
            BLOCK_DMODEL=Lk,
            num_warps=num_warps,
            num_stages=1,
        )
        if not no_grad:
            ctx.save_for_backward(q, k, v, o, L, m)
            ctx.BLOCK = BLOCK
            ctx.grid = grid
            ctx.sm_scale = sm_scale
            ctx.BLOCK_DMODEL = Lk
        return o

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        causal,
    ):
        return _flash_attention._flash_attn_forward(
            ctx, q, k, v, sm_scale, causal, no_grad=False
        )

    @staticmethod
    def forward_no_grad(
        ctx,
        q,
        k,
        v,
        sm_scale,
        causal,
    ):
        return _flash_attention._flash_attn_forward(
            ctx, q, k, v, sm_scale, causal, no_grad=True
        )

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, m = ctx.saved_tensors
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(l)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1],)](
            o,
            do,
            l,
            do_scaled,
            delta,
            BLOCK_M=ctx.BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )

        # NOTE: kernel currently buggy for other values of `num_warps`
        num_warps = 8
        _bwd_kernel[(ctx.grid[1],)](
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            l,
            m,
            delta,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            q.shape[0],
            q.shape[1],
            q.shape[2],
            ctx.grid[0],
            BLOCK_M=ctx.BLOCK,
            BLOCK_N=ctx.BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            num_warps=num_warps,
            num_stages=1,
        )
        return dq, dk, dv, None, None
