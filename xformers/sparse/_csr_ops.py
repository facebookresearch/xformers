# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from .utils import _csr_to_coo, _transpose_with_info


def _should_use_coo(a, sparsity):
    if not a.is_cuda:
        return False
    B, M, K = a.shape
    # amortize overhead of converting from csr to coo
    if B < 32 and M < 4096:
        return False
    if sparsity > 0.995:
        return False
    if sparsity < 0.9:
        return False
    if K > 64:
        return False
    # let's be overly cautious here for now
    return sparsity > 0.97


def _should_use_csr_ge(a, sparsity):
    if not a.is_cuda:
        return False
    return sparsity > 0.99


def _sddmm_func(a, b, row_indices, row_offsets, column_indices):
    sparsity = 1 - column_indices.shape[0] / (a.shape[1] * b.shape[1])
    if _should_use_coo(a, sparsity):
        m = a.shape[-2]
        n = b.shape[-2]
        # converting from csr to coo has a constant overhead of ~150us
        # so only dispatch to it for reasonably large problem sizes
        ro, ci = _csr_to_coo(m, n, row_offsets, column_indices)
        return torch.ops.xformers.coo_sddmm(a, b, row_indices, ro, ci)
    elif _should_use_csr_ge(a, sparsity):
        return torch.ops.xformers.csr_sddmm(
            a, b, row_indices, row_offsets, column_indices
        )
    return torch.ops.xformers.sddmm_sputnik(
        a, b, row_indices, row_offsets, column_indices
    )


class _SparseSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, n, row_indices, values, row_offsets, column_indices):
        out = torch.ops.xformers.sparse_softmax_sputnik(
            m, n, row_indices, values, row_offsets, column_indices
        )
        # note: save out and not values, as an optimization step
        ctx.save_for_backward(row_indices, out, row_offsets, column_indices)
        ctx.size = (m, n)
        return out

    @staticmethod
    def backward(ctx, grad):
        row_indices, out, row_offsets, column_indices = ctx.saved_tensors
        m, n = ctx.size

        # gradients w.r.t. values
        grad = grad.contiguous()
        ga = torch.ops.xformers.sparse_softmax_backward_sputnik(
            m, n, row_indices, out, grad, row_offsets, column_indices
        )

        return None, None, None, ga, None, None


class _sddmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, row_indices, row_offsets, column_indices, _transp_info):
        out = _sddmm_func(a, b, row_indices, row_offsets, column_indices)

        ctx.save_for_backward(
            a, b, row_indices, row_offsets, column_indices, *_transp_info
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        (
            a,
            b,
            row_indices,
            row_offsets,
            column_indices,
            *_transp_info,
        ) = ctx.saved_tensors
        m, n = a.shape[1], b.shape[1]

        # gradients w.r.t. values
        grad = grad.contiguous()
        a = a.contiguous()
        b = b.contiguous()

        a_grad = torch.ops.xformers.spmm_sputnik(
            b, row_indices, grad, row_offsets, column_indices, m
        )

        (
            row_indices_t,
            grad_t,
            row_offsets_t,
            column_indices_t,
        ) = _transpose_with_info(grad, _transp_info)

        b_grad = torch.ops.xformers.spmm_sputnik(
            a, row_indices_t, grad_t, row_offsets_t, column_indices_t, n
        )

        return a_grad, b_grad, None, None, None, None


class _spmm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, b, row_indices, values, row_offsets, column_indices, m, _transp_info
    ):
        b = b.contiguous()
        out = torch.ops.xformers.spmm_sputnik(
            b, row_indices, values, row_offsets, column_indices, m
        )

        ctx.save_for_backward(
            b, row_indices, values, row_offsets, column_indices, *_transp_info
        )
        return out

    @staticmethod
    def backward(ctx, grad):
        (
            b,
            row_indices,
            values,
            row_offsets,
            column_indices,
            *_transp_info,
        ) = ctx.saved_tensors
        k = b.shape[1]

        # gradients w.r.t. values
        grad = grad.contiguous()

        grad_sparse = _sddmm_func(grad, b, row_indices, row_offsets, column_indices)

        (
            row_indices_t,
            values_t,
            row_offsets_t,
            column_indices_t,
        ) = _transpose_with_info(values, _transp_info)

        grad_dense = torch.ops.xformers.spmm_sputnik(
            grad, row_indices_t, values_t, row_offsets_t, column_indices_t, k
        )

        return grad_dense, None, grad_sparse, None, None, None, None
