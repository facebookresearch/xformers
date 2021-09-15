# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch


def _coo_to_csr(m, n, row_indices, column_indices):
    # assumes coalesced coo
    row_offsets = row_indices.bincount(minlength=n).cumsum(0, dtype=row_indices.dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))
    return row_offsets, column_indices


def _csr_to_coo(m, n, row_offsets, column_indices):
    # convert from compressed rows to uncompressed
    indices = torch.arange(m, dtype=row_offsets.dtype, device=row_offsets.device)
    row_sizes = torch.diff(row_offsets)
    row_coo = torch.repeat_interleave(indices, row_sizes.long())
    return row_coo, column_indices


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


class SparseCS:
    def __init__(self, matrix, device=None):
        if device is None:
            device = torch.device("cpu")
        if matrix.ndim == 2:
            matrix = matrix[None]
        assert matrix.ndim == 3
        (
            self.values,
            self.row_indices,
            self.row_offsets,
            self.column_indices,
        ) = _dense3d_to_sparse(matrix, device)
        self.shape = matrix.shape[1:]
        m, n = self.shape
        self._transp_info = _get_transpose_info(
            m, n, self.row_indices, self.row_offsets, self.column_indices
        )

    @property
    def device(self):
        return self.values.device

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def is_sparse(self):
        return True

    @classmethod
    def wrap(
        cls, shape, values, row_indices, row_offsets, column_indices, _transp_info
    ):
        matrix = cls.__new__(cls)
        matrix.shape = shape
        matrix.values = values
        matrix.row_indices = row_indices
        matrix.row_offsets = row_offsets
        matrix.column_indices = column_indices
        matrix._transp_info = _transp_info
        return matrix

    def __mul__(self, other):
        out = self.values * other
        return type(self).wrap(
            self.shape,
            out,
            self.row_indices,
            self.row_offsets,
            self.column_indices,
            self._transp_info,
        )

    def __add__(self, other):
        assert isinstance(other, type(self))
        # TODO add cheap assert for indices
        out = self.values + other.values
        return type(self).wrap(
            self.shape,
            out,
            self.row_indices,
            self.row_offsets,
            self.column_indices,
            self._transp_info,
        )

    def matmul_with_mask(self, a, b):
        assert self.shape[0] == a.shape[1]
        assert self.shape[1] == b.shape[2]
        row_indices = self.row_indices
        row_offsets = self.row_offsets
        column_indices = self.column_indices
        out = _sddmm.apply(
            a,
            b.transpose(-2, -1),
            row_indices,
            row_offsets,
            column_indices,
            self._transp_info,
        )
        return type(self).wrap(
            self.shape, out, row_indices, row_offsets, column_indices, self._transp_info
        )

    def softmax(self):
        m, n = self.shape
        row_indices = self.row_indices
        values = self.values
        row_offsets = self.row_offsets
        column_indices = self.column_indices
        out = _SparseSoftmax.apply(
            m, n, row_indices, values, row_offsets, column_indices
        )
        return type(self).wrap(
            self.shape, out, row_indices, row_offsets, column_indices, self._transp_info
        )

    def spmm(self, b):
        m, n = self.shape
        row_indices = self.row_indices
        values = self.values
        row_offsets = self.row_offsets
        column_indices = self.column_indices
        out = _spmm.apply(
            b, row_indices, values, row_offsets, column_indices, m, self._transp_info
        )
        return out

    def transpose(self):
        m, n = self.shape
        values = self.values

        (
            output_row_indices,
            output_values,
            output_row_offsets,
            output_column_indices,
        ) = _transpose_with_info(values, self._transp_info)
        new_transp_info = _get_transpose_info(
            n, m, output_row_indices, output_row_offsets, output_column_indices
        )

        return type(self).wrap(
            (n, m),
            output_values,
            output_row_indices,
            output_row_offsets,
            output_column_indices,
            new_transp_info,
        )

    def to(self, device):
        assert isinstance(device, torch.device)
        return type(self).wrap(
            self.shape,
            self.values.to(device=device),
            self.row_indices.to(device=device),
            self.row_offsets.to(device=device),
            self.column_indices.to(device=device),
            tuple(t.to(device=device) for t in self._transp_info),
        )

    def to_dense(self):
        m, n = self.shape
        shape = (self.values.shape[0], m, n)
        matrix = torch.zeros(shape, dtype=self.values.dtype, device=self.values.device)
        row_offsets = self.row_offsets.long()
        column_indices = self.column_indices.long()
        row_coo, _ = _csr_to_coo(m, n, row_offsets, column_indices)
        b_idxs = torch.arange(len(self.values), device=self.values.device)[:, None]
        matrix[b_idxs, row_coo, column_indices] = self.values
        return matrix

    def logical_and(self, other: torch.Tensor):
        assert not isinstance(other, SparseCS)
        # FIXME: This is unecessarily slow, we should just walk through the intersect of the indices
        return SparseCS(self.to_dense() & other, device=self.device)

    def __and__(self, other):
        return self.logical_and(other)


def _diffsort(a):
    return torch.argsort(torch.diff(a), dim=0, descending=True)


def _get_transpose_info(m, n, row_indices, row_offsets, column_indices):
    # strategy:
    # - uncompress the rows to have data in COO format
    # - get permutation for stable sort of the columns to get the rows for the transposed matrix
    # - compress the new rows and return the permutation to be applied on the values

    # convert from compressed rows to uncompressed
    row_coo, _ = _csr_to_coo(m, n, row_offsets, column_indices)

    # get the permutation for the stable sort
    # unfortunately PyTorch sort is not stable, so we need to
    # do it in two steps
    row_offsets_t, perm1 = column_indices.sort(0)
    new_columns = row_coo[perm1]

    # workaround the lack of stable sorting in PyTorch
    perm2 = torch.argsort(new_columns + row_offsets_t * m)
    column_indices_t = new_columns[perm2]

    # find the final permutation corresponding to the indices of the stable sort
    perm = perm1[perm2]

    row_offsets_t, _ = _coo_to_csr(m, n, row_offsets_t, column_indices)
    row_indices_t = _diffsort(row_offsets_t).int()

    return row_indices_t, row_offsets_t, column_indices_t, perm


def _transpose_with_info(values, _transpose_info):
    row_indices_t, row_offsets_t, column_indices_t, perm = _transpose_info
    values_t = values[:, perm]
    return row_indices_t, values_t, row_offsets_t, column_indices_t


def _transpose(m, n, row_indices, values, row_offsets, column_indices):
    _transpose_info = _get_transpose_info(
        m, n, row_indices, row_offsets, column_indices
    )
    return _transpose_with_info(values, _transpose_info)


def _nonzero_mask_to_sparse_csr_indices(mask, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(mask.shape) == 2
    index_dtype = torch.int32

    # Calculate the offset of each row.
    row_offsets = mask.sum(dim=-1, dtype=index_dtype).cumsum(dim=-1, dtype=index_dtype)
    row_offsets = torch.nn.functional.pad(row_offsets, (1, 0))

    # Create the row indices and sort them.
    row_indices = _diffsort(row_offsets).to(index_dtype)

    # Extract the column indices for the nonzero values.
    column_indices = torch.where(mask)[1].to(index_dtype).contiguous()

    row_indices = row_indices.to(device)
    row_offsets = row_offsets.to(device)
    column_indices = column_indices.to(device)
    return row_indices, row_offsets, column_indices


def _dense_to_sparse(matrix, device):
    """Converts dense 2d matrix to a csr sparse matrix."""

    assert len(matrix.shape) == 2
    value_dtype = torch.float32

    # Extract the nonzero values.
    mask = matrix != 0
    values = matrix[mask].to(dtype=value_dtype, device=device)

    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(
        mask, device
    )
    return values, row_indices, row_offsets, column_indices


def _round_nnz(mask, divisible_by=4):
    nonzero = torch.where(mask)
    nnz = nonzero[0].shape[0]
    nonzero = tuple(n[: (nnz - nnz % divisible_by)] for n in nonzero)
    nm = torch.zeros_like(mask)
    nm[nonzero] = True
    return nm


def _dense3d_to_sparse(matrix, device):
    assert len(matrix.shape) == 3
    mask = matrix != 0
    if not torch.all(mask == mask[0]):
        raise ValueError("Expected the same sparsity pattern over the batch dimension")

    # for now, our kernels assume that we have the number of
    # nnz to be divisible by 4
    mask = _round_nnz(mask[0], divisible_by=4)
    mask = mask[None].expand(matrix.shape)

    values = matrix[mask].reshape(matrix.shape[0], -1).to(device)
    row_indices, row_offsets, column_indices = _nonzero_mask_to_sparse_csr_indices(
        mask[0], device
    )
    return values, row_indices, row_offsets, column_indices
