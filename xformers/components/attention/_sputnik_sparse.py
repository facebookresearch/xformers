import torch


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
    def forward(ctx, a, b, row_indices, row_offsets, column_indices):
        out = torch.ops.xformers.sddmm_sputnik(
            a, b, row_indices, row_offsets, column_indices
        )

        ctx.save_for_backward(a, b, row_indices, row_offsets, column_indices)
        return out

    @staticmethod
    def backward(ctx, grad):
        a, b, row_indices, row_offsets, column_indices = ctx.saved_tensors
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
        ) = _transpose(m, n, row_indices, grad, row_offsets, column_indices)

        b_grad = torch.ops.xformers.spmm_sputnik(
            a, row_indices_t, grad_t, row_offsets_t, column_indices_t, n
        )

        return a_grad, b_grad, None, None, None


class _spmm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, b, row_indices, values, row_offsets, column_indices, m):
        out = torch.ops.xformers.spmm_sputnik(
            b, row_indices, values, row_offsets, column_indices, m
        )

        ctx.save_for_backward(b, row_indices, values, row_offsets, column_indices)
        ctx.m = m
        return out

    @staticmethod
    def backward(ctx, grad):
        b, row_indices, values, row_offsets, column_indices = ctx.saved_tensors
        m = ctx.m
        k = b.shape[1]

        # gradients w.r.t. values
        grad = grad.contiguous()

        grad_sparse = torch.ops.xformers.sddmm_sputnik(
            grad, b.T, row_indices, row_offsets, column_indices
        )

        (
            row_indices_t,
            values_t,
            row_offsets_t,
            column_indices_t,
        ) = _transpose(m, k, row_indices, values, row_offsets, column_indices)

        grad_dense = torch.ops.xformers.spmm_sputnik(
            grad, row_indices_t, values_t, row_offsets_t, column_indices_t, k
        )

        return grad_dense, None, grad_sparse, None, None, None


class SparseCS:
    def __init__(self, matrix, device):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
        assert matrix.ndim == 3
        (
            self.values,
            self.row_indices,
            self.row_offsets,
            self.column_indices,
        ) = _dense_to_sparse(matrix[0], device)
        matrix = torch.as_tensor(matrix, device=device)
        self.values = matrix[(matrix[0] != 0).expand_as(matrix)].reshape(
            matrix.shape[0], -1
        )
        self.shape = matrix.shape[1:]

    @classmethod
    def wrap(cls, shape, values, row_indices, row_offsets, column_indices):
        matrix = cls.__new__(cls)
        matrix.shape = shape
        matrix.values = values
        matrix.row_indices = row_indices
        matrix.row_offsets = row_offsets
        matrix.column_indices = column_indices
        return matrix

    def __mul__(self, other):
        out = self.values * other
        return type(self).wrap(
            self.shape, out, self.row_indices, self.row_offsets, self.column_indices
        )

    def matmul_with_mask(self, a, b):
        row_indices = self.row_indices
        row_offsets = self.row_offsets
        column_indices = self.column_indices
        out = _sddmm.apply(
            a, b.transpose(-2, -1), row_indices, row_offsets, column_indices
        )
        return type(self).wrap(
            self.shape, out, row_indices, row_offsets, column_indices
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
            self.shape, out, row_indices, row_offsets, column_indices
        )

    def spmm(self, b):
        m, n = self.shape
        row_indices = self.row_indices
        values = self.values
        row_offsets = self.row_offsets
        column_indices = self.column_indices
        out = _spmm.apply(b, row_indices, values, row_offsets, column_indices, m)
        return out

    def transpose(self):
        m, n = self.shape
        row_indices = self.row_indices
        values = self.values
        row_offsets = self.row_offsets
        column_indices = self.column_indices

        (
            output_row_indices,
            output_values,
            output_row_offsets,
            output_column_indices,
        ) = _transpose(m, n, row_indices, values, row_offsets, column_indices)

        return type(self).wrap(
            (n, m),
            output_values,
            output_row_indices,
            output_row_offsets,
            output_column_indices,
        )

    def to(self, device):
        assert isinstance(device, torch.device)
        return type(self).wrap(
            self.shape,
            self.values.to(device=device),
            self.row_indices.to(device=device),
            self.row_offsets.to(device=device),
            self.column_indices.to(device=device),
        )

    def to_dense(self):
        shape = (self.values.shape[0],) + self.shape
        matrix = torch.zeros(shape, dtype=self.values.dtype, device=self.values.device)
        sizes = torch.diff(self.row_offsets).long()
        idxs = torch.arange(self.shape[0], device=self.values.device)
        r_idxs = torch.repeat_interleave(idxs, sizes)
        for i, v in enumerate(self.values):
            matrix[i, r_idxs, self.column_indices.long()] = v
            # matrix[r_idxs, self.column_indices.long()] = self.values
        return matrix


def _diffsort(a):
    return torch.argsort(torch.diff(a), 0, True)


def _transpose(m, n, row_indices, values, row_offsets, column_indices):
    ar = torch.arange(m, device=values.device)
    row_sizes = torch.diff(row_offsets)
    row_indices = torch.repeat_interleave(ar, row_sizes.long())

    new_rows, idxs = column_indices.sort(0)
    new_columns = row_indices[idxs]

    n_idxs = torch.argsort(new_columns + new_rows * m)
    correct_columns = new_columns[n_idxs]

    new_values = values[:, idxs][:, n_idxs]

    new_rows = torch.cat(
        [
            torch.zeros(1, dtype=new_rows.dtype, device=new_rows.device),
            new_rows.bincount(minlength=n).cumsum(0),
        ]
    )
    ro = _diffsort(new_rows)

    return ro.int(), new_values, new_rows.int(), correct_columns.int()


def _dense_to_sparse(matrix, device):
    import numpy as np

    """Converts dense numpy matrix to a csr sparse matrix."""
    assert len(matrix.shape) == 2

    if True:
        nonzero = np.nonzero(matrix != 0)
        nnz = nonzero[0].shape[0]
        # NOTE: need to make it a multiple of 4 for sputnik
        nonzero = tuple(n[: (nnz - nnz % 4)] for n in nonzero)
        nm = np.zeros_like(matrix)
        nm[nonzero] = matrix[nonzero]
        matrix = nm

    # Extract the nonzero values.
    values = matrix.compress((matrix != 0).flatten())

    # Calculate the offset of each row.
    mask = (matrix != 0).astype(np.int32)
    row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))), axis=0)

    # Create the row indices and sort them.
    row_indices = np.argsort(-1 * np.diff(row_offsets))

    # Extract the column indices for the nonzero values.
    x = mask * (np.arange(matrix.shape[1]) + 1)
    column_indices = x.compress((x != 0).flatten())
    column_indices = column_indices - 1

    # Cast the desired precision.
    values = torch.as_tensor(values.astype(np.float32), device=device)
    row_indices, row_offsets, column_indices = [
        # x.astype(np.uint32) for x in
        torch.as_tensor(x.astype(np.int32), device=device)
        for x in [row_indices, row_offsets, column_indices]
    ]
    return values, row_indices, row_offsets, column_indices
