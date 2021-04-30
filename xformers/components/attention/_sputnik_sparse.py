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
    def forward(ctx, a, b, row_indices, row_offsets, column_indices, _transp_info):
        out = torch.ops.xformers.sddmm_sputnik(
            a, b, row_indices, row_offsets, column_indices
        )

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

        grad_sparse = torch.ops.xformers.sddmm_sputnik(
            grad, b, row_indices, row_offsets, column_indices
        )

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
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.cpu().numpy()
        if matrix.ndim == 2:
            matrix = matrix[None]
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
        m, n = self.shape
        self._transp_info = _get_transpose_info(
            m, n, self.row_indices, self.row_offsets, self.column_indices
        )

    @property
    def device(self):
        return self.values.device

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
    return torch.argsort(torch.diff(a), dim=0, descending=True)


def _get_transpose_info(m, n, row_indices, row_offsets, column_indices):
    # strategy:
    # - uncompress the rows to have data in COO format
    # - get permutation for stable sort of the columns to get the rows for the transposed matrix
    # - compress the new rows and return the permutation to be applied on the values

    # convert from compressed rows to uncompressed
    indices = torch.arange(m, device=row_offsets.device)
    row_sizes = torch.diff(row_offsets)
    row_coo = torch.repeat_interleave(indices, row_sizes.long())

    # get the permutation for the stable sort
    # unfortunately PyTorch sort is not stable, so we need to
    # do it in two steps
    row_offsets_t, perm1 = column_indices.sort(0)
    new_columns = row_coo[perm1]

    # workaround the lack of stable sorting in PyTorch
    perm2 = torch.argsort(new_columns + row_offsets_t * m)
    column_indices_t = new_columns[perm2].int()

    # find the final permutation corresponding to the indices of the stable sort
    perm = perm1[perm2]

    row_offsets_t = torch.cat(
        [
            torch.zeros(1, dtype=row_offsets_t.dtype, device=row_offsets_t.device),
            row_offsets_t.bincount(minlength=n).cumsum(0),
        ]
    ).int()
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
