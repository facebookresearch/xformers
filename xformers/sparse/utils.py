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
    row_offsets_t, perm = column_indices.sort(dim=0, stable=True)
    column_indices_t = row_coo[perm]

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
