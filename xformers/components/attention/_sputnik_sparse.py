# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers.ops import masked_matmul
from xformers.sparse import SparseCSRTensor

# TODO: this is here for BC
from xformers.sparse.utils import _csr_to_coo, _dense_to_sparse  # noqa: F401


class SparseCS:
    def __init__(self, matrix, device=None):
        if device is None:
            device = torch.device("cpu")
        if matrix.ndim == 2:
            matrix = matrix[None]
        assert matrix.ndim == 3
        self._mat = SparseCSRTensor.from_dense(matrix).to(device)

    @property
    def device(self):
        return self._mat.device

    @property
    def ndim(self):
        return self._mat.ndim

    @property
    def dtype(self):
        return self._mat.dtype

    @property
    def is_sparse(self):
        return True

    @property
    def shape(self):
        return self._mat.shape[1:]

    @property
    def values(self):
        return self._mat.values()

    @property
    def row_indices(self):
        return self._mat._csr_row_indices

    @property
    def column_indices(self):
        return self._mat._csr_column_indices

    @property
    def row_offsets(self):
        return self._mat._csr_row_offsets

    @property
    def _transp_info(self):
        return self._mat._csr_transp_info

    @classmethod
    def wrap(
        cls, shape, values, row_indices, row_offsets, column_indices, _transp_info
    ):
        matrix = cls.__new__(cls)
        _shape = (values.shape[0],) + shape
        csr_matrix = SparseCSRTensor._wrap(
            _shape, values, row_indices, row_offsets, column_indices, _transp_info
        )
        matrix._mat = csr_matrix
        return matrix

    @classmethod
    def _wrap(cls, csr_matrix):
        assert isinstance(csr_matrix, SparseCSRTensor)
        matrix = cls.__new__(cls)
        matrix._mat = csr_matrix
        return matrix

    def __mul__(self, other):
        assert isinstance(other, (int, float))
        return type(self)._wrap(self._mat * other)

    def __add__(self, other):
        assert isinstance(other, type(self))
        return type(self)._wrap(self._mat + other._mat)

    def matmul_with_mask(self, a, b):
        return type(self)._wrap(masked_matmul(a, b, self._mat))

    def softmax(self):
        out = torch.nn.functional.softmax(self._mat, -1)
        return type(self)._wrap(out)

    def spmm(self, b):
        out = torch.bmm(self._mat, b)
        return out

    def transpose(self):
        out = torch.transpose(self._mat, -2, -1)
        return type(self)._wrap(out)

    def to(self, device):
        assert isinstance(device, torch.device)
        out = self._mat.to(device)
        return type(self)._wrap(out)

    def to_dense(self):
        return self._mat.to_dense()

    def logical_and(self, other: torch.Tensor):
        assert not isinstance(other, SparseCS)
        out = torch.logical_and(self._mat, other)
        return type(self)._wrap(out)

    def __and__(self, other):
        return self.logical_and(other)
