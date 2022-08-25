# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers.ops import masked_matmul
from xformers.sparse import _csr_ops
from xformers.sparse.utils import (
    _csr_to_coo,
    _dense3d_to_sparse,
    _diffsort,
    _get_transpose_info,
    _transpose_with_info,
)


class SparseCSRTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, row_offsets, column_indices, values, shape):
        kwargs = {}
        kwargs["device"] = values.device
        kwargs["dtype"] = values.dtype
        kwargs["layout"] = values.layout
        kwargs["requires_grad"] = values.requires_grad
        assert len(shape) == 3
        assert torch.__version__ > (1, 10), "SparseCSRTensor requires PyTorch 1.11+"
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, row_offsets, column_indices, values, shape):
        assert row_offsets.ndim == 1
        assert column_indices.ndim == 1
        assert values.ndim == 2

        self.__row_offsets = row_offsets.contiguous()
        self.__row_indices = _diffsort(row_offsets).to(row_offsets.dtype)
        self.__column_indices = column_indices.contiguous()
        self.__values = values.contiguous()

        self.__transp_info = _get_transpose_info(
            self.shape[1],
            self.shape[2],
            self.__row_indices,
            self.__row_offsets,
            self.__column_indices,
        )

    def __repr__(self):
        return f"sparse_csr_tensor(shape={self.shape}, values={self.__values})"

    @classmethod
    def from_dense(cls, matrix):
        values, row_indices, row_offsets, column_indices = _dense3d_to_sparse(
            matrix, matrix.device
        )
        return cls(row_offsets, column_indices, values, matrix.shape)

    @classmethod
    def from_sparse_coo(cls, arg0):
        """
        assert arg0.is_sparse
        x = arg0.coalesce()
        rows, cols = x.indices().unbind(0)
        vals = x.values()
        _coo_to_csr()
        """
        pass

    @classmethod
    def _wrap(
        cls, shape, values, row_indices, row_offsets, column_indices, _transp_info
    ):
        matrix = cls.__new__(cls, row_offsets, column_indices, values, shape)
        matrix.__values = values
        matrix.__row_indices = row_indices
        matrix.__row_offsets = row_offsets
        matrix.__column_indices = column_indices
        matrix.__transp_info = _transp_info
        return matrix

    def values(self):
        return self.__values

    @property
    def _csr_row_indices(self):
        return self.__row_indices

    @property
    def _csr_row_offsets(self):
        return self.__row_offsets

    @property
    def _csr_column_indices(self):
        return self.__column_indices

    @property
    def _csr_transp_info(self):
        return self.__transp_info

    @classmethod
    def _bmm(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and type(arg1) == torch.Tensor):
            return NotImplemented

        assert arg0.ndim == 3
        assert arg1.ndim == 3

        self = arg0
        b = arg1

        _, m, n = self.shape
        row_indices = self.__row_indices
        values = self.__values
        row_offsets = self.__row_offsets
        column_indices = self.__column_indices

        out = _csr_ops._spmm.apply(
            b, row_indices, values, row_offsets, column_indices, m, self.__transp_info
        )
        return out

    @classmethod
    def _softmax(cls, arg0, dim):
        if not (dim == -1 or dim == 2):
            return NotImplemented

        self = arg0
        _, m, n = self.shape
        row_indices = self.__row_indices
        values = self.__values
        row_offsets = self.__row_offsets
        column_indices = self.__column_indices
        out = _csr_ops._SparseSoftmax.apply(
            m, n, row_indices, values, row_offsets, column_indices
        )
        return cls._wrap(
            self.shape,
            out,
            row_indices,
            row_offsets,
            column_indices,
            self.__transp_info,
        )

    @classmethod
    def _transpose(cls, arg0, dim0, dim1):
        # TODO: check if need to return this or not
        if not (dim0 == 1 or dim0 == -2):
            return NotImplemented
        if not (dim1 == 2 or dim1 == -1):
            return NotImplemented

        B, m, n = arg0.shape
        values = arg0.__values

        (
            output_row_indices,
            output_values,
            output_row_offsets,
            output_column_indices,
        ) = _transpose_with_info(values, arg0.__transp_info)
        new_transp_info = _get_transpose_info(
            n, m, output_row_indices, output_row_offsets, output_column_indices
        )

        return cls._wrap(
            (B, n, m),
            output_values,
            output_row_indices,
            output_row_offsets,
            output_column_indices,
            new_transp_info,
        )

    @classmethod
    def _masked_matmul(cls, a, b, mask):
        if not (type(a) == torch.Tensor and type(b) == torch.Tensor):
            return NotImplemented
        assert mask.shape[1] == a.shape[1]
        assert mask.shape[2] == b.shape[2]
        row_indices = mask.__row_indices
        row_offsets = mask.__row_offsets
        column_indices = mask.__column_indices
        a = a.contiguous()
        out = _csr_ops._sddmm.apply(
            a,
            b.transpose(-2, -1).contiguous(),
            row_indices,
            row_offsets,
            column_indices,
            mask.__transp_info,
        )
        # TODO add bias here
        return cls._wrap(
            mask.shape,
            out,
            row_indices,
            row_offsets,
            column_indices,
            mask.__transp_info,
        )

    @classmethod
    def _to(cls, arg0, device):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        return cls._wrap(
            arg0.shape,
            arg0.__values.to(device=device),
            arg0.__row_indices.to(device=device),
            arg0.__row_offsets.to(device=device),
            arg0.__column_indices.to(device=device),
            tuple(t.to(device=device) for t in arg0.__transp_info),
        )

    @classmethod
    def _copy(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        assert arg0.shape == arg1.shape
        av0, av1 = arg0.__values, arg1.__values
        av0.resize_as_(av1).copy_(av1)
        av0, av1 = arg0.__row_indices, arg1.__row_indices
        av0.resize_as_(av1).copy_(av1)
        av0, av1 = arg0.__row_offsets, arg1.__row_offsets
        av0.resize_as_(av1).copy_(av1)
        av0, av1 = arg0.__column_indices, arg1.__column_indices
        av0.resize_as_(av1).copy_(av1)
        for v0, v1 in zip(arg0.__transp_info, arg1.__transp_info):
            v0.resize_as_(v1).copy_(v1)
        return arg0

    @classmethod
    def _equal(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        if arg0.shape != arg1.shape:
            return False
        if not torch.equal(arg0.__values, arg1.__values):
            return False
        if not torch.equal(arg0.__row_offsets, arg1.__row_offsets):
            return False
        if not torch.equal(arg0.__column_indices, arg1.__column_indices):
            return False
        return True

    @classmethod
    def _to_dense(cls, arg0):
        _, m, n = arg0.shape
        shape = arg0.shape
        matrix = torch.zeros(shape, dtype=arg0.dtype, device=arg0.device)
        row_offsets = arg0.__row_offsets.long()
        column_indices = arg0.__column_indices.long()
        row_coo, _ = _csr_to_coo(m, n, row_offsets, column_indices)
        b_idxs = torch.arange(len(arg0.__values), device=arg0.device)[:, None]
        matrix[b_idxs, row_coo, column_indices] = arg0.__values
        return matrix

    @classmethod
    def _binary_op(cls, func, arg0, arg1):
        if not (
            isinstance(arg0, (cls, int, float)) and isinstance(arg1, (cls, int, float))
        ):
            return NotImplemented
        v0, v1 = arg0, arg1
        if isinstance(arg0, cls):
            v0 = arg0.__values
        if isinstance(arg1, cls):
            v1 = arg1.__values
        # assert arg0.shape == arg1.shape
        if isinstance(arg0, cls) and isinstance(arg1, cls):
            msg = f"arg0 and arg1 need to have the same sparsity pattern in {func} (for now)"
            if not arg0.__row_offsets.shape == arg1.__row_offsets.shape:
                raise NotImplementedError(msg)
            if not arg0.__column_indices.shape == arg1.__column_indices.shape:
                raise NotImplementedError(msg)
            if not arg0.__values.shape == arg1.__values.shape:
                raise NotImplementedError(msg)
            # TODO this is not always true, but is a fast approximation for now
            if arg0.__row_offsets is not arg1.__row_offsets:
                raise NotImplementedError(msg)
            if arg0.__column_indices is not arg1.__column_indices:
                raise NotImplementedError(msg)
        out = func(v0, v1)
        return cls._wrap(
            arg0.shape,
            out,
            arg0.__row_indices,
            arg0.__row_offsets,
            arg0.__column_indices,
            arg0.__transp_info,
        )

    @classmethod
    def _binary_op_slow(cls, func, arg0, arg1):
        # assert arg0.shape == arg1.shape
        v0, v1 = arg0, arg1
        if isinstance(arg0, cls):
            v0 = arg0.to_dense()
        if isinstance(arg1, cls):
            v1 = arg1.to_dense()
        out = func(v0, v1)
        return cls.from_dense(out)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in [
            torch.Tensor.bmm,
            torch.bmm,
            torch.Tensor.__matmul__,
            torch.matmul,
            torch.Tensor.matmul,
        ]:
            assert len(args) == 2
            return cls._bmm(args[0], args[1])

        if func in [torch.Tensor.softmax, torch.nn.functional.softmax, torch.softmax]:
            return cls._softmax(args[0], kwargs["dim"])

        if func in [torch.Tensor.transpose, torch.transpose]:
            assert len(kwargs) == 0
            return cls._transpose(args[0], args[1], args[2])

        if func == masked_matmul:
            assert len(args) == 3
            return cls._masked_matmul(args[0], args[1], args[2])

        if func in [
            torch.Tensor.add,
            torch.add,
            torch.Tensor.__add__,
        ]:
            assert len(args) == 2
            if not (isinstance(args[0], cls) and isinstance(args[1], cls)):
                raise NotImplementedError(
                    f"{func} with {type(args[0])} and {type(args[1])} not implemented"
                )
            return cls._binary_op(func, args[0], args[1])

        if func in [
            torch.Tensor.mul,
            torch.mul,
            torch.Tensor.__mul__,
        ]:
            assert len(args) == 2
            return cls._binary_op(func, args[0], args[1])

        if func in [torch.Tensor.logical_and, torch.logical_and, torch.Tensor.__and__]:
            assert len(args) == 2
            return cls._binary_op_slow(func, args[0], args[1])

        if func in [torch.nn.functional.dropout, torch.dropout, torch.dropout_]:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(
                x.shape,
                values,
                x.__row_indices,
                x.__row_offsets,
                x.__column_indices,
                x.__transp_info,
            )

        if func == torch.Tensor.to:
            # print(args, kwargs)
            assert len(args) >= 2
            return cls._to(args[0], args[1])
            # return cls._to(args[0], kwargs["device"])

        if func in [torch.Tensor.copy_]:
            assert len(args) == 2
            return cls._copy(args[0], args[1])

        if func in [torch.Tensor.equal, torch.equal]:
            assert len(args) == 2
            return cls._equal(args[0], args[1])

        if func == torch.Tensor.to_dense:
            assert len(args) == 1
            return cls._to_dense(args[0])

        if func == torch.Tensor.detach:
            x = args[0]
            return cls._wrap(
                x.shape,
                x.__values.detach(),
                x.__row_indices,
                x.__row_offsets,
                x.__column_indices,
                x.__transp_info,
            )

        if func == torch.Tensor.__deepcopy__:
            x = args[0]
            memo = args[1]
            return cls._wrap(
                x.shape,
                x.__values.__deepcopy__(memo),
                x.__row_indices.__deepcopy__(memo),
                x.__row_offsets.__deepcopy__(memo),
                x.__column_indices.__deepcopy__(memo),
                tuple(v.__deepcopy__(memo) for v in x.__transp_info),
            )

        if func in [torch.Tensor.grad.__get__, torch.Tensor._grad.__get__]:
            assert len(args) == 1
            assert len(kwargs) == 0
            x = args[0]
            return cls._wrap(
                x.shape,
                x.__values.grad,
                x.__row_indices,
                x.__row_offsets,
                x.__column_indices,
                x.__transp_info,
            )

        if func == torch.Tensor.requires_grad_:
            func(args[0].__values)

        with torch._C.DisableTorchFunction():
            ret = func(*args, **kwargs)
            # TODO: check this
            if func in torch.overrides.get_default_nowrap_functions():
                return ret
            return torch._tensor._convert(ret, cls)

        return NotImplemented

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        return NotImplemented
