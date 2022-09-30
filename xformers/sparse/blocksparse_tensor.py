# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from xformers.ops import masked_matmul

logger = logging.getLogger("xformers")


try:
    from triton.ops.blocksparse import matmul as blocksparse_matmul
    from triton.ops.blocksparse import softmax as blocksparse_softmax
except ImportError as e:
    logger.warning(
        "Triton is not available, some optimizations will not be enabled.\n"
        + f"This is just a warning: {e}"
    )
    blocksparse_matmul = None
    blocksparse_softmax = None


def _can_use_triton(a):
    if a.device.type == "cpu":
        return False

    if blocksparse_matmul is None:
        return False

    return True


def _spmm(b, layout, values):
    N, nnz, _, block_size = values.shape
    br = b.reshape(
        b.shape[0], b.shape[1], b.shape[2] // block_size, block_size, b.shape[3]
    )
    # perform matmul on blocks
    h, r, c = layout.nonzero(as_tuple=True)
    temp = values @ br[:, h, c, :]

    linear_idx = h * (b.shape[2] // block_size) + r
    out = torch.zeros(
        N,
        b.shape[1] * layout.shape[-2],
        block_size,
        b.shape[3],
        dtype=b.dtype,
        device=b.device,
    )
    # now aggregate the results of the different blocks
    out.index_add_(1, linear_idx.to(b.device), temp)
    out = out.reshape(N, b.shape[1], -1, b.shape[3])
    return out


def _softmax(layout, values):
    h, r, c = layout.nonzero(as_tuple=True)
    norms = torch.logsumexp(values, dim=-1, keepdim=True)
    linear_idx = h * layout.shape[1] + r

    out_t = torch.zeros(
        norms.shape[0],
        layout.shape[0] * layout.shape[1],
        norms.shape[2],
        norms.shape[3],
        dtype=norms.dtype,
        device=norms.device,
    )
    max_val = norms.max()
    out_t.index_add_(
        1, linear_idx.to(values.device), (norms - max_val).exp()
    ).clamp_min_(1e-24).log_().add_(max_val)
    out = torch.exp(values - out_t[:, linear_idx])
    return out


def _sddmm(a, b, layout):
    block_size = a.shape[-2] // layout.shape[-2]
    a = a.reshape(
        a.shape[0], a.shape[1], a.shape[2] // block_size, block_size, a.shape[3]
    )
    b = b.reshape(
        b.shape[0], b.shape[1], b.shape[2] // block_size, block_size, b.shape[3]
    )

    h, r, c = layout.nonzero(as_tuple=True)

    out = torch.einsum("nhik,nhjk->nhij", a[:, h, r, :, :], b[:, h, c, :, :])
    return out


class BlockSparseTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, values, layout):
        kwargs = {}
        kwargs["device"] = values.device
        kwargs["dtype"] = values.dtype
        kwargs["layout"] = values.layout
        kwargs["requires_grad"] = values.requires_grad
        assert values.ndim == 4
        B, _, block_size, _ = values.shape
        C, h, w = layout.shape
        # TODO validate shape of layout vs values
        shape = (B, C, block_size * h, block_size * w)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, values, layout):
        assert values.shape[-2] == values.shape[-1]
        assert (
            values.device == layout.device
        ), "Both values and layout need to reside on the same device"
        block_size = values.shape[-1]
        # TODO: make this check conditioned on the use of Triton
        assert block_size >= 16, "Minimum block size is 16, for now at least"

        # Pure blocksparse data
        self.__values = values
        self.__layout = layout

        # blocksparse operators for triton
        if blocksparse_matmul:
            self._initialize_triton_ops()
        else:
            self.__sparse_dot_sdd = None
            self.__sparse_dot_dsd = None
            self.__sparse_softmax = None

    def _initialize_triton_ops(self):
        block_size = self.__values.shape[-1]
        self.__sparse_dot_sdd = blocksparse_matmul(
            self.__layout,
            block_size,
            "sdd",
            trans_a=False,
            trans_b=True,
            device=self.__layout.device,
        )
        self.__sparse_dot_dsd = blocksparse_matmul(
            self.__layout,
            block_size,
            "dsd",
            trans_a=False,
            trans_b=False,
            device=self.__layout.device,
        )
        self.__sparse_softmax = blocksparse_softmax(
            self.__layout, block_size, device=self.__layout.device
        )

    def __repr__(self):
        return f"block_sparse_tensor(shape={self.shape}, values={self.__values})"

    def values(self):
        return self.__values

    @classmethod
    def _raw_wrap(cls, values, layout, sparse_dot_sdd, sparse_dot_dsd, sparse_softmax):
        matrix = cls.__new__(cls, values, layout)
        matrix.__values = values
        matrix.__layout = layout
        matrix.__sparse_dot_sdd = sparse_dot_sdd
        matrix.__sparse_dot_dsd = sparse_dot_dsd
        matrix.__sparse_softmax = sparse_softmax
        return matrix

    @classmethod
    def _wrap(cls, values, bmat):
        matrix = cls.__new__(cls, values, bmat.__layout)
        matrix.__values = values
        matrix.__layout = bmat.__layout
        matrix.__sparse_dot_sdd = bmat.__sparse_dot_sdd
        matrix.__sparse_dot_dsd = bmat.__sparse_dot_dsd
        matrix.__sparse_softmax = bmat.__sparse_softmax
        return matrix

    @classmethod
    def _bmm(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and type(arg1) == torch.Tensor):
            return NotImplemented
        if _can_use_triton(arg1):
            res = arg0.__sparse_dot_dsd(arg0.__values, arg1)
        else:
            res = _spmm(arg1, arg0.__layout, arg0.__values)
        return res

    @classmethod
    def _masked_matmul(cls, a, b, mask):
        if not (type(a) == torch.Tensor and type(b) == torch.Tensor):
            return NotImplemented
        b = b.transpose(-2, -1)
        assert b.is_contiguous()
        if _can_use_triton(a):
            res = mask.__sparse_dot_sdd(a, b)
        else:
            res = _sddmm(a, b, mask.__layout)
        return cls._wrap(res, mask)

    @classmethod
    def _softmax(cls, arg0, dim):
        if not (dim == -1 or dim == 2):
            return NotImplemented
        if _can_use_triton(arg0):
            res = arg0.__sparse_softmax(arg0.__values)
        else:
            res = _softmax(arg0.__layout, arg0.__values)
        return cls._wrap(res, arg0)

    @classmethod
    def _to(cls, arg0, device):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        return cls(
            arg0.__values.to(device=device),
            arg0.__layout,
        )

    @classmethod
    def _copy(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        assert arg0.shape == arg1.shape
        av0, av1 = arg0.__values, arg1.__values
        av0.resize_as_(av1).copy_(av1)
        av0, av1 = arg0.__layout, arg1.__layout
        av0.resize_as_(av1).copy_(av1)
        out = cls(arg0.__values, arg0.__layout)
        arg0.__sparse_dot_sdd = out.__sparse_dot_sdd
        arg0.__sparse_dot_dsd = out.__sparse_dot_dsd
        arg0.__sparse_softmax = out.__sparse_softmax
        return arg0

    @classmethod
    def _equal(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and isinstance(arg1, cls)):
            return NotImplemented
        if arg0.shape != arg1.shape:
            return False
        if not torch.equal(arg0.__values, arg1.__values):
            return False
        if not torch.equal(arg0.__layout, arg1.__layout):
            return False
        return True

    @classmethod
    def _to_dense(cls, arg0):
        # out = torch.zeros(arg0.shape, dtype=arg0.dtype, device=arg0.device, requires_grad=arg0.requires_grad)
        out = torch.zeros(arg0.shape, dtype=arg0.dtype, device=arg0.device)
        values = arg0.__values
        layout = arg0.__layout
        block_size = values.shape[-1]
        blocks_i = layout.shape[-2]
        blocks_j = layout.shape[-1]

        out_r = out.reshape(
            arg0.shape[0], arg0.shape[1], blocks_i, block_size, blocks_j, block_size
        )

        for idx, (h, i, j) in enumerate(zip(*layout.nonzero(as_tuple=True))):
            out_r[:, h, i, :, j, :] = values[:, idx, :, :]

        return out

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

        if func == masked_matmul:
            assert len(args) == 3
            return cls._masked_matmul(args[0], args[1], args[2])

        if func in [torch.nn.functional.dropout, torch.dropout, torch.dropout_]:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)

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
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)

        if func == torch.Tensor.__deepcopy__:
            x = args[0]
            memo = args[1]
            return cls._raw_wrap(
                x.__values.__deepcopy__(memo),
                x.__layout.__deepcopy__(memo),
                # x.__sparse_dot_sdd.__deepcopy__(memo),
                # x.__sparse_dot_dsd.__deepcopy__(memo),
                # x.__sparse_softmax.__deepcopy__(memo),
                x.__sparse_dot_sdd,
                x.__sparse_dot_dsd,
                x.__sparse_softmax,
            )

        if func in [torch.Tensor.grad.__get__, torch.Tensor._grad.__get__]:
            assert len(args) == 1
            assert len(kwargs) == 0
            x = args[0]
            return cls._wrap(x.__values.grad, x)

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
