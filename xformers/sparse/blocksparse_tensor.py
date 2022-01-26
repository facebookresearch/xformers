import torch
from xformers.ops import masked_matmul

from triton.ops.blocksparse import matmul as blocksparse_matmul
from triton.ops.blocksparse import softmax as blocksparse_softmax


class BlockSparseTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, values, layout):
        kwargs = {}
        kwargs["device"] = values.device
        kwargs["dtype"] = values.dtype
        kwargs["layout"] = values.layout
        kwargs["requires_grad"] = values.requires_grad
        assert values.ndim == 4
        B, C, H, W = values.shape
        h, w = layout.shape[-2:]
        shape = (B, C, H * h, W * w)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, values, layout):
        assert values.shape[-2] == values.shape[-1]
        block_size = values.shape[-1]
        assert block_size >= 16, "Minimum block size is 16, for now at least"

        # Pure blocksparse data
        self.__values = values
        self.__layout = layout

        # blocksparse operators
        self.__sparse_dot_sdd = blocksparse_matmul(
            self.__layout,
            block_size,
            "sdd",
            trans_a=False,
            trans_b=True,
        )
        self.__sparse_dot_dsd = blocksparse_matmul(
            self.__layout,
            block_size,
            "dsd",
            trans_a=False,
            trans_b=False,
        )
        self.__sparse_softmax = blocksparse_softmax(self.__layout, block_size)

    def __repr__(self):
        return f"block_sparse_tensor(shape={self.shape}, values={self.__values})"

    @classmethod
    def _wrap(cls, values, bmat):
        matrix = cls.__new__(cls, values, bmat.__layout)
        matrix.__values = values
        matrix.__layout = layout
        matrix.__sparse_dot_sdd = bmat.__sparse_dot_sdd
        matrix.__sparse_dot_dsd = bmat.__sparse_dot_dsd
        matrix.__sparse_softmax = bmat.__sparse_softmax
        return matrix

    @classmethod
    def _bmm(cls, arg0, arg1):
        if not (isinstance(arg0, cls) and type(arg1) == torch.Tensor):
            return NotImplemented
        res = arg0.__sparse_dot_dsd(arg0.__values, arg1)
        return res


    @classmethod
    def _masked_matmul(cls, a, b, mask):
        if not (type(a) == torch.Tensor and type(b) == torch.Tensor):
            return NotImplemented
        b = b.transpose(-2, -1)
        assert b.is_contiguous()
        res = mask.__sparse_dot_sdd(a, b)
        return cls._wrap(res, mask)

    @classmethod
    def _softmax(cls, arg0, dim):
        if not (dim == -1 or dim == 2):
            return NotImplemented
        res = arg0.__sparse_softmax(arg0.__values)
        return cls._wrap(res, arg0)

    @classmethod
    def _to_dense(cls, arg0):
        return NotImplemented

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in [torch.Tensor.bmm, torch.bmm]:
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

        if func == torch.Tensor.to_dense:
            assert len(args) == 1
            return cls._to_dense(args[0])

        if func == torch.Tensor.detach:
            x = args[0]
            values = x.__values.clone()
            values = func(values, *args[1:], **kwargs)
            return cls._wrap(values, x)

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
