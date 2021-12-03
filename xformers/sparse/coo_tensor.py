import torch
from torch.utils._pytree import tree_map


from xformers.ops import masked_matmul


def _broadcast_batch(mask, batch_size):
    if mask.ndim == 3:
        return mask
    assert mask.ndim == 2

    mask = mask.coalesce()
    values = mask.values()
    indices = mask.indices()
    nnz = len(values)
    # strategy: repeat the indices and append the extra batch dimension to the indices
    indices = indices.repeat(1, batch_size)
    # now create the batch indices
    batch_indices = torch.arange(batch_size, device=indices.device)
    batch_indices = batch_indices[:, None].expand(batch_size, nnz).flatten()

    # put them together
    indices = torch.cat([batch_indices[None, :], indices], dim=0)

    # now repeat the values
    values = values.repeat(batch_size)

    size = (batch_size,) + mask.shape

    return torch.sparse_coo_tensor(indices, values, size)


class SparseBMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        a = a.coalesce()
        r = torch.bmm(a, b)
        ctx.save_for_backward(a, b)
        return r

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors

        # gradients w.r.t. a
        ga = None
        if ctx.needs_input_grad[0]:
            ga = torch.ops.xformers.matmul_with_mask(grad, b.transpose(-2, -1), a)

        # gradients w.r.t. b
        gb = None
        if ctx.needs_input_grad[1]:
            gb = a.transpose(1, 2).bmm(grad)

        return ga, gb


class SparseCOOTensor(torch.Tensor):
    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem):
        kwargs = {}
        kwargs["device"] = elem.device
        kwargs["dtype"] = elem.dtype
        kwargs["layout"] = elem.layout
        kwargs["requires_grad"] = elem.requires_grad
        assert torch.__version__ > (1, 10), "SparseCSRTensor requires PyTorch 1.11+"
        r = torch.Tensor._make_wrapper_subclass(cls, elem.shape, **kwargs)
        r.elem = elem
        return r

    def __repr__(self):
        return f"sparse_coo_tensor_wrapper({repr(self.elem)})"

    @classmethod
    def _masked_matmul(cls, a, b, _mask):
        if not (type(a) == torch.Tensor and type(b) == torch.Tensor):
            return NotImplemented

        mask = _mask.elem
        assert mask.shape[1] == a.shape[1]
        assert mask.shape[2] == b.shape[2]

        mask = _broadcast_batch(mask, a.shape[0])

        # coalesced is not implemented for bool tensors, so need to cast
        mask = mask.to(dtype=a.dtype)  # type: ignore  # mypy is missing the catch above

        return cls(torch.ops.xformers.matmul_with_mask(a, b, mask))

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
            a, b = args[0], args[1]
            assert a.layout == torch.sparse_coo
            assert b.layout == torch.strided
            assert a.ndim == b.ndim == 3
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[1]
            return SparseBMM.apply(a.elem, b)

        if func in [torch.Tensor.softmax, torch.nn.functional.softmax, torch.softmax]:
            dim = kwargs["dim"]
            if dim < 0:
                dim += args[0].elem.ndim
            return cls(torch.sparse.softmax(args[0].elem, dim))

        if func == masked_matmul:
            assert len(args) == 3
            return cls._masked_matmul(args[0], args[1], args[2])

        if func in [torch.nn.functional.dropout, torch.dropout, torch.dropout_]:
            x = args[0].elem
            x = x.coalesce()
            values = x.values().clone()  # protect against in-place dropout
            values = func(values, *args[1:], **kwargs)
            res = torch.sparse_coo_tensor(x.indices(), values, x.shape)
            return cls(res)

        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e

        with torch._C.DisableTorchFunction():
            # ret = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
            ret = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            # ret = func(*args, **kwargs)
            # TODO: check this
            if func in torch.overrides.get_default_nowrap_functions():
                return ret
            if isinstance(ret, torch.Tensor) and ret.layout == torch.strided:
                return ret
            return tree_map(wrap, ret)
            # return torch._tensor._convert(ret, cls)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        return NotImplemented
