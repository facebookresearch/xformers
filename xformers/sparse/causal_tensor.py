import torch
from torch.utils._pytree import tree_map

from xformers import _is_triton_available

if _is_triton_available:
    from xformers.triton.softmax import softmax as triton_softmax


class CausalTensor(torch.Tensor):
    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem):
        kwargs = {}
        kwargs["device"] = elem.device
        kwargs["dtype"] = elem.dtype
        kwargs["layout"] = elem.layout
        kwargs["requires_grad"] = elem.requires_grad
        assert torch.__version__ > (1, 10), "CausalTensor requires PyTorch 1.11+"
        assert _is_triton_available, "Triton needs to be available"
        r = torch.Tensor._make_wrapper_subclass(cls, elem.shape, **kwargs)
        r.elem = elem
        return r

    def __repr__(self):
        return f"causal_tensor_wrapper({repr(self.elem)})"

    def to_dense(self):
        return torch.tril(self.elem)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        if func in [torch.Tensor.softmax, torch.nn.functional.softmax, torch.softmax]:
            a = args[0].elem
            dim = kwargs["dim"]
            if not (dim == -1 or dim == a.ndim - 1):
                return NotImplemented
            return cls(triton_softmax(a, mask=None, causal=True))

        def unwrap(e):
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            return cls(e) if isinstance(e, torch.Tensor) else e

        with torch._C.DisableTorchFunction():
            ret = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            # ret = func(*args, **kwargs)
            # TODO: check this
            if func in torch.overrides.get_default_nowrap_functions():
                return ret
            return tree_map(wrap, ret)
            # return torch._tensor._convert(ret, cls)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        return NotImplemented
