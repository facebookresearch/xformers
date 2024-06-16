# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast

import torch

from .common import BaseOperator, get_operator, get_xformers_operator, register_operator


@register_operator
class SparsifyBothWays(BaseOperator):
    OPERATOR = get_xformers_operator("sparse24_sparsify_both_ways")
    OPERATOR_CATEGORY = "sp24"
    NAME = "sparse24_sparsify_both_ways"


@register_operator
class SparsifyApply(BaseOperator):
    OPERATOR = get_xformers_operator("sparse24_apply")
    OPERATOR_CATEGORY = "sp24"
    NAME = "sparse24_apply"


@register_operator
class SparsifyApplyDenseOutput(BaseOperator):
    OPERATOR = get_xformers_operator("sparse24_apply_dense_output")
    OPERATOR_CATEGORY = "sp24"
    NAME = "sparse24_apply_dense_output"


@register_operator
class Sp24Gemm(BaseOperator):
    OPERATOR = get_xformers_operator("_sparse24_gemm")
    OPERATOR_CATEGORY = "sp24"
    NAME = "_sparse24_gemm"


def _get_cusparselt_lib() -> Optional[str]:
    libs = glob.glob(
        str(Path(torch._C.__file__).parent / "lib" / "libcusparseLt*.so.0")
    )
    if len(libs) != 1:
        return None
    return libs[0]


def _get_cusparselt_torch_version() -> Tuple[int, int, int]:
    """
    Returns the version of the cusparselt.so library that ships with pytorch 2.2+
    """
    lib_path = _get_cusparselt_lib()
    if lib_path is None:
        return (0, 0, 0)
    lib = ctypes.CDLL(lib_path)

    def get_version_part(version_part: int) -> int:
        value = ctypes.c_int()
        ret = lib.cusparseLtGetProperty(version_part, ctypes.byref(value))
        if ret != 0:
            return -1
        return value.value

    return (get_version_part(0), get_version_part(1), get_version_part(2))


_cusplt_version = _get_cusparselt_torch_version()
_cusplt_version_str = ".".join(str(v) for v in _cusplt_version)


@register_operator
class Sp24GemmCusplt(BaseOperator):
    OPERATOR = get_operator("aten", "_cslt_sparse_mm")
    OPERATOR_CATEGORY = "sp24"
    NAME = f"_cslt_sparse_mm@{_cusplt_version_str}"


def _has_cusparseLt() -> bool:
    available = _cusplt_version >= (0, 4, 0)
    if not available:
        return False
    if _cusplt_version < (0, 5, 0):
        # Version 0.5.0 has much better perf because it can fuse the
        # transpose within the GEMM epilogue
        warnings.warn(
            f"You have cusparseLt version {_cusplt_version_str} "
            f"but you get better performance with v0.5.0+ if "
            f"you replace the .so file ({_get_cusparselt_lib()})"
        )

    # Sm90 added in 6.0
    compute_capability = (0, 0)
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability("cuda")
    if _cusplt_version < (6, 0, 0):
        if compute_capability >= (9, 0):
            return False
    return available


def sparse24_pointwise_op(
    func, types, args=(), kwargs=None, allow_sparsify_args_list=()
):
    self = None
    for tensor in args:
        if isinstance(tensor, Sparse24Tensor):
            self = tensor
    assert self is not None
    args_updated = []
    for i, tensor in enumerate(args):
        if isinstance(tensor, torch.Tensor):
            if not isinstance(tensor, Sparse24Tensor):
                if i in allow_sparsify_args_list:
                    tensor = sparsify24_like(tensor, self)
                else:
                    raise ValueError(
                        f"Operation {func.__module__}.{func.__name__} on Sparse24Tensor requires all operands to "
                        f"be Sparse24Tensors, but operand {i} is a {type(tensor)}"
                    )
            if (
                tensor.threads_masks is None
                or self.threads_masks is None
                or tensor.threads_masks.data_ptr() != self.threads_masks.data_ptr()
                or tensor.threads_masks.stride() != self.threads_masks.stride()
            ):
                raise ValueError(
                    f"Operation {func.__module__}.{func.__name__} on Sparse24Tensor requires all operands to be "
                    "Sparse24Tensors with the same sparsity pattern"
                )
        args_updated.append(tensor)
    assert isinstance(
        self, Sparse24TensorCutlass
    ), "Only implemented for CUTLASS tensors"
    return Sparse24TensorCutlass(
        self.shape,
        func(
            *[(x.packed if isinstance(x, Sparse24Tensor) else x) for x in args_updated]
        ),
        self.meta,
        func(
            *[
                (x.packed_t if isinstance(x, Sparse24Tensor) else x)
                for x in args_updated
            ]
        ),
        self.meta_t,
        self.threads_masks,
    )


def sparse24_mm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`Sparse24Tensor` matmul: Broadcasting is not implemented"
        )
    if isinstance(A, Sparse24Tensor):
        return A._mm(B)
    else:
        B_t = B.t()
        assert isinstance(B_t, Sparse24Tensor)
        return B_t._mm(A.t(), prefer_col_major_output=True).t()


def sparse24_addmm(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 3
    bias, A, B = args
    if A.ndim != 2 or B.ndim != 2:
        raise NotImplementedError(
            "`Sparse24Tensor` matmul: Broadcasting is not implemented"
        )
    if bias.ndim != 1:
        raise NotImplementedError(
            f"`Sparse24Tensor` matmul: only bias dim=1 supported. Shape={bias.shape}"
        )
    if isinstance(A, Sparse24Tensor):
        raise NotImplementedError(
            "`Sparse24Tensor` matmul: only operand B of `addmm` can be sparse"
        )
    B_t = B.t()
    assert isinstance(B_t, Sparse24Tensor)
    return B_t._mm(A.t(), bias=bias, prefer_col_major_output=True).t()


def sparse24_linear(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) in [2, 3]
    A, B = args[:2]
    bias = args[2] if len(args) == 3 else None
    if bias is None:
        return A @ B.t()
    return sparse24_addmm(
        func=None,
        types=None,
        args=[bias, A, B.t()],
    )


def sparse24_t(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    assert isinstance(self, Sparse24Tensor)
    assert len(self.shape) == 2
    return self.__class__(
        (self.shape[-1], self.shape[0]),
        packed=self.packed_t,
        meta=self.meta_t,
        packed_t=self.packed,
        meta_t=self.meta,
        threads_masks=self.threads_masks.transpose(0, 1),
    )


def sparse24_view(func, types, args=(), kwargs=None) -> torch.Tensor:
    assert len(args) == 2
    self, shape = args
    if tuple(shape) != self.shape:
        raise NotImplementedError(
            f"`view` is not implemented for Sparse24Tensor, except for the dummy case (shape={shape})"
        )
    return self


def sparse24_detach(func, types, args, kwargs) -> torch.Tensor:
    assert len(args) == 1
    self = args[0]
    return self.__class__(
        shape=self.shape,
        packed=self.packed,
        meta=self.meta,
        packed_t=self.packed_t,
        meta_t=self.meta_t,
        threads_masks=self.threads_masks,
        requires_grad=False,
    )


@contextlib.contextmanager
def no_dispatch():
    guard = torch._C._DisableTorchDispatch()
    try:
        yield
    finally:
        del guard


def fallback_dispatcher(func, types, args, kwargs):
    with no_dispatch():
        return func(*args)


SPARSE24_DISPATCH_CUTLASS = {
    torch.ops.aten.is_same_size: fallback_dispatcher,
    torch.ops.aten.detach_: fallback_dispatcher,
    torch.ops.aten.detach: sparse24_detach,
    torch.ops.aten.relu: sparse24_pointwise_op,
    torch.ops.aten.gelu: sparse24_pointwise_op,
    torch.ops.aten.silu: sparse24_pointwise_op,
    torch.ops.aten.mul: partial(
        # `mul` BW in swiglu
        sparse24_pointwise_op,
        allow_sparsify_args_list=(
            0,
            1,
        ),
    ),
    torch.ops.aten.add: sparse24_pointwise_op,
    # Note: for these ops, we allow the gradient to come in as a `torch.Tensor`
    # and we will run the sparsification right before calling the BW aten func
    torch.ops.aten.gelu_backward: partial(
        sparse24_pointwise_op, allow_sparsify_args_list=(0,)
    ),
    torch.ops.aten.silu_backward: partial(
        sparse24_pointwise_op, allow_sparsify_args_list=(0, 1)
    ),
    torch.ops.aten.threshold_backward: partial(  # relu BW
        sparse24_pointwise_op,
        allow_sparsify_args_list=(0,),
    ),
    torch.ops.aten.mm: sparse24_mm,
    torch.ops.aten.matmul: sparse24_mm,
    torch.ops.aten.t: sparse24_t,
    torch.ops.aten.view: sparse24_view,
    torch.ops.aten.linear: sparse24_linear,
}

SPARSE24_DISPATCH_CUSPARSELT = {
    torch.ops.aten.is_same_size: fallback_dispatcher,
    torch.ops.aten.detach_: fallback_dispatcher,
    torch.ops.aten.detach: sparse24_detach,
    torch.ops.aten.t: sparse24_t,
    torch.ops.aten.view: sparse24_view,
    torch.ops.aten.mm: sparse24_mm,
    torch.ops.aten.matmul: sparse24_mm,
    torch.ops.aten.addmm: sparse24_addmm,
    torch.ops.aten.linear: sparse24_linear,
}


class Sparse24Tensor(torch.Tensor):
    packed: torch.Tensor
    meta: torch.Tensor
    packed_t: torch.Tensor
    meta_t: torch.Tensor
    threads_masks: torch.Tensor
    __slots__ = ["packed", "meta", "packed_t", "meta_t", "threads_masks"]

    # We need to update the new method here to tell PyTorch what should be
    # the Tensor corresponding to the wrapper object
    @staticmethod
    def __new__(
        cls,
        shape,
        packed: torch.Tensor,
        meta: torch.Tensor,
        packed_t: torch.Tensor,
        meta_t: torch.Tensor,
        threads_masks: torch.Tensor,
        *,
        requires_grad=False,
    ):
        assert isinstance(packed, torch.Tensor)
        tensor = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            shape,
            device=packed.device,
            dtype=packed.dtype,
            requires_grad=requires_grad,
        )
        tensor.packed = packed
        tensor.meta = meta
        tensor.packed_t = packed_t
        tensor.meta_t = meta_t
        tensor.threads_masks = threads_masks
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def _sp24_to_dense(self) -> torch.Tensor:
        # Multiply by identity
        # WARN: This is not efficient at all
        e = torch.eye(
            self.shape[1], self.shape[1], device=self.device, dtype=self.dtype
        )
        return self @ e

    def _mm(
        self,
        B: torch.Tensor,
        *,
        prefer_col_major_output: bool = False,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    __torch_function__ = torch._C._disabled_torch_function_impl

    def __tensor_flatten__(self):
        return self.__slots__, (self.shape, self.requires_grad)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, flatten_spec):
        shape, requires_grad = flatten_spec
        return cls(
            shape,
            **inner_tensors,
            requires_grad=requires_grad,
        )


class Sparse24TensorCutlass(Sparse24Tensor):
    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        prefer_col_major_output: bool = False,
    ) -> torch.Tensor:
        if isinstance(B, Sparse24Tensor):
            raise ValueError(
                "`Sparse24Tensor @ Sparse24Tensor` is not supported by the hardware"
            )
        if bias is not None:
            raise NotImplementedError(
                f"`Sparse24Tensor` with backend='{BACKEND_CUTLASS}' does not support matmul with bias. "
                f"Remove the bias, or use backend='{BACKEND_CUSPARSELT}'"
            )
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: Broadcasting is not implemented"
            )
        if self.shape[1] != B.shape[0]:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: invalid shapes \
    ({self.shape[0]}, {self.shape[1]}) @ ({B.shape[0]}, {B.shape[1]})"
            )
        return Sp24Gemm.OPERATOR(self.packed, B, self.meta)[: self.shape[0]]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func._overloadpacket not in SPARSE24_DISPATCH_CUTLASS:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__})"
            )
        return SPARSE24_DISPATCH_CUTLASS[func._overloadpacket](
            func, types, args, kwargs
        )


class Sparse24TensorCuSparseLt(Sparse24Tensor):
    def _mm(
        self,
        B: torch.Tensor,
        *,
        prefer_col_major_output: bool = False,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(B, Sparse24Tensor):
            raise ValueError(
                "`Sparse24Tensor @ Sparse24Tensor` is not supported by the hardware"
            )
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: Broadcasting is not implemented"
            )
        if self.shape[1] != B.shape[0]:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: invalid shapes \
    ({self.shape[0]}, {self.shape[1]}) @ ({B.shape[0]}, {B.shape[1]})"
            )
        if B.shape[1] % 8 != 0:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`. "
                "The dense matrix B should have the second dimension aligned to 8."
            )
        if B.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype={self.dtype} and B.dtype={B.dtype}. "
                "This operation is only supported when A and B have the same data type."
            )
        if bias is not None and bias.dtype != self.dtype:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)} + C`, "
                "with A.dtype=B.dtype={self.dtype} and C.dtype={B.dtype}. "
                "This operation is only supported when A, B and C have the same data type."
            )
        assert _has_cusparseLt()
        out = Sp24GemmCusplt.OPERATOR(
            self.packed, B, bias=bias, transpose_result=prefer_col_major_output
        )
        if prefer_col_major_output:
            out = out.t()
        return out[: self.shape[0]]

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if func._overloadpacket not in SPARSE24_DISPATCH_CUSPARSELT:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__})"
            )
        return SPARSE24_DISPATCH_CUSPARSELT[func._overloadpacket](
            func, types, args, kwargs
        )


if torch.__version__ >= "2.1.0":
    torch._dynamo.allow_in_graph(Sparse24TensorCuSparseLt)
    torch._dynamo.allow_in_graph(Sparse24TensorCutlass)

GRADIENT_SP24 = "24sparse"
GRADIENT_DENSE = "24dense"
GRADIENT_STE = "ste"  # Straight-Through Estimator

BACKEND_CUTLASS = "cutlass"
BACKEND_CUSPARSELT = "cusparselt"
BACKEND_DENSE = "dense"


def _sparsify24_forward(x: torch.Tensor, *, algo: str, backend: str) -> Sparse24Tensor:
    assert backend in [
        BACKEND_CUTLASS,
        BACKEND_CUSPARSELT,
    ], f"Invalid backend: {backend}"
    if isinstance(x, Sparse24Tensor):
        if x.threads_masks is None:
            raise ValueError("Input to `sparsify24` is already sparse")
        return x

    (packed, meta, packed_t, meta_t, threads_masks) = SparsifyBothWays.OPERATOR(
        x, algorithm=algo, backend=backend
    )
    cls = (
        Sparse24TensorCutlass
        if backend == BACKEND_CUTLASS
        else Sparse24TensorCuSparseLt
    )
    return cls(
        x.shape,
        packed=packed,
        meta=meta,
        packed_t=packed_t,
        meta_t=meta_t,
        threads_masks=threads_masks,
        requires_grad=False,
    )


class _Sparsify24Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, algo: str, gradient: str, backend: str):  # type: ignore[override]
        if gradient not in [GRADIENT_SP24, GRADIENT_DENSE, GRADIENT_STE]:
            raise ValueError(
                f"Invalid gradient type: '{gradient}'. "
                f"Expected '{GRADIENT_SP24}' or '{GRADIENT_DENSE}' or '{GRADIENT_STE}"
            )
        out = _sparsify24_forward(x, algo=algo, backend=backend)
        ctx.threads_masks = out.threads_masks
        ctx.meta = out.meta
        ctx.meta_t = out.meta_t
        ctx.dtype = out.dtype
        ctx.gradient = gradient
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if isinstance(grad_out, Sparse24Tensor) or ctx.gradient == GRADIENT_STE:
            return grad_out, None, None, None
        assert not isinstance(grad_out, Sparse24Tensor)
        assert grad_out.dtype == ctx.dtype
        if ctx.gradient == GRADIENT_SP24:
            packed, _, packed_t, _ = SparsifyApply.OPERATOR(grad_out, ctx.threads_masks)
            grad_in: torch.Tensor = Sparse24TensorCutlass(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=grad_out.requires_grad,
            )
        elif ctx.gradient == GRADIENT_DENSE:
            assert ctx.threads_masks.is_contiguous()
            grad_in = SparsifyApplyDenseOutput.OPERATOR(grad_out, ctx.threads_masks)
        else:
            assert False, f"Unsupported gradient type: {ctx.gradient}"
        return (
            grad_in,
            None,
            None,
            None,
        )


class _Sparsify24STEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        algo: str,
        backend: str,
        bw_mul0: float,
        bw_mul1: float,
    ):  # type: ignore[override]
        out = _sparsify24_forward(x, algo=algo, backend=backend)
        ctx.threads_masks = out.threads_masks
        ctx.bw_mul0 = bw_mul0
        ctx.bw_mul1 = bw_mul1
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        assert not isinstance(grad_out, Sparse24Tensor)
        if ctx.bw_mul0 == 1.0 and ctx.bw_mul1 == 1.0:
            grad_in = grad_out
        else:
            grad_in = SparsifyApplyDenseOutput.OPERATOR(
                grad_out, ctx.threads_masks, mul0=ctx.bw_mul0, mul1=ctx.bw_mul1
            )
        return (
            grad_in,
            None,  # algo
            None,  # backend
            None,  # bw_mul0
            None,  # bw_mul1
        )


class _Sparsify24LikeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, pattern: Sparse24Tensor, gradient: str, backend: str):  # type: ignore[override]
        if not isinstance(pattern, Sparse24Tensor):
            raise NotImplementedError(
                "`sparsify24_like`: `pattern` must be a sparse tensor"
            )
        if not pattern.threads_masks.is_contiguous():
            raise NotImplementedError(
                "`sparsify24_like` is not implemented when `pattern` is transposed"
            )
        if gradient not in [GRADIENT_DENSE, GRADIENT_SP24, GRADIENT_STE]:
            raise ValueError(f'`sparsify24_like`: invalid gradient type "{gradient}"')
        ctx.threads_masks = pattern.threads_masks
        ctx.meta = pattern.meta
        ctx.meta_t = pattern.meta_t
        ctx.dtype = pattern.dtype
        ctx.gradient = gradient
        if backend == BACKEND_DENSE:
            assert ctx.threads_masks.is_contiguous()
            return SparsifyApplyDenseOutput.OPERATOR(x, ctx.threads_masks)
        packed, meta, packed_t, meta_t = SparsifyApply.OPERATOR(
            x, ctx.threads_masks, backend=backend
        )
        if backend == BACKEND_CUTLASS:
            return Sparse24TensorCutlass(
                x.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=x.requires_grad,
            )
        assert backend == BACKEND_CUSPARSELT, f"Invalid backend: {backend}"
        meta.copy_(pattern.meta)
        meta_t.copy_(pattern.meta_t)
        return Sparse24TensorCuSparseLt(
            x.shape,
            packed,
            meta,
            packed_t,
            meta_t,
            ctx.threads_masks,
            requires_grad=x.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        if ctx.gradient == GRADIENT_STE or isinstance(grad_out, Sparse24Tensor):
            return grad_out, None, None, None
        assert not isinstance(grad_out, Sparse24Tensor)
        assert grad_out.dtype == ctx.dtype

        if ctx.gradient == GRADIENT_DENSE:
            assert ctx.threads_masks.is_contiguous()
            return (
                SparsifyApplyDenseOutput.OPERATOR(grad_out, ctx.threads_masks),
                None,
                None,
                None,
            )
        assert ctx.gradient == GRADIENT_SP24

        packed, _, packed_t, _ = SparsifyApply.OPERATOR(
            grad_out, ctx.threads_masks, backend=BACKEND_CUTLASS
        )
        return (
            Sparse24TensorCutlass(
                grad_out.shape,
                packed,
                ctx.meta,
                packed_t,
                ctx.meta_t,
                ctx.threads_masks,
                requires_grad=grad_out.requires_grad,
            ),
            None,
            None,
            None,
        )


# We want to use `torch._dynamo.allow_in_graph` as a decorator
# (see https://fburl.com/workplace/uimiz0mf) but it breaks mypy.
# This is a hack to work around this
F = TypeVar("F", bound=Callable[..., Any])


def allow_in_graph(func: F) -> F:
    return cast(F, torch._dynamo.allow_in_graph(func))


@allow_in_graph
def sparsify24(
    x: torch.Tensor,
    algo: str = "",
    gradient: str = GRADIENT_SP24,
    backend: str = BACKEND_CUTLASS,
) -> Sparse24Tensor:
    return _Sparsify24Func.apply(x, algo, gradient, backend)


@allow_in_graph
def sparsify24_ste(
    x: torch.Tensor,
    algo: str = "",
    backend: str = BACKEND_CUTLASS,
    bw_mul0: float = 1.0,
    bw_mul1: float = 1.0,
) -> Sparse24Tensor:
    """
    2:4 sparsification, with Straight Through Estimator for the
    backward pass (eg the gradient is *not* sparsified).
    Optionally, `bw_mul[0-1]` provide the option to rescale the gradient
    differently for pruned (`bw_mul0`) and kept values (`bw_mul1`).
    """
    return _Sparsify24STEFunc.apply(x, algo, backend, bw_mul0, bw_mul1)


@allow_in_graph
def sparsify24_like(
    x: torch.Tensor,
    pattern: torch.Tensor,
    gradient: str = GRADIENT_SP24,
    backend: str = "",
    out_dense: Optional[bool] = None,  # <-- TODO: Deprecate this in favor of "gradient"
) -> Sparse24Tensor:
    if out_dense is not None and out_dense:
        backend = BACKEND_DENSE
    if backend == "":
        backend = (
            BACKEND_CUSPARSELT
            if isinstance(pattern, Sparse24TensorCuSparseLt)
            else BACKEND_CUTLASS
        )
    if not isinstance(pattern, Sparse24Tensor):
        raise ValueError(
            f"`pattern` must be a `Sparse24Tensor` but got a {type(pattern)}"
        )
    # Handle transposed case
    if not pattern.threads_masks.is_contiguous():
        return _Sparsify24LikeFunc.apply(x.t(), pattern.t(), gradient, backend).t()
    return _Sparsify24LikeFunc.apply(x, pattern, gradient, backend)
