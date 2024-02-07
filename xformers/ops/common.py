# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, List, Type, TypeVar, Union

import torch
from torch.torch_version import TorchVersion
from typing_extensions import Annotated, get_args, get_origin

from .. import _is_triton_available


def get_operator(library: str, name: str):
    def no_such_operator(*args, **kwargs):
        raise RuntimeError(
            f"No such operator {library}::{name} - did you forget to build xformers with `python setup.py develop`?"
        )

    try:
        return getattr(getattr(torch.ops, library), name)
    except (RuntimeError, AttributeError):
        return no_such_operator


def get_xformers_operator(name: str):
    return get_operator("xformers", name)


class BaseOperator:
    OPERATOR: Any
    NAME: str
    OPERATOR_CATEGORY: str

    @classmethod
    def is_available(cls) -> bool:
        # cls.OPERATOR can be either a kernel or a Triton Autotuner object, which doesn't have __name__
        if (
            cls.OPERATOR is None
            or getattr(cls.OPERATOR, "__name__", "") == "no_such_operator"
        ):
            return False
        return True

    @classmethod
    def operator_flop(cls, *inputs) -> int:
        """Calculate number of FLOP given inputs to `OPERATOR`"""
        return -1


OPERATORS_REGISTRY: List[Type[BaseOperator]] = []
FUNC_TO_XFORMERS_OPERATOR: Dict[Any, Type[BaseOperator]] = {}

ClsT = TypeVar("ClsT")


def register_operator(cls: ClsT) -> ClsT:
    global OPERATORS_REGISTRY, FUNC_TO_XFORMERS_OPERATOR
    OPERATORS_REGISTRY.append(cls)  # type: ignore
    FUNC_TO_XFORMERS_OPERATOR[cls.OPERATOR] = cls  # type: ignore
    return cls


# post-2.0, avoids a warning
# (`torch.Tensor.storage` will also be deleted in the future)
_GET_TENSOR_STORAGE = getattr(torch.Tensor, "untyped_storage", None)
if _GET_TENSOR_STORAGE is None:  # pre-2.0, `untyped_storage` didn't exist
    _GET_TENSOR_STORAGE = torch.Tensor.storage


def _get_storage_base(x: torch.Tensor) -> int:
    return _GET_TENSOR_STORAGE(x).data_ptr()  # type: ignore


@dataclass(frozen=True)
class Alias:
    name: str
    write: bool


def make_pytorch_cuda_operator(fn: ClsT) -> ClsT:
    return turn_into_pytorch_op(fn, "CUDA")


def make_pytorch_operator_for_dispatch_key(dispatch_key: str) -> Callable[[ClsT], ClsT]:
    def decorator(fn: ClsT) -> ClsT:
        return turn_into_pytorch_op(fn, dispatch_key)

    return decorator


def turn_into_pytorch_op(fn: ClsT, dispatch_key: str) -> ClsT:
    from .. import get_python_lib

    def render_arg_type(annotation) -> str:
        # Optional[T] is an alias for Union[T, None]
        if get_origin(annotation) is Union:
            inner_types = [
                t for t in get_args(annotation) if t is not type(None)  # noqa: E721
            ]
            if len(inner_types) == 1:
                return f"{render_arg_type(inner_types[0])}?"
        if get_origin(annotation) is list:
            (inner_type,) = get_args(annotation)
            return f"{render_arg_type(inner_type)}[]"
        if get_origin(annotation) is tuple:
            return (
                "("
                + ", ".join([render_arg_type(t) for t in get_args(annotation)])
                + ")"
            )
        if get_origin(annotation) is Annotated:
            inner_type, annotation = get_args(annotation)
            if isinstance(annotation, Alias):
                alias = annotation.name + ("!" if annotation.write else "")
                return f"{render_arg_type(inner_type)}({alias})"
        if annotation is torch.Tensor:
            return "Tensor"
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is torch.dtype:
            return "ScalarType"
        if annotation is torch.distributed.ProcessGroup:
            return "__torch__.torch.classes.c10d.ProcessGroup"
        assert False, f"Unable to parse annotation: `{annotation}`"

    def render_default_value(default):
        if default is inspect.Parameter.empty:
            return ""
        return f" = {default!r}"

    sign = inspect.signature(fn)  # type: ignore
    arguments = [
        f"{render_arg_type(arg.annotation)} {arg.name}{render_default_value(arg.default)}"
        for arg in sign.parameters.values()
    ]
    op_name = fn.__name__  # type: ignore
    definition = f"{op_name}({', '.join(arguments)}) -> {render_arg_type(sign.return_annotation)}"

    def callee(*args, **kwargs):
        ba = sign.bind(*args, **kwargs)
        for name, value in ba.arguments.items():
            if sign.parameters[name].annotation is torch.distributed.ProcessGroup:
                from .._C import unbox_process_group

                ba.arguments[name] = unbox_process_group(value)
        return fn(*ba.args, **ba.kwargs)

    xformers_lib = get_python_lib()
    xformers_lib.define(definition)
    xformers_lib.impl(op_name, callee, dispatch_key)
    dispatcher_impl = getattr(getattr(torch.ops, xformers_lib.ns), op_name)

    @wraps(fn)  # type: ignore[arg-type]
    def caller(*args, **kwargs):
        ba = sign.bind(*args, **kwargs)
        for name, value in ba.arguments.items():
            if sign.parameters[name].annotation is torch.distributed.ProcessGroup:
                from .._C import box_process_group

                ba.arguments[name] = box_process_group(value)
        return dispatcher_impl(*ba.args, **ba.kwargs)

    return caller  # type: ignore


def _has_triton2():
    if not _is_triton_available():
        return False
    import triton

    tv = TorchVersion(triton.__version__)
    return tv >= (2, 1) or tv == (2, 0)


def _has_triton21():
    if not _is_triton_available():
        return False
    import triton

    tv = TorchVersion(triton.__version__)
    return tv >= (2, 1)
