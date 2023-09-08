# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os
from typing import Any, Dict, List, Type, TypeVar

import torch
from torch.torch_version import TorchVersion


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
        if cls.OPERATOR is None or cls.OPERATOR.__name__ == "no_such_operator":
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


def make_pytorch_cuda_operator(fn: ClsT) -> ClsT:
    from .. import get_python_lib

    def render_arg_type(annotation) -> str:
        if annotation is torch.Tensor:
            return "Tensor"
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is List[int]:
            return "int[]"
        if annotation is List[torch.Tensor]:
            return "Tensor[]"
        assert False, f"Unable to parse annotation: `{annotation}`"

    sign = inspect.signature(fn)  # type: ignore
    arguments = [
        f"{render_arg_type(arg.annotation)} {arg.name}"
        for arg in sign.parameters.values()
    ]
    op_name = fn.__name__  # type: ignore
    definition = f"{op_name}({', '.join(arguments)}) -> {render_arg_type(sign.return_annotation)}"

    xformers_lib = get_python_lib()
    xformers_lib.define(definition)
    xformers_lib.impl(op_name, fn, "CUDA")
    dispatcher_impl = getattr(getattr(torch.ops, xformers_lib.ns), op_name)

    def wrapper(*args, **kwargs):
        return dispatcher_impl(*args, **kwargs)

    return wrapper  # type: ignore


def _has_a_version_of_triton():
    if os.environ.get("XFORMERS_FORCE_DISABLE_TRITON", "0") == "1":
        return False
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _has_triton2():
    if not _has_a_version_of_triton():
        return False
    import triton

    tv = TorchVersion(triton.__version__)
    return tv >= (2, 1) or tv == (2, 0)


def _has_triton21():
    if not _has_a_version_of_triton():
        return False
    import triton

    tv = TorchVersion(triton.__version__)
    return tv >= (2, 1)
