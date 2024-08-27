# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Type, TypeVar

import torch


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
