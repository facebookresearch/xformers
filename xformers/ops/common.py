# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, TypeVar

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


OPERATORS_REGISTRY: List[Any] = []

ClsT = TypeVar("ClsT")


def register_operator(cls: ClsT) -> ClsT:
    global OPERATORS_REGISTRY
    OPERATORS_REGISTRY.append(cls)
    return cls


class BaseOperator:
    OPERATOR: Any
    NAME: str
    OPERATOR_CATEGORY: str

    @classmethod
    def info(cls):
        if cls.OPERATOR is None or cls.OPERATOR.__name__ == "no_such_operator":
            return "unavailable"
        return "available"
