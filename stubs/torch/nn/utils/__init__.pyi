# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, TypeVar, Union

from pyre_extensions import TypeVarTuple
from torch import Tensor

DType = TypeVar("DType")

Ts = TypeVarTuple("Ts")
_tensor_or_tensors = Union[Tensor, Iterable[Tensor]]

def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = ...,
    error_if_nonfinite: bool = ...,
) -> Tensor: ...
def clip_grad_value_(
    parameters: _tensor_or_tensors,
    clip_value: float,
) -> Tensor: ...
