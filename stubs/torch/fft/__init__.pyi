# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, TypeVar, overload

from pyre_extensions import TypeVarTuple, Unpack
from torch import Tensor, complex64

DType = TypeVar("DType")

Ts = TypeVarTuple("Ts")

@overload
def fft(
    input: Tensor[DType, Unpack[Ts]],
    n: int = ...,
    dim: int = ...,
    norm: str = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[complex64, Unpack[Ts]]: ...
@overload
def fft2(
    input: Tensor[DType, Unpack[Ts]],
    s: Optional[Tuple[int, ...]] = ...,
    dim: Tuple[int, ...] = ...,
    norm: Optional[str] = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
