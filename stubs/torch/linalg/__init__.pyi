# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, TypeVar, overload

from pyre_extensions import TypeVarTuple, Unpack
from torch import Tensor, complex64, complex128, float32, float64
from typing_extensions import Literal as L

DType = TypeVar("DType")
FloatOrDouble = TypeVar("FloatOrDouble", float32, float64, complex64, complex128)
M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)

Ts = TypeVarTuple("Ts")

@overload
def pinv(
    input: Tensor[FloatOrDouble, Unpack[Ts], N1, N1],
    rcond: float = ...,
    *,
    hermitian: L[True],
    out: Optional[Tensor] = ...,
) -> Tensor[FloatOrDouble, Unpack[Ts], N1, N1]: ...
@overload
def pinv(
    input: Tensor[FloatOrDouble, Unpack[Ts], N1, N2],
    rcond: float = ...,
    hermitian: bool = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[FloatOrDouble, Unpack[Ts], N2, N1]: ...
@overload
def qr(
    A: Tensor[FloatOrDouble, Unpack[Ts], M, N],
    mode: L["complete"],
    *,
    out: Optional[Tensor] = ...,
) -> Tuple[
    Tensor[FloatOrDouble, Unpack[Ts], M, M], Tensor[FloatOrDouble, Unpack[Ts], M, N]
]: ...

# The return type should use k=min(m, n), but we don't have that operator yet.
# Assume that M <= N for now.
@overload
def qr(
    A: Tensor[FloatOrDouble, Unpack[Ts], M, N],
    mode: L["reduced"] = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tuple[
    Tensor[FloatOrDouble, Unpack[Ts], M, M], Tensor[FloatOrDouble, Unpack[Ts], M, N]
]: ...
@overload
def norm(
    A: Tensor[DType, Unpack[Ts], N, M], dim: Tuple[L[-2], L[-1]]
) -> Tensor[DType, Unpack[Ts]]: ...
