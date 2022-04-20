# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, TypeVar, overload

import torch

# pyre-ignore[21]: Could not find module `pyre_extensions`. (Spurious error)
from pyre_extensions import Add, TypeVarTuple, Unpack
from torch import Tensor
from typing_extensions import Literal as L

DType = TypeVar("DType")
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
N1 = TypeVar("N1", bound=int)
N2 = TypeVar("N2", bound=int)
N3 = TypeVar("N3", bound=int)
N4 = TypeVar("N4", bound=int)

@overload
def pad(
    input: Tensor[DType, Unpack[Ts], N],
    pad: Tuple[N1, N2],
    mode: str = ...,
    value: float = ...,
) -> Tensor[DType, Unpack[Ts], Add[Add[N, N1], N2]]: ...
@overload
def pad(
    input: Tensor[DType, Unpack[Ts], N, M],
    pad: Tuple[N1, N2, N3, N4],
    mode: str = ...,
    value: float = ...,
) -> Tensor[DType, Unpack[Ts], Add[Add[N, N3], N4], Add[Add[M, N1], N2]]: ...

x: str
