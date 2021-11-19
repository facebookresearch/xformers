# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import builtins
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from numpy import ndarray
from pyre_extensions import (
    Add,
    Broadcast,
    Divide,
    Multiply,
    Product,
    TypeVarTuple,
    Unpack,
)
from typing_extensions import Literal as L

from . import nn as nn
from . import sparse as sparse
from .autograd import *
from .random import initial_seed, set_rng_state

DType = TypeVar("DType")
DType2 = TypeVar("DType2")
Layout = TypeVar("Layout")
Wild = TypeVar("Wild")

T = TypeVar("T")
Ts = TypeVarTuple("Ts")
Rs = TypeVarTuple("Rs")
Rs2 = TypeVarTuple("Rs2")
Qs = TypeVarTuple("Qs")
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
B = TypeVar("B", bound=int)
P = TypeVar("P", bound=int)
R = TypeVar("R", bound=int)
N1 = TypeVar("N1", bound=int)
N2 = TypeVar("N2", bound=int)
N3 = TypeVar("N3", bound=int)
N4 = TypeVar("N4", bound=int)
N5 = TypeVar("N5", bound=int)
N6 = TypeVar("N6", bound=int)

builtin_bool = builtins.bool
builtin_float = builtins.float

# These are torch's datatypes, which have the same names as the builtins.
class complex64: ...
class complex128: ...
class float16: ...
class float32: ...
class float64: ...
class int64: ...
class int32: ...
class bool: ...
class memory_format: ...

int = int32
float = float32
double = float64

class long: ...
class layout: ...

strided: layout = ...

Number = Union[builtins.int, builtins.float, builtins.bool]

class MaxNamedTuple(Generic[DType, Unpack[Ts]]):
    values: Tensor[DType, Unpack[Ts]]
    indices: Tensor[int64, Unpack[Ts]]
    @overload
    def __getitem__(self, key: L[0]) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __getitem__(self, key: L[1]) -> Tensor[int64, Unpack[Ts]]: ...

class device:
    def __init__(self, device_str: str): ...

_just_device = device
_device = Union[device, str]

class Size(Tuple[builtins.int, ...]):
    @overload
    def __getitem__(self: Size, key: builtins.int) -> builtins.int: ...
    @overload
    def __getitem__(self: Size, key: slice) -> Size: ...
    def numel(self: Size) -> builtins.int: ...

class Generator(object):
    device: _device
    def __init__(self, device: Union[_device, str, None] = None) -> None: ...
    def get_state(self) -> Tensor: ...
    def set_state(self, _new_state: Tensor) -> Generator: ...
    def manual_seed(self, seed: builtins.int) -> Generator: ...
    def seed(self) -> builtins.int: ...
    def initial_seed(self) -> builtins.int: ...

default_generator: Generator = ...

class Storage(object):
    _cdata: int
    def __deepcopy__(self, memo) -> "Storage": ...
    def _new_shared(self, int) -> "Storage": ...
    def _write_file(
        self, f: Any, is_real_file: builtins.bool, save_size: builtins.bool
    ) -> None: ...
    def element_size(self) -> int: ...
    def is_shared(self) -> bool: ...
    def share_memory_(self) -> "Storage": ...
    def size(self) -> int: ...

class Tensor(Generic[DType, Unpack[Ts]]):
    requires_grad: builtins.bool
    data: Tensor[DType, Unpack[Ts]]
    names: List[str]
    layout: layout
    T: Tensor[DType, Unpack[Ts]]
    output_nr: builtins.int
    _version: builtins.int
    _base: Optional[Tensor]
    _cdata: builtins.int
    grad_fn: Any
    grad: Optional[Tensor]
    _grad_fn: Any
    _grad: Optional[Tensor]
    _backward_hooks: Optional[Dict[builtins.int, Callable[[Tensor], Optional[Tensor]]]]
    @overload
    def __init__(self, other: Tensor[DType, Unpack[Ts]]) -> None: ...
    @overload
    def __init__(
        self, *args: Unpack[Ts], device: Union[_device, str, None] = ...
    ) -> None: ...
    @overload
    def __init__(self, storage: Storage) -> None: ...
    @overload
    def __init__(
        self, size: Tuple[Unpack[Ts]], *, device: Union[_device, str, None] = ...
    ) -> None: ...
    @property
    def device(self) -> _device: ...
    @property
    def dtype(self) -> Type[DType]: ...
    def long(self) -> "LongTensor[DType, Unpack[Ts]]": ...
    # BEWARE: The type for self must not reuse `Ts`. This is because the type
    # of the object is `Tensor[DType, Unpack[Ts]]`.
    # We are trying to match part of it by using fresh type variables N1 and
    # Rs: `self: Tensor[DType, N1, Unpack[Rs]]`.
    # If we used Ts, then `Ts` would be the one from the object type. We would
    # be saying that the object type `Tensor[DType, Unpack[Ts]]` must match
    # `Tensor[DType, N1, Unpack[Ts]]`, which is absurd.
    @overload
    def size(self: Tensor[DType, N1, Unpack[Rs]], axis: L[0]) -> N1: ...
    @overload
    def size(self: Tensor[DType, N1, N2, Unpack[Rs]], axis: L[1]) -> N2: ...
    @overload
    def size(self: Tensor[DType, Unpack[Rs], N1], axis: L[-1]) -> N1: ...
    @overload
    def size(self: Tensor[DType, Unpack[Rs], N1, N2], axis: L[-2]) -> N1: ...
    @overload
    def size(self: Tensor[DType, Unpack[Rs]]) -> Tuple[Unpack[Rs]]: ...
    @overload
    def split(
        self: Tensor[DType, N1, Unpack[Rs]], split_size_or_sections: N, dim: L[0] = ...
    ) -> Iterable[Tensor[DType, N, Unpack[Rs]]]: ...
    @overload
    def split(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        split_size_or_sections: N,
        dim: L[1] = ...,
    ) -> Iterable[Tensor[DType, N1, N, Unpack[Rs]]]: ...
    @overload
    def item(self: Tensor[DType, L[1]]) -> DType: ...
    @overload
    def item(self: Tensor[DType]) -> DType: ...
    def numel(self) -> builtins.int: ...
    def backward(self) -> None: ...
    @overload
    def __getitem__(
        self: Tensor[DType, N, Unpack[Rs]], item: L[0]
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def __getitem__(
        self: Tensor[DType, Unpack[Rs]], item: None
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def __getitem__(
        self: Tensor[DType, Unpack[Rs]], item: Tensor[bool, Unpack[Rs]]
    ) -> Tensor[DType, builtins.int]: ...
    @overload
    def __getitem__(self, item: Any) -> Any: ...
    @overload
    def expand(
        self: Tensor[DType, Unpack[Rs]], sizes: Tuple[Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def expand(
        self: Tensor[DType, Unpack[Rs]], *sizes: Unpack[Rs2]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    def detach(self: T) -> T: ...
    # pyre-ignore[24]: Pyre is unable to find the custom stubs for numpy.
    def numpy(self) -> ndarray[DType, Unpack[Ts]]: ...
    shape: Tuple[Unpack[Ts]]
    ndim: builtins.int
    @overload
    def to(
        self: Tensor[DType, Unpack[Rs]], dtype: Type[T], device: _device = ...
    ) -> Tensor[T, Unpack[Rs]]: ...
    @overload
    def to(
        self: Tensor[DType, Unpack[Rs]], device: _device
    ) -> Tensor[DType, Unpack[Rs]]: ...
    device: _just_device
    @overload
    def __add__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __add__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __iadd__(
        self, other: Tensor[DType, Unpack[Rs]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __iadd__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __radd__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __radd__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __sub__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __sub__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __isub__(
        self, other: Tensor[DType, Unpack[Rs]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __isub__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __rsub__(
        self: Tensor[DType, Unpack[Rs]], other: Tensor[DType, Unpack[Rs2]]
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __rsub__(
        self: Tensor[DType, Unpack[Rs]],
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def __mul__(
        self: Tensor[DType, Unpack[Rs]],
        other: Tensor[DType, Unpack[Rs2]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __mul__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __imul__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __imul__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __rmul__(
        self: Tensor[DType, Unpack[Rs]],
        other: Tensor[DType, Unpack[Rs2]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __rmul__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __pow__(
        self: Tensor[DType, Unpack[Rs]],
        other: Tensor[DType, Unpack[Rs2]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __pow__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __truediv__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __truediv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[float32, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __itruediv__(
        self,
        other: builtin_float,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __itruediv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __rtruediv__(
        self,
        other: builtin_float,
    ) -> Tensor[float32, Unpack[Ts]]: ...
    @overload
    def __floordiv__(
        self,
        other: builtins.int,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __floordiv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __ifloordiv__(
        self,
        other: builtins.int,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __ifloordiv__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __rfloordiv__(
        self,
        other: builtins.int,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    def __invert__(self) -> Tensor[DType, Unpack[Ts]]: ...
    def __neg__(self) -> Tensor[DType, Unpack[Ts]]: ...
    def __iand__(
        self: Tensor[bool, Unpack[Rs]],
        other: Tensor[bool, Unpack[Rs2]],
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    def __and__(
        self: Tensor[bool, Unpack[Rs]],
        other: Tensor[bool, Unpack[Rs2]],
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Rs2]]]]]: ...
    @overload
    def __matmul__(
        self: Tensor[DType, N1],
        other: Tensor[DType, N1],
    ) -> Tensor[DType]: ...
    @overload
    def __matmul__(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        other: Tensor[DType, Unpack[Qs], N2, N3],
    ) -> Tensor[
        DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3
    ]: ...
    def __ne__(
        self: Tensor[DType, Unpack[Rs]], other: DType
    ) -> Tensor[bool, Unpack[Rs]]: ...
    def abs(self) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def all(
        self: Tensor[DType, Unpack[Ts]],
    ) -> Tensor[bool, L[1]]: ...
    @overload
    def all(
        self: Tensor[DType, N, Unpack[Ts]],
        dim: L[0],
    ) -> Tensor[bool, Unpack[Ts]]: ...
    @overload
    def all(
        self: Tensor[DType, N1, N2, Unpack[Ts]],
        dim: L[1],
    ) -> Tensor[bool, N1, Unpack[Ts]]: ...
    @overload
    def argmax(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> LongTensor[int64, L[1], Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> LongTensor[int64, N1, L[1], Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, N1, Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> LongTensor[int64, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, N1, N2, Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> LongTensor[int64, Unpack[Rs], L[1]]: ...
    @overload
    def argmax(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, Unpack[Rs]]: ...
    @overload
    def argmax(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> LongTensor[int64]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> LongTensor[int64, L[1], Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> LongTensor[int64, N1, L[1], Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, N1, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> LongTensor[int64, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, N1, N2, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> LongTensor[int64, Unpack[Rs], L[1]]: ...
    @overload
    def argmin(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> LongTensor[int64, Unpack[Rs]]: ...
    @overload
    def argmin(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> LongTensor[int64]: ...
    # Note: Not defining this as a method `def bool()` because that confuses
    # Pyre in other method signatures that use `torch.bool`. Not sure why.
    bool: Callable[[], Tensor[bool, Unpack[Ts]]] = ...
    @overload
    def chunk(
        self: Tensor[DType, Unpack[Rs], N], chunks: L[2], dim: L[-1]
    ) -> Tuple[
        Tensor[DType, Unpack[Rs], Divide[N, L[2]]],
        Tensor[DType, Unpack[Rs], Divide[N, L[2]]],
    ]: ...
    @overload
    def chunk(
        self: Tensor[DType, N, Unpack[Rs]], chunks: L[2], dim: L[0] = ...
    ) -> Tuple[
        Tensor[DType, Divide[N, L[2]], Unpack[Rs]],
        Tensor[DType, Divide[N, L[2]], Unpack[Rs]],
    ]: ...
    def clone(
        input, *, memory_format: Optional[memory_format] = ...
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
    ) -> Tensor[int64, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
    ) -> Tensor[int64, N1, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
    ) -> Tensor[int64, N1, N2, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
    ) -> Tensor[int64, Unpack[Rs]]: ...
    @overload
    def count_nonzero(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> Tensor[int64]: ...
    @overload
    def dim(self: Tensor[DType]) -> L[0]: ...
    @overload
    def dim(self: Tensor[DType, builtins.int]) -> L[1]: ...
    @overload
    def dim(self: Tensor[DType, builtins.int, builtins.int]) -> L[2]: ...
    @overload
    def dim(self: Tensor[DType, builtins.int, builtins.int, builtins.int]) -> L[3]: ...
    def half(
        self: Tensor[DType, Unpack[Rs]], memory_format: Optional[memory_format] = ...
    ) -> Tensor[float16, Unpack[Rs]]: ...
    def is_contiguous(
        self, memory_format: Optional[memory_format] = ...
    ) -> builtins.bool: ...
    def indices(self) -> Tensor: ...
    is_cuda: builtins.bool
    def masked_select(self, mask: Tensor, *, out: Optional[Tensor] = ...) -> Tensor: ...
    @overload
    def max(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> MaxNamedTuple[DType, L[1], Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> MaxNamedTuple[DType, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> MaxNamedTuple[DType, N1, L[1], Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> MaxNamedTuple[DType, N1, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> MaxNamedTuple[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> MaxNamedTuple[DType, N1, N2, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> MaxNamedTuple[DType, Unpack[Rs], L[1]]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> MaxNamedTuple[DType, Unpack[Rs]]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        dim: L[-2],
        keepdim: L[True],
    ) -> MaxNamedTuple[DType, Unpack[Rs], L[1], N2]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        dim: L[-2],
        keepdim: L[False] = ...,
    ) -> MaxNamedTuple[DType, Unpack[Rs], N2]: ...
    @overload
    def max(
        self: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[DType]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[True],
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[True],
    ) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, N1, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[True],
    ) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        dim: L[2],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[True],
    ) -> Tensor[DType, Unpack[Rs], L[1]]: ...
    @overload
    def mean(
        self: Tensor[DType, Unpack[Rs], N1],
        dim: L[-1],
        keepdim: L[False] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def mean(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        keepdim: builtins.bool = ...,
    ) -> Tensor[DType]: ...
    def bitwise_not(self) -> Tensor[DType, Unpack[Ts]]: ...
    def bitwise_not_(self) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def diff(
        self: Tensor[DType, Unpack[Rs], Add[N1, L[1]], N2],
        dim: L[-2],
    ) -> Tensor[DType, Unpack[Rs], N1, N2]: ...
    @overload
    def diff(
        self: Tensor[DType, Add[N, L[1]], Unpack[Rs]],
        dim: L[0],
    ) -> Tensor[DType, N, Unpack[Rs]]: ...
    @overload
    def diff(
        self: Tensor[DType, N1, Add[N2, L[1]], Unpack[Rs]],
        dim: L[1],
    ) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
    @overload
    def diff(
        self: Tensor[DType, Unpack[Rs], Add[N, L[1]]],
        dim: L[-1] = ...,
    ) -> Tensor[DType, Unpack[Rs], N]: ...
    def is_sparse(self) -> builtins.bool: ...
    def coalesce(self: Tensor[DType, Unpack[Rs]]) -> Tensor[DType, Unpack[Rs]]: ...
    def values(self: Tensor[DType, Unpack[Rs]]) -> Tensor[DType, Unpack[Rs]]: ...
    def to_sparse(self: Tensor[DType, Unpack[Ts]]) -> Tensor[DType, Unpack[Ts]]: ...
    # Note: Not defining this as a method `def float()` because that confuses
    # Pyre in other method signatures that use `torch.float`. Not sure why.
    float: Callable[[], Tensor[float32, Unpack[Ts]]] = ...
    @overload
    def __eq__(
        self,
        other: Tensor[DType, Unpack[Rs]],
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def __eq__(
        self,
        other: object,
    ) -> Tensor[bool, Unpack[Ts]]: ...
    def argsort(
        self, dim: builtins.int = ..., descending: builtin_bool = ...
    ) -> Tensor[DType, Unpack[Ts]]: ...
    def bmm(
        self: Tensor[DType, B, N, M], mat2: Tensor[DType, B, M, P]
    ) -> Tensor[DType, B, N, P]: ...
    def diag_embed(
        self: Tensor[DType, Unpack[Rs], N]
    ) -> Tensor[DType, Unpack[Rs], N, N]: ...
    @overload
    def matmul(
        self: Tensor[DType, N1],
        other: Tensor[DType, N1],
    ) -> Tensor[DType]: ...
    @overload
    def matmul(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        other: Tensor[DType, Unpack[Qs], N2, N3],
    ) -> Tensor[
        DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3
    ]: ...
    def multinomial(
        self: Tensor[DType, Unpack[Rs], N1],
        num_samples: N2,
        replacement: builtins.bool = ...,
        *,
        generator: Optional[Generator] = ...,
    ) -> Tensor[DType, Unpack[Rs], N2]: ...
    @overload
    def new_ones(
        self,
        size: Tuple[Unpack[Rs]],
        dtype: Type[DType2],
        device: _device = ...,
        requires_grad: builtins.bool = ...,
    ) -> Tensor[DType2, Unpack[Rs]]: ...
    @overload
    def new_ones(
        self,
        size: Tuple[Unpack[Rs]],
        dtype: Type[DType] = ...,
        device: _device = ...,
        requires_grad: builtins.bool = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, Unpack[Rs]], dim: L[-1]
    ) -> Tensor[DType, Unpack[Rs], L[1]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, Unpack[Rs]], dim: L[0]
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, N, Unpack[Rs]], dim: L[1]
    ) -> Tensor[DType, N, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim: L[2]
    ) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze_(
        self: Tensor[DType, Unpack[Rs]], dim: L[-1]
    ) -> Tensor[DType, Unpack[Rs], L[1]]: ...
    @overload
    def unsqueeze_(
        self: Tensor[DType, Unpack[Rs]], dim: L[0]
    ) -> Tensor[DType, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze_(
        self: Tensor[DType, N, Unpack[Rs]], dim: L[1]
    ) -> Tensor[DType, N, L[1], Unpack[Rs]]: ...
    @overload
    def unsqueeze_(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim: L[2]
    ) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
    @property
    def real(self: Tensor[complex64, Unpack[Rs]]) -> Tensor[float32, Unpack[Rs]]: ...
    @overload
    def repeat(
        self: Tensor[DType, N1], size1: N2
    ) -> Tensor[DType, Multiply[N1, N2]]: ...
    @overload
    def repeat(
        self: Tensor[DType, N1, N2], size1: N3, size2: N4
    ) -> Tensor[DType, Multiply[N1, N3], Multiply[N2, N4]]: ...
    @overload
    def repeat(
        self: Tensor[DType, N1, N2, N3], size1: N4, size2: N5, size3: N6
    ) -> Tensor[DType, Multiply[N1, N4], Multiply[N2, N5], Multiply[N3, N6]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, Unpack[Rs], N1], repeats: N, dim: L[-1]
    ) -> Tensor[DType, Unpack[Rs], Multiply[N1, N]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, N1, Unpack[Rs]], repeats: N, dim: L[0]
    ) -> Tensor[DType, Multiply[N1, N], Unpack[Rs]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, N1, N2, Unpack[Rs]], repeats: N, dim: L[1]
    ) -> Tensor[DType, N1, Multiply[N2, N], Unpack[Rs]]: ...
    @overload
    def repeat_interleave(
        self: Tensor[DType, Unpack[Rs]], repeats: N, dim: L[None] = ...
    ) -> Tensor[DType, Product[N, Unpack[Rs]]]: ...
    # The output shape here depends on the contents of `repeats`, so give up.
    @overload
    def repeat_interleave(
        input: Tensor[DType, Unpack[Rs]], repeats: Tensor, dim: builtins.int = ...
    ) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
    def __setitem__(self, item: object, other: object) -> None: ...
    @overload
    def scatter(
        self,
        dim: builtins.int,
        index: Tensor,
        src: Union[Tensor, float],
        reduce: Optional[str] = ...,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def scatter_(
        self,
        dim: builtins.int,
        index: Tensor,
        src: Union[Tensor, float],
        reduce: Optional[str] = ...,
    ) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def softmax(self, dim: builtins.int) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def softmax(
        self, dim: builtins.int, dtype: Type[DType2]
    ) -> Tensor[DType2, Unpack[Ts]]: ...
    @overload
    def stride(
        self: Tensor[DType, builtins.int, Unpack[Rs]], dim: L[0]
    ) -> Product[Unpack[Rs]]: ...
    @overload
    def stride(
        self: Tensor[DType, builtins.int, builtins.int, Unpack[Rs]], dim: L[1]
    ) -> Product[Unpack[Rs]]: ...
    @overload
    def stride(
        self: Tensor[DType, builtins.int, builtins.int, builtins.int], dim: L[2]
    ) -> L[1]: ...
    @overload
    def stride(self) -> Tuple[Unpack[Ts]]: ...
    @overload
    def squeeze(
        self: Tensor[DType, Unpack[Rs], L[1], L[1]], *, out: Optional[Tensor] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze(
        self: Tensor[DType, L[1], L[1], Unpack[Rs]], *, out: Optional[Tensor] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze(
        self: Tensor[DType, L[1], Unpack[Rs]],
        dim: L[0] = ...,
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze(
        self: Tensor[DType, Unpack[Rs], L[1]],
        dim: L[-1] = ...,
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze(
        self: Tensor[DType, Unpack[Rs]], *, out: Optional[Tensor] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    def type_as(
        self, other: Tensor[DType2, Unpack[Rs]]
    ) -> Tensor[DType2, Unpack[Rs]]: ...
    @overload
    def squeeze_(
        self: Tensor[DType, Unpack[Rs], L[1], L[1]], *, out: Optional[Tensor] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze_(
        self: Tensor[DType, L[1], L[1], Unpack[Rs]], *, out: Optional[Tensor] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze_(
        self: Tensor[DType, L[1], Unpack[Rs]],
        dim: L[0] = ...,
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze_(
        self: Tensor[DType, Unpack[Rs], L[1]],
        dim: L[-1] = ...,
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def squeeze_(
        self: Tensor[DType, Unpack[Rs]], *, out: Optional[Tensor] = ...
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def view(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def view(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[N1, L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, N1, Divide[Product[Unpack[Rs]], Product[N1, Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def view(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[Unpack[Rs2], L[-1]]]
    ) -> Tensor[
        DType, Unpack[Rs2], Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]]
    ]: ...
    @overload
    def view(self, *shape: Unpack[Rs]) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def transpose(
        self: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-2], dim1: L[-1]
    ) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
    @overload
    def transpose(
        self: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-1], dim1: L[-2]
    ) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
    @overload
    def transpose(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[0], dim1: L[1]
    ) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
    @overload
    def transpose(
        self: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[1], dim1: L[0]
    ) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
    @overload
    def transpose(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]], dim0: L[1], dim1: L[2]
    ) -> Tensor[DType, N1, N3, N2, Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, Unpack[Rs], N2],
        start_dim: L[0] = ...,
        end_dim: L[-1] = ...,
    ) -> Tensor[DType, Product[N1, Unpack[Rs], N2]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        start_dim: L[0] = ...,
        end_dim: L[1] = ...,
    ) -> Tensor[DType, Multiply[N1, N2], Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, N2, N3, Unpack[Rs]],
        start_dim: L[1] = ...,
        end_dim: L[2] = ...,
    ) -> Tensor[DType, N1, Multiply[N2, N3], Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType, N1, N2, N3, N4, Unpack[Rs]],
        start_dim: L[2] = ...,
        end_dim: L[3] = ...,
    ) -> Tensor[DType, N1, N2, Multiply[N3, N4], Unpack[Rs]]: ...
    @overload
    def flatten(
        self: Tensor[DType],
        start_dim: L[0] = ...,
        end_dim: L[0] = ...,
    ) -> Tensor[DType, L[1]]: ...
    @overload
    def __lt__(
        self: Tensor[DType, Unpack[Rs]], x: DType
    ) -> Tensor[bool, Unpack[Rs]]: ...
    @overload
    def __lt__(
        self: Tensor[float32, Unpack[Rs]], x: builtin_float
    ) -> Tensor[bool, Unpack[Rs]]: ...
    @overload
    def __gt__(
        self: Tensor[DType, Unpack[Rs]], x: DType
    ) -> Tensor[bool, Unpack[Rs]]: ...
    @overload
    def __gt__(
        self: Tensor[float32, Unpack[Rs]], x: builtin_float
    ) -> Tensor[bool, Unpack[Rs]]: ...
    def logical_and(
        self,
        other: Tensor[DType2, Unpack[Rs]],
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    def logical_and_(
        self,
        other: Tensor[DType2, Unpack[Rs]],
        *,
        out: Optional[Tensor] = ...,
    ) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
    @overload
    def reshape(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def reshape(
        self: Tensor[DType, Unpack[Rs]], *shape: Unpack[Tuple[N1, L[-1], Unpack[Rs2]]]
    ) -> Tensor[
        DType, N1, Divide[Product[Unpack[Rs]], Product[N1, Unpack[Rs2]]], Unpack[Rs2]
    ]: ...
    @overload
    def reshape(self, *shape: Unpack[Rs]) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def unbind(
        self: Tensor[DType, Unpack[Rs], N], dim: L[-1]
    ) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
    @overload
    def unbind(
        self: Tensor[DType, N, N1, Unpack[Rs]], dim: L[1]
    ) -> Tuple[Tensor[DType, N, Unpack[Rs]], ...]: ...
    @overload
    def unbind(
        self: Tensor[DType, N, Unpack[Rs]], dim: L[0] = ...
    ) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
    def sign(self, *, out: Optional[Tensor] = ...) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def sum(
        self: Tensor[DType, N1, Unpack[Rs]],
        dim: L[0],
        *,
        dtype: Optional[_device] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def sum(
        self: Tensor[DType, N1, N2, Unpack[Rs]],
        dim: L[1],
        *,
        dtype: Optional[_device] = ...,
    ) -> Tensor[DType, N1, Unpack[Rs]]: ...
    @overload
    def sum(
        self: Tensor[DType, Unpack[Rs], N],
        dim: L[-1],
        *,
        dtype: Optional[_device] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    @overload
    def sum(
        self: Tensor[DType, Unpack[Rs], N1, N2],
        dim: L[-2],
        *,
        dtype: Optional[_device] = ...,
    ) -> Tensor[DType, Unpack[Rs], N2]: ...
    @overload
    def sum(
        self: Tensor[DType, Unpack[Rs]],
        dim: L[None] = ...,
        *,
        dtype: Optional[_device] = ...,
    ) -> Tensor[DType]: ...
    def cumsum(
        self: Tensor[DType, Unpack[Rs]],
        dim: builtins.int = ...,
        dtype: Optional[_device] = ...,
    ) -> Tensor[DType, Unpack[Rs]]: ...
    def contiguous(input: Tensor[DType, Unpack[Rs]]) -> Tensor[DType, Unpack[Rs]]: ...

class LongTensor(Tensor[DType, Unpack[Ts]], Generic[DType, Unpack[Ts]]):
    @overload
    def __getitem__(
        self: LongTensor[DType, Unpack[Rs], N], val: Tuple[object, None]
    ) -> LongTensor[DType, Unpack[Tuple[Any, ...]]]: ...
    @overload
    def __getitem__(
        self: LongTensor[DType, Unpack[Rs], N], val: Tuple[None, object]
    ) -> LongTensor[DType, Unpack[Tuple[Any, ...]]]: ...
    @overload
    def __getitem__(
        self: LongTensor[DType, Unpack[Rs], N], val: slice
    ) -> LongTensor[DType, Unpack[Tuple[Any, ...]]]: ...
    def __eq__(
        self: LongTensor[DType, Unpack[Rs]],
        other: LongTensor[DType, Unpack[Rs]],
    ) -> LongTensor[bool, Unpack[Rs]]: ...

# NOTE: These `torch` functions below have a method counterpart in `Tensor`. So,
# if you update the stubs here, please update the method stub as well.

def allclose(
    input: Tensor,
    other: Tensor,
    rtol: builtin_float = ...,
    atol: builtin_float = ...,
    equal_nan: builtins.bool = ...,
) -> builtins.bool: ...
def bitwise_not(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
def einsum(
    equation: str,
    *operands: Tensor,
) -> Tensor: ...
@overload
def eye(
    n: N,
    *,
    dtype: Type[float32] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, N, N]: ...
@overload
def eye(
    n: N,
    m: M,
    *,
    dtype: Type[float32] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, N, M]: ...
@overload
def eye(
    n: N,
    *,
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N, N]: ...
@overload
def eye(
    n: N,
    m: M,
    *,
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N, M]: ...
@overload
def zeros(
    size: Tuple[Unpack[Ts]],
    *,
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def zeros(
    size: Tuple[Unpack[Ts]],
    *,
    dtype: Type[float32] = ...,
    low: builtins.int = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def zeros(
    *size: Unpack[Ts],
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def zeros(
    *size: Unpack[Ts],
    dtype: Type[float32] = ...,
    low: builtins.int = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def ones(*size: Unpack[Ts]) -> Tensor[float, Unpack[Ts]]: ...
@overload
def ones(
    *size: Unpack[Ts], dtype: Type[DType] = ..., device: _device = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def ones_like(
    input: Tensor[DType, Unpack[Ts]],
    *,
    dtype: Type[DType2],
    memory_format: Optional[memory_format] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType2, Unpack[Ts]]: ...
@overload
def ones_like(
    input: Tensor[DType, Unpack[Ts]],
    *,
    memory_format: Optional[memory_format] = ...,
    dtype: Type[DType] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
def tril(
    x: Tensor[DType, Unpack[Ts]], diagonal: builtins.int = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def arange(
    end: N1,
    *,
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[int64, N1]: ...
@overload
def arange(
    start: N1,
    end: N2,
    *,
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[int64, Add[N2, Multiply[L[-1], N1]]]: ...
@overload
def arange(
    start: N1,
    end: N2,
    step: N3,
    out: Optional[int] = ...,
    dtype: Type[int64] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[int64, Divide[Add[N2, Multiply[L[-1], N1]], N3]]: ...

# dtype is explicitly provided.
@overload
def arange(
    end: N1,
    *,
    dtype: Type[DType],
    out: Optional[int] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N1]: ...
@overload
def arange(
    start: N1,
    end: N2,
    *,
    dtype: Type[DType],
    out: Optional[int] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Add[N2, Multiply[L[-1], N1]]]: ...
@overload
def arange(
    start: N1,
    end: N2,
    step: N3,
    dtype: Type[DType],
    out: Optional[int] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Divide[Add[N2, Multiply[L[-1], N1]], N3]]: ...
@overload
def arange(
    end: builtin_float,
    start: builtin_float = ...,
    step: builtin_float = ...,
    out: Optional[int] = ...,
    dtype: Type[DType] = ...,
    layout: Type[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, builtins.str]: ...
@overload
def argmax(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> LongTensor[int64, L[1], Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> LongTensor[int64, Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> LongTensor[int64, N1, L[1], Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> LongTensor[int64, N1, Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> LongTensor[int64, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> LongTensor[int64, N1, N2, Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> LongTensor[int64, Unpack[Rs], L[1]]: ...
@overload
def argmax(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> LongTensor[int64, Unpack[Rs]]: ...
@overload
def argmax(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> LongTensor[int64]: ...
@overload
def argmin(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> LongTensor[int64, L[1], Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> LongTensor[int64, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> LongTensor[int64, N1, L[1], Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> LongTensor[int64, N1, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> LongTensor[int64, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> LongTensor[int64, N1, N2, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> LongTensor[int64, Unpack[Rs], L[1]]: ...
@overload
def argmin(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> LongTensor[int64, Unpack[Rs]]: ...
@overload
def argmin(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> LongTensor[int64]: ...
def bmm(
    input: Tensor[DType, B, N, M], mat2: Tensor[DType, B, M, P]
) -> Tensor[DType, B, N, P]: ...
@overload
def chunk(
    input: Tensor[DType, Unpack[Ts], N], chunks: L[2], dim: L[-1]
) -> Tuple[
    Tensor[DType, Unpack[Ts], Divide[N, L[2]]],
    Tensor[DType, Unpack[Ts], Divide[N, L[2]]],
]: ...
@overload
def chunk(
    input: Tensor[DType, N, Unpack[Ts]], chunks: L[2], dim: L[0] = ...
) -> Tuple[
    Tensor[DType, Divide[N, L[2]], Unpack[Ts]],
    Tensor[DType, Divide[N, L[2]], Unpack[Ts]],
]: ...
def diag(
    input: Tensor[DType, Unpack[Tuple[Any, ...]]],
    diagonal: builtins.int = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
def diagonal(
    input: Tensor[DType, Unpack[Tuple[Any, ...]]],
    offset: builtins.int = ...,
    dim1: builtins.int = ...,
    dim2: builtins.int = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
def diag_embed(
    input: Tensor[DType, Unpack[Tuple[Any, ...]]],
    offset: builtins.int = ...,
    dim1: builtins.int = ...,
    dim2: builtins.int = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
@overload
def empty_like(
    input: Tensor[DType, Unpack[Ts]],
    *,
    dtype: Type[DType2],
    memory_format: Optional[memory_format] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
    out: Optional[Tensor] = ...,
) -> Tensor[DType2, Unpack[Ts]]: ...
@overload
def empty_like(
    input: Tensor[DType, Unpack[Ts]],
    *,
    memory_format: Optional[memory_format] = ...,
    dtype: Type[DType] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
    out: Optional[Tensor] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
def logical_and(
    input: Tensor[DType, Unpack[Ts]],
    other: Tensor[DType2, Unpack[Rs]],
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[bool, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]]]: ...
@overload
def log_softmax(
    input: Tensor[DType, Unpack[Rs]],
    dtype: Type[DType2],
    dim: Optional[builtins.int] = ...,
) -> Tensor[DType2, Unpack[Rs]]: ...
@overload
def log_softmax(
    input: Tensor[DType, Unpack[Rs]],
    *,
    dim: Optional[builtins.int] = ...,
    dtype: Optional[DType] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
def masked_select(
    input: Tensor, mask: Tensor, *, out: Optional[Tensor] = ...
) -> Tensor: ...
@overload
def max(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> MaxNamedTuple[DType, L[1], Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> MaxNamedTuple[DType, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> MaxNamedTuple[DType, N1, L[1], Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> MaxNamedTuple[DType, N1, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> MaxNamedTuple[DType, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> MaxNamedTuple[DType, N1, N2, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> MaxNamedTuple[DType, Unpack[Rs], L[1]]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> MaxNamedTuple[DType, Unpack[Rs]]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    dim: L[-2],
    keepdim: L[True],
) -> MaxNamedTuple[DType, Unpack[Rs], L[1], N2]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    dim: L[-2],
    keepdim: L[False] = ...,
) -> MaxNamedTuple[DType, Unpack[Rs], N2]: ...
@overload
def max(
    input: Tensor[DType, Unpack[Rs]],
) -> Tensor[DType]: ...
@overload
def mean(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
) -> Tensor[DType, L[1], Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
) -> Tensor[DType, N1, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
) -> Tensor[DType, Unpack[Rs], L[1]]: ...
@overload
def mean(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def mean(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> Tensor[DType]: ...
@overload
def meshgrid(tensor1: Tensor[DType, N1]) -> Tuple[Tensor[DType, N1]]: ...
@overload
def meshgrid(
    tensor1: Tensor[DType, N1],
    tensor2: Tensor[DType, N2],
) -> Tuple[Tensor[DType, N1, N2], Tensor[DType, N1, N2]]: ...
@overload
def meshgrid(
    tensor1: Tensor[DType, N1],
    tensor2: Tensor[DType, N2],
    tensor3: Tensor[DType, N3],
) -> Tuple[
    Tensor[DType, N1, N2, N3], Tensor[DType, N1, N2, N3], Tensor[DType, N1, N2, N3]
]: ...
@overload
def meshgrid(*tensors: Tensor) -> Tuple[Tensor, ...]: ...
@overload
def norm(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[True],
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, L[1], Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
    keepdim: L[False] = ...,
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[True],
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, N1, L[1], Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    keepdim: L[False] = ...,
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, N1, Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[True],
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, N1, N2, L[1], Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
    keepdim: L[False] = ...,
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[True],
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, Unpack[Rs], L[1]]: ...
@overload
def norm(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
    keepdim: L[False] = ...,
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def norm(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
    *,
    out: Optional[Tensor] = ...,
    p: Optional[Number] = ...,
) -> Tensor[DType]: ...
@overload
def normal(
    mean: builtins.float,
    std: builtins.float,
    size: Tuple[Unpack[Rs]],
    *,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, Unpack[Rs]]: ...
@overload
def rand(
    size: Tuple[Unpack[Ts]],
    *,
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def rand(
    size: Tuple[Unpack[Ts]],
    *,
    dtype: Type[float32] = ...,
    low: builtins.int = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def rand(
    *size: Unpack[Ts],
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def rand(
    *size: Unpack[Ts],
    dtype: Type[float32] = ...,
    low: builtins.int = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    generator: Optional[Generator] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def randint(
    low: builtins.int,
    high: builtins.int,
    size: Tuple[Unpack[Ts]],
    dtype: Type[DType],
    *,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def randint(
    high: builtins.int,
    size: Tuple[Unpack[Ts]],
    dtype: Type[DType],
    *,
    low: builtins.int = ...,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def randint(
    low: builtins.int,
    high: builtins.int,
    size: Tuple[Unpack[Ts]],
    *,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    dtype: Type[int64] = ...,
) -> Tensor[int64, Unpack[Ts]]: ...
@overload
def randint(
    high: builtins.int,
    size: Tuple[Unpack[Ts]],
    *,
    low: builtins.int = ...,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Optional[_device] = ...,
    requires_grad: Optional[builtins.bool] = ...,
    dtype: Type[int64] = ...,
) -> Tensor[int64, Unpack[Ts]]: ...
def rand_like(
    input: Tensor[Wild, Unpack[Ts]], dtype: Type[DType]
) -> Tensor[DType, Unpack[Ts]]: ...
def nonzero(
    input: Tensor[DType, Unpack[Ts]], as_tuple: L[False] = ...
) -> LongTensor[DType, builtins.int, builtins.int]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, Unpack[Rs], N1], repeats: N, dim: L[-1]
) -> Tensor[DType, Unpack[Rs], Multiply[N1, N]]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, N1, Unpack[Rs]], repeats: N, dim: L[0]
) -> Tensor[DType, Multiply[N1, N], Unpack[Rs]]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, N1, N2, Unpack[Rs]], repeats: N, dim: L[1]
) -> Tensor[DType, N1, Multiply[N2, N], Unpack[Rs]]: ...
@overload
def repeat_interleave(
    input: Tensor[DType, Unpack[Rs]], repeats: N, dim: L[None] = ...
) -> Tensor[DType, Product[N, Unpack[Rs]]]: ...

# The output shape here depends on the contents of `repeats`, so give up.
@overload
def repeat_interleave(
    input: Tensor[DType, Unpack[Rs]], repeats: Tensor, dim: builtins.int = ...
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...
@overload
def stack(
    tensors: Tuple[Tensor[DType, N, Unpack[Ts]], Tensor[DType, N, Unpack[Ts]]],
    dim: L[1],
    *,
    out: Optional[Tensor[DType, L[2], Unpack[Ts]]] = ...,
) -> Tensor[DType, N, L[2], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[Tensor[DType, Unpack[Ts]], Tensor[DType, Unpack[Ts]]],
    dim: L[0] = ...,
    *,
    out: Optional[Tensor[DType, L[2], Unpack[Ts]]] = ...,
) -> Tensor[DType, L[2], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[
        Tensor[DType, N, Unpack[Ts]],
        Tensor[DType, N, Unpack[Ts]],
        Tensor[DType, N, Unpack[Ts]],
    ],
    dim: L[1],
    *,
    out: Optional[Tensor[DType, L[3], Unpack[Ts]]] = ...,
) -> Tensor[DType, N, L[3], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[
        Tensor[DType, Unpack[Ts]],
        Tensor[DType, Unpack[Ts]],
        Tensor[DType, Unpack[Ts]],
    ],
    dim: L[0] = ...,
    *,
    out: Optional[Tensor[DType, L[3], Unpack[Ts]]] = ...,
) -> Tensor[DType, L[3], Unpack[Ts]]: ...
@overload
def stack(
    tensors: Tuple[Any, ...],
    dim: N = ...,
    *,
    out: Optional[Tensor] = ...,
) -> Tensor: ...
def cdist(
    input: Tensor[DType, Unpack[Ts], P, M],
    other: Tensor[DType, Unpack[Rs], R, M],
    p: builtin_float = ...,
    compute_mode: str = ...,
) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]]], P, R]: ...
def clone(
    input: Tensor[DType, Unpack[Ts]], *, memory_format: Optional[memory_format] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, N1, Unpack[Rs]],
    dim: L[0],
) -> Tensor[int64, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
) -> Tensor[int64, N1, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    dim: L[2],
) -> Tensor[int64, N1, N2, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, Unpack[Rs], N1],
    dim: L[-1],
) -> Tensor[int64, Unpack[Rs]]: ...
@overload
def count_nonzero(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    keepdim: builtins.bool = ...,
) -> Tensor[int64]: ...
@overload
def sum(
    input: Tensor[DType, N1, Unpack[Rs]], dim: L[0], *, dtype: Optional[_device] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sum(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    dim: L[1],
    *,
    dtype: Optional[_device] = ...,
) -> Tensor[DType, N1, Unpack[Rs]]: ...
@overload
def sum(
    input: Tensor[DType, Unpack[Rs], N], dim: L[-1], *, dtype: Optional[_device] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sum(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    dim: L[-2],
    *,
    dtype: Optional[_device] = ...,
) -> Tensor[DType, Unpack[Rs], N2]: ...
@overload
def sum(
    input: Tensor[DType, Unpack[Rs]],
    dim: L[None] = ...,
    *,
    dtype: Optional[_device] = ...,
) -> Tensor[DType]: ...
@overload
def sin(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
def cos(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
def exp(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def matmul(
    input: Tensor[DType, N1],
    other: Tensor[DType, N1],
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType]: ...
@overload
def matmul(
    input: Tensor[DType, Unpack[Rs], N1, N2],
    other: Tensor[DType, Unpack[Qs], N2, N3],
    *,
    out: Optional[Tensor] = ...,
) -> Tensor[DType, Unpack[Broadcast[Tuple[Unpack[Rs]], Tuple[Unpack[Qs]]]], N1, N3]: ...
def multinomial(
    input: Tensor[DType, Unpack[Rs], N1],
    num_samples: N2,
    replacement: builtins.bool = ...,
    *,
    generator: Optional[Generator] = ...,
) -> Tensor[DType, Unpack[Rs], N2]: ...
@overload
def unbind(
    input: Tensor[DType, Unpack[Rs], N], dim: L[-1]
) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
@overload
def unbind(
    input: Tensor[DType, N, N1, Unpack[Rs]], dim: L[1]
) -> Tuple[Tensor[DType, N, Unpack[Rs]], ...]: ...
@overload
def unbind(
    input: Tensor[DType, N, Unpack[Rs]], dim: L[0] = ...
) -> Tuple[Tensor[DType, Unpack[Rs]], ...]: ...
@overload
def unsqueeze(
    input: Tensor[DType, Unpack[Ts]], dim: L[-1]
) -> Tensor[DType, Unpack[Ts], L[1]]: ...
@overload
def unsqueeze(
    input: Tensor[DType, Unpack[Ts]], dim: L[0]
) -> Tensor[DType, L[1], Unpack[Ts]]: ...
@overload
def unsqueeze(
    input: Tensor[DType, N, Unpack[Ts]], dim: L[1]
) -> Tensor[DType, N, L[1], Unpack[Ts]]: ...
@overload
def unsqueeze(
    input: Tensor[DType, N1, N2, Unpack[Ts]], dim: L[2]
) -> Tensor[DType, N1, N2, L[1], Unpack[Ts]]: ...
@overload
def real(input: Tensor[complex64, Unpack[Ts]]) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def real(input: Tensor[complex128, Unpack[Ts]]) -> Tensor[float64, Unpack[Ts]]: ...
def zeros_like(
    input: Tensor[DType, Unpack[Ts]],
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def randn(
    size: Tuple[Unpack[Ts]],
    dtype: Type[DType],
    *,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def randn(
    *size: Unpack[Ts],
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def randn(
    size: Tuple[Unpack[Ts]],
    *,
    out: Optional[Tensor] = ...,
    dtype: Type[float32] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def randn(
    *size: Unpack[Ts],
    out: Optional[Tensor] = ...,
    dtype: Type[float32] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def all(
    input: Tensor[DType, Unpack[Ts]],
) -> Tensor[bool, L[1]]: ...
@overload
def all(
    input: Tensor[DType, N, Unpack[Ts]],
    dim: L[0],
) -> Tensor[bool, Unpack[Ts]]: ...
@overload
def all(
    input: Tensor[DType, N1, N2, Unpack[Ts]],
    dim: L[1],
) -> Tensor[bool, N1, Unpack[Ts]]: ...
@overload
def randperm(
    n: N,
    *,
    dtype: Type[DType],
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, N]: ...
@overload
def randperm(
    n: N,
    *,
    generator: Optional[Generator] = ...,
    out: Optional[Tensor] = ...,
    dtype: Type[float32] = ...,
    layout: Optional[layout] = ...,
    device: Union[_device, str, None] = ...,
    pin_memory: builtins.bool = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[float32, N]: ...
def sqrt(
    input: Tensor[DType, Unpack[Ts]], *, out: Optional[Tensor[DType, Unpack[Ts]]] = ...
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def where(
    condition: Tensor[bool, Unpack[Ts]],
    x: Tensor[DType, Unpack[Rs]],
    y: Tensor[DType, Unpack[Rs2]],
) -> Tensor[
    DType,
    Unpack[
        Broadcast[Broadcast[Tuple[Unpack[Ts]], Tuple[Unpack[Rs]]], Tuple[Unpack[Rs2]]]
    ],
]: ...

# The exact output shape in this case depends on the contents of the tensor,
# meaning this is too dynamic for shape types.
@overload
def where(condition: Tensor[DType, Unpack[Ts]]) -> Any: ...
@overload
def diff(
    input: Tensor[DType, Unpack[Rs], Add[N1, L[1]], N2],
    dim: L[-2],
) -> Tensor[DType, Unpack[Rs], N1, N2]: ...
@overload
def diff(
    input: Tensor[DType, Add[N, L[1]], Unpack[Rs]],
    dim: L[0],
) -> Tensor[DType, N, Unpack[Rs]]: ...
@overload
def diff(
    input: Tensor[DType, N1, Add[N2, L[1]], Unpack[Rs]],
    dim: L[1],
) -> Tensor[DType, N1, N2, Unpack[Rs]]: ...
@overload
def diff(
    input: Tensor[DType, Unpack[Rs], Add[N, L[1]]],
    dim: L[-1] = ...,
) -> Tensor[DType, Unpack[Rs], N]: ...
def argsort(
    input: Tensor[DType, Unpack[Ts]],
    dim: builtins.int = ...,
    descending: builtin_bool = ...,
) -> Tensor[DType, Unpack[Ts]]: ...

# Input tuple has 2 elements.
@overload
def cat(
    tensors: Tuple[Tensor[DType, Unpack[Rs], N1], Tensor[DType, Unpack[Rs], N2]],
    dim: L[-1],
    *,
    out: Any = ...,
) -> Tensor[DType, Unpack[Rs], Add[N1, N2]]: ...
@overload
def cat(
    tensors: Tuple[Tensor[DType, N, N1, Unpack[Rs]], Tensor[DType, N, N2, Unpack[Rs]]],
    dim: L[1],
    *,
    out: Any = ...,
) -> Tensor[DType, N, Add[N1, N2], Unpack[Rs]]: ...
@overload
def cat(
    tensors: Tuple[Tensor[DType, N1, Unpack[Rs]], Tensor[DType, N2, Unpack[Rs]]],
    dim: L[0] = ...,
    *,
    out: Any = ...,
) -> Tensor[DType, Add[N1, N2], Unpack[Rs]]: ...

# Input tuple has 3 elements.
@overload
def cat(
    tensors: Tuple[
        Tensor[DType, Unpack[Rs], N1],
        Tensor[DType, Unpack[Rs], N2],
        Tensor[DType, Unpack[Rs], N3],
    ],
    dim: L[-1],
    *,
    out: Any = ...,
) -> Tensor[DType, Unpack[Rs], Add[Add[N1, N2], N3]]: ...
@overload
def cat(
    tensors: Tuple[
        Tensor[DType, N, N1, Unpack[Rs]],
        Tensor[DType, N, N2, Unpack[Rs]],
        Tensor[DType, N, N3, Unpack[Rs]],
    ],
    dim: L[1],
    *,
    out: Any = ...,
) -> Tensor[DType, N, Add[Add[N1, N2], N3], Unpack[Rs]]: ...
@overload
def cat(
    tensors: Tuple[
        Tensor[DType, N1, Unpack[Rs]],
        Tensor[DType, N2, Unpack[Rs]],
        Tensor[DType, N3, Unpack[Rs]],
    ],
    dim: L[0] = ...,
    *,
    out: Any = ...,
) -> Tensor[DType, Add[Add[N1, N2], N3], Unpack[Rs]]: ...

# This takes an arbitrary-length list of tensors and concatenates them across an
# axis. We don't know the length of the list and thus can't tell the final
# dimensions of the tensor.
@overload
def cat(
    tensors: Iterable[Tensor[DType, Unpack[Ts]]],
    dim: builtins.int = ...,
    *,
    out: Any = ...,
) -> Tensor[DType, Unpack[Tuple[Any, ...]]]: ...

save: Any
manual_seed: Any
load: Any
from_numpy: Any
no_grad: Any

def sign(
    input: Tensor[DType, Unpack[Rs]], *, out: Optional[Tensor] = ...
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sparse_coo_tensor(
    indices: Tensor,
    values: Union[Tensor, List[Any]],
    size: Tuple[Unpack[Rs]],
    *,
    dtype: Optional[DType],
    device: Union[_device, str, None] = ...,
    requires_grad: builtin_bool = ...,
) -> Tensor[DType, Unpack[Rs]]: ...
@overload
def sparse_coo_tensor(
    indices: Tensor,
    values: Union[Tensor, List[Any]],
    size: Tuple[Unpack[Rs]],
    *,
    dtype: Type[float32] = ...,
    device: Union[_device, str, None] = ...,
    requires_grad: builtin_bool = ...,
) -> Tensor[float32, Unpack[Rs]]: ...
@overload
def sparse_coo_tensor(
    indices: Tensor,
    values: Union[Tensor, List[Any]],
    size: L[None] = ...,
    *,
    dtype: Type[Any] = ...,
    device: Union[_device, str, None] = ...,
    requires_grad: builtin_bool = ...,
) -> Tensor: ...
@overload
def softmax(
    input: Tensor[DType, Unpack[Ts]], dim: builtins.int
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def softmax(
    input: Tensor[DType, Unpack[Ts]], dim: builtins.int, dtype: Type[DType2] = ...
) -> Tensor[DType2, Unpack[Ts]]: ...
@overload
def transpose(
    input: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-2], dim1: L[-1]
) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
@overload
def transpose(
    input: Tensor[DType, Unpack[Rs], N1, N2], dim0: L[-1], dim1: L[-2]
) -> Tensor[DType, Unpack[Rs], N2, N1]: ...
@overload
def transpose(
    input: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[0], dim1: L[1]
) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
@overload
def transpose(
    input: Tensor[DType, N1, N2, Unpack[Rs]], dim0: L[1], dim1: L[0]
) -> Tensor[DType, N2, N1, Unpack[Rs]]: ...
@overload
def transpose(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]], dim0: L[1], dim1: L[2]
) -> Tensor[DType, N1, N3, N2, Unpack[Rs]]: ...
@overload
def empty(
    size: Tuple[Unpack[Ts]],
    dtype: Type[DType],
    *,
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def empty(
    *size: Unpack[Ts],
    dtype: Type[DType],
    out: Optional[Tensor] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
    pin_memory: builtins.bool = ...,
    memory_format: Optional[memory_format] = ...,
) -> Tensor[DType, Unpack[Ts]]: ...
@overload
def empty(
    size: Tuple[Unpack[Ts]],
    *,
    out: Optional[Tensor] = ...,
    dtype: Type[float32] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
    pin_memory: builtins.bool = ...,
    memory_format: Optional[memory_format] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def empty(
    *size: Unpack[Ts],
    out: Optional[Tensor] = ...,
    dtype: Type[float32] = ...,
    layout: Optional[layout] = ...,
    device: _device = ...,
    requires_grad: builtins.bool = ...,
    pin_memory: builtins.bool = ...,
    memory_format: Optional[memory_format] = ...,
) -> Tensor[float32, Unpack[Ts]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, Unpack[Rs], N2],
    start_dim: L[0] = ...,
    end_dim: L[-1] = ...,
) -> Tensor[DType, Product[N1, Unpack[Rs], N2]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, N2, Unpack[Rs]],
    start_dim: L[0] = ...,
    end_dim: L[1] = ...,
) -> Tensor[DType, Multiply[N1, N2], Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, N2, N3, Unpack[Rs]],
    start_dim: L[1] = ...,
    end_dim: L[2] = ...,
) -> Tensor[DType, N1, Multiply[N2, N3], Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType, N1, N2, N3, N4, Unpack[Rs]],
    start_dim: L[2] = ...,
    end_dim: L[3] = ...,
) -> Tensor[DType, N1, N2, Multiply[N3, N4], Unpack[Rs]]: ...
@overload
def flatten(
    input: Tensor[DType],
    start_dim: L[0] = ...,
    end_dim: L[0] = ...,
) -> Tensor[DType, L[1]]: ...
@overload
def reshape(
    input: Tensor[DType, Unpack[Rs]], shape: Tuple[L[-1], Unpack[Rs2]]
) -> Tensor[DType, Divide[Product[Unpack[Rs]], Product[Unpack[Rs2]]], Unpack[Rs2]]: ...
@overload
def reshape(
    input: Tensor[DType, Unpack[Rs]], shape: Tuple[N1, L[-1], Unpack[Rs2]]
) -> Tensor[
    DType, N1, Divide[Product[Unpack[Rs]], Product[N1, Unpack[Rs2]]], Unpack[Rs2]
]: ...
@overload
def reshape(
    input: Tensor[DType, Unpack[Rs]], shape: Tuple[Unpack[Rs2]]
) -> Tensor[DType, Unpack[Rs2]]: ...
