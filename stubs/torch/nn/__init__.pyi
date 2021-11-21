# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import builtins
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import torch
from pyre_extensions import Add, Divide, Multiply, Subtract, TypeVarTuple, Unpack
from torch import Tensor
from typing_extensions import Literal as L

DType = TypeVar("DType")
T = TypeVar("T")
Ts = TypeVarTuple("Ts")
InputSize = TypeVar("InputSize", bound=int)
OutputSize = TypeVar("OutputSize", bound=int)
HiddenSize = TypeVar("HiddenSize", bound=int)
Batch = TypeVar("Batch", bound=int)
N = TypeVar("N", bound=int)
EmbeddingDimension = TypeVar("EmbeddingDimension", bound=int)
H = TypeVar("H", bound=int)
W = TypeVar("W", bound=int)

class Module:
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def parameters(self) -> Iterator[Any]: ...
    def double(self: T) -> T: ...
    def to(self, dtype: Type[T], device: torch._device = ...) -> Module: ...
    def eval(self) -> Module: ...
    def train(self, mode: bool) -> Module: ...
    def register_parameter(self, name: str, param: Optional[Parameter]) -> None: ...
    bias: Parameter = ...
    training: bool = ...

class LSTMCell(Module, Generic[InputSize, HiddenSize]):
    def __init__(
        self, input_size: InputSize, hidden_size: HiddenSize, bias: bool = ...
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, Batch, InputSize],
        hidden: Tuple[
            Tensor[DType, Batch, HiddenSize], Tensor[DType, Batch, HiddenSize]
        ] = ...,
    ) -> Tuple[Tensor[DType, Batch, HiddenSize], Tensor[DType, Batch, HiddenSize]]: ...

class Linear(Module, Generic[InputSize, OutputSize]):
    def __init__(
        self, in_features: InputSize, out_features: OutputSize, bias: bool = ...
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, N, Unpack[Ts], InputSize],
    ) -> Tensor[DType, N, Unpack[Ts], OutputSize]: ...

class _Loss(Module): ...

class MSELoss(_Loss):
    def __init__(
        self,
        size_average: Optional[bool] = ...,
        reduce: Optional[bool] = ...,
        reduction: str = ...,
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, N, Unpack[Ts]],
        target: Tensor[DType, N, Unpack[Ts]],
    ) -> Tensor[DType]: ...

InChannels = TypeVar("InChannels", bound=int)
OutChannels = TypeVar("OutChannels", bound=int)
KernelSize1 = TypeVar("KernelSize1", bound=int)
KernelSize2 = TypeVar("KernelSize2", bound=int)
Stride = TypeVar("Stride", bound=int)
Batch = TypeVar("Batch", bound=int)
Height = TypeVar("Height", bound=int)
Width = TypeVar("Width", bound=int)
Channels = TypeVar("Channels", bound=int)
Padding = TypeVar("Padding", bound=int)
Padding1 = TypeVar("Padding1", bound=int)
Padding2 = TypeVar("Padding2", bound=int)

class Conv2d(
    Module,
    Generic[InChannels, OutChannels, KernelSize1, KernelSize2, Padding1, Padding2],
):
    def __init__(
        self,
        in_channels: InChannels,
        out_channels: OutChannels,
        kernel_size: Tuple[KernelSize1, KernelSize2],
        padding: Tuple[Padding1, Padding2],
        bias: bool = ...,
    ) -> None: ...
    def __call__(
        self, input: Tensor[DType, Batch, InChannels, Height, Width]
    ) -> Tensor[
        DType,
        Batch,
        OutChannels,
        # We assume stride = 1.
        # (H − K[0] + 2P[0]) + 1.
        Add[Add[Subtract[Height, KernelSize1], Multiply[Padding1, L[2]]], L[1]],
        # (W − K[1] + 2P[1]) + 1.
        Add[Add[Subtract[Width, KernelSize2], Multiply[Padding2, L[2]]], L[1]],
    ]: ...

class ReflectionPad2d(Module, Generic[Padding]):
    def __init__(
        self,
        padding: Padding,
    ) -> None: ...
    def __call__(
        self,
        input: Tensor[DType, Batch, Channels, Height, Width],
    ) -> Tensor[
        DType,
        Batch,
        Channels,
        Add[Add[Height, Padding], Padding],
        Add[Add[Width, Padding], Padding],
    ]: ...

class InstanceNorm2d(Generic[Channels]):
    def __init__(self, num_features: Channels, affine: bool = False) -> None: ...
    def __call__(
        self, input: Tensor[DType, Batch, Channels, Height, Width]
    ) -> Tensor[DType, Batch, Channels, Height, Width]: ...

class LeakyReLU(Module):
    def __init__(self, negative_slope: float = ..., inplace: bool = ...) -> None: ...
    def __call__(
        self, input: Tensor[DType, N, Unpack[Ts]]
    ) -> Tensor[DType, N, Unpack[Ts]]: ...

class ReLU(Module):
    def __call__(
        self, input: Tensor[DType, Batch, Channels, Height, Width]
    ) -> Tensor[DType, Batch, Channels, Height, Width]: ...

class GELU(Module):
    def __call__(
        self, input: Tensor[DType, Batch, Channels, Height, Width]
    ) -> Tensor[DType, Batch, Channels, Height, Width]: ...

class Dropout(Module):
    def __init__(self, p: float, inplace: bool = ...) -> None: ...
    def __call__(
        self, input: Tensor[DType, Unpack[Ts]]
    ) -> Tensor[DType, Unpack[Ts]]: ...

class Embedding(Module, Generic[N, EmbeddingDimension]):
    def __init__(
        self,
        num_embeddings: N,
        embedding_dim: EmbeddingDimension,
        padding_idx: Optional[int] = ...,
        max_norm: Optional[float] = ...,
        norm_type: float = ...,
        scale_grad_by_freq: bool = ...,
        sparse: bool = ...,
        _weight: Optional[Tensor] = ...,
    ) -> None: ...
    @property
    def padding_idx(self) -> int: ...
    @property
    def max_norm(self) -> float: ...
    @property
    def norm_type(self) -> float: ...
    @property
    def scale_grad_by_freq(self) -> bool: ...
    @property
    def sparse(self) -> bool: ...
    @property
    def weight(self) -> Tensor[torch.float32, N, EmbeddingDimension]: ...
    @classmethod
    def from_pretrained(
        cls,
        embeddings: Tensor[DType, N, EmbeddingDimension],
        freeze: bool = True,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
    ) -> Embedding[N, EmbeddingDimension]: ...
    def forward(
        self, x: Tensor[DType, Unpack[Ts]]
    ) -> Tensor[DType, Unpack[Ts], EmbeddingDimension]: ...
    def __call__(
        self, x: Tensor[DType, Unpack[Ts]]
    ) -> Tensor[DType, Unpack[Ts], EmbeddingDimension]: ...

_shape_t = Union[int, List[int], Tuple[Any, ...]]

class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = ...,
        elementwise_affine: bool = ...,
        device=...,
        dtype=...,
    ) -> None: ...
    def forward(self, x: Tensor[DType, Unpack[Ts]]) -> Tensor[DType, Unpack[Ts]]: ...
    def __call__(self, x: Tensor[DType, Unpack[Ts]]) -> Tensor[DType, Unpack[Ts]]: ...

class AdaptiveAvgPool2d(Module, Generic[H, W]):
    @overload
    def __new__(
        self,
        output_size: Tuple[N, L[None]],
    ) -> AdaptiveAvgPool2d[N, L[-1]]: ...
    @overload
    def __new__(
        self,
        output_size: Tuple[L[None], N],
    ) -> AdaptiveAvgPool2d[L[-1], N]: ...
    @overload
    def __new__(
        self,
        output_size: H,
    ) -> AdaptiveAvgPool2d[H, H]: ...
    @overload
    def __new__(
        self,
        output_size: Tuple[H, W],
    ) -> AdaptiveAvgPool2d[H, W]: ...
    def forward(self, x: Tensor[DType, Unpack[Ts]]) -> Tensor[DType, Unpack[Ts]]: ...
    @overload
    def __call__(
        self: AdaptiveAvgPool2d[L[-1], W], x: Tensor[DType, Unpack[Ts], N, int]
    ) -> Tensor[DType, Unpack[Ts], N, W]: ...
    @overload
    def __call__(
        self: AdaptiveAvgPool2d[H, L[-1]], x: Tensor[DType, Unpack[Ts], int, N]
    ) -> Tensor[DType, Unpack[Ts], H, N]: ...
    @overload
    def __call__(
        self: AdaptiveAvgPool2d[H, W], x: Tensor[DType, Unpack[Ts], int, int]
    ) -> Tensor[DType, Unpack[Ts], H, W]: ...

class ModuleList(Module):
    def __init__(self, modules: Optional[Iterable[Module]] = ...) -> None: ...
    def __iter__(self) -> Iterator[Module]: ...
    def __len__(self) -> int: ...

class Parameter(Tensor[DType, Unpack[Ts]]):
    def __init__(
        self, data: Tensor[DType, Unpack[Ts]] = ..., requires_grad: builtins.bool = ...
    ) -> None: ...

Sequential: Any = ...
