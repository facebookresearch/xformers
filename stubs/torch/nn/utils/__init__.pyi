from typing import Iterable, TypeVar, Union, overload

from pyre_extensions import TypeVarTuple, Unpack
from torch import Tensor, complex64

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
