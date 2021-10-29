from typing import Any, Optional, Sequence, Union

import torch

_TensorOrTensors = Union[torch.Tensor, Sequence[torch.Tensor]]

class Function:
    @classmethod
    def apply(cls, *args: object) -> Any: ...

class enable_grad:
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

def backward(
    tensors: _TensorOrTensors,
    grad_tensors: Optional[_TensorOrTensors] = None,
    retain_graph: Optional[bool] = None,
    create_graph: bool = False,
    grad_variables: Optional[_TensorOrTensors] = None,
    inputs: Optional[_TensorOrTensors] = None,
) -> None: ...
