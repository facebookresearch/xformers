# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Tuple, Type, TypeVar, Union

import torch
from torch import nn

from .unbind import stack_or_none, unbind

T_GLU_OP_DISPATCH = TypeVar("T_GLU_OP_DISPATCH", bound="GLUOpDispatchBase")
T_GLU_OP = TypeVar("T_GLU_OP", bound="GLUOpBase")


class GLUOpBase(Generic[T_GLU_OP_DISPATCH]):
    """Base class for any variant of the GLU operator"""

    def __init__(self, op, packed_weights: bool, name: str, constraints):
        self.NAME = name
        self.PACKED_WEIGHTS = packed_weights
        self.op = op
        self.constraints = constraints

    def supports(self, op: T_GLU_OP_DISPATCH) -> bool:
        if self.PACKED_WEIGHTS and not op.packed_weights:
            return False
        return all(c(op) for c in self.constraints)

    def __call__(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class GLUOpDispatchBase(Generic[T_GLU_OP], ABC):
    """Dispatcher to automatically select the best operator"""

    device: Union[torch.device, str]
    dtype: torch.dtype
    dtype_autocast_gpu: Optional[torch.dtype]
    packed_weights: bool
    bias_enabled: bool

    @abstractmethod
    def get_op_priorities(self):
        raise NotImplementedError

    @abstractmethod
    def get_default_op(self):
        raise NotImplementedError

    @property
    def op(self) -> T_GLU_OP:
        """Computes the best operator

        Returns:
            An instance of the GLUOpBase subclass: The best operator for the configuration
        """
        priorities = self.get_op_priorities()
        for op in priorities:
            if op.supports(self):
                return op
        return self.get_default_op()

    @classmethod
    def from_arguments(
        cls: Type[T_GLU_OP_DISPATCH],
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: Optional[torch.Tensor],
        w2: torch.Tensor,
        b2: Optional[torch.Tensor],
        w3: torch.Tensor,
        b3: Optional[torch.Tensor],
    ) -> T_GLU_OP_DISPATCH:
        return cls(
            device=x.device,
            dtype=x.dtype,
            packed_weights=stack_or_none((w1, w2), dim=0) is not None,
            dtype_autocast_gpu=torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else w1.dtype,
            bias_enabled=b1 is not None and b2 is not None and b3 is not None,
        )


def _only_sm80(op: GLUOpDispatchBase) -> bool:
    device_type = op.device if isinstance(op.device, str) else op.device.type
    return device_type == "cuda" and torch.cuda.get_device_capability(op.device)[0] >= 8


def _only_half_or_autocast(op: GLUOpDispatchBase) -> bool:
    HALF_DTYPES = [torch.half, torch.bfloat16]
    return op.dtype in HALF_DTYPES or (
        op.dtype_autocast_gpu is not None and op.dtype_autocast_gpu in HALF_DTYPES
    )


def _bias_enabled(op: GLUOpDispatchBase) -> bool:
    return op.bias_enabled


def _glu_ffn_variant(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    b3: Optional[torch.Tensor],
    *,
    op: GLUOpBase,
) -> torch.Tensor:
    """
    Computes one of the GLU FFN variants given the weights/bias of the 3
    linear layers.

    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = activation_function(x1) * x2
        return F.linear(hidden, w3, b3)

    :Packing weights:

    To allow faster implementations, it's recommended to have w1/w2 come from the same storage, as in:
        .. code-block:: python

            w1, w2 = xformers.ops.unbind(w12, 0)

    :Supported hardware:

    This operator is only optimized on A100+ on ``torch.half`` or ``torch.bfloat16`` \
        (autocast is supported), and will fallback to a functional pytorch \
        implementation otherwise.
    """

    batch_shape = x.shape[:-1]
    x = x.reshape([-1, x.shape[-1]])
    if w1.ndim != 2 or w1.shape != w2.shape:
        raise ValueError(f"Invalid shapes for w1: {w1.shape} / w2: {w2.shape}")
    if b1 is not None:
        if b1.ndim != 1 or b1.shape[0] != w1.shape[0]:
            raise ValueError(f"Invalid shapes for b1: {b1.shape}")
    if b2 is not None:
        if b2.ndim != 1 or b2.shape[0] != w2.shape[0]:
            raise ValueError(f"Invalid shapes for b2: {b2.shape}")
    if w3.ndim != 2 or w3.shape[1] != w2.shape[0]:
        raise ValueError(f"Invalid shape for w3: {w3.shape}")
    if b3 is not None:
        if b3.ndim != 1 or b3.shape[0] != w3.shape[0]:
            raise ValueError(f"Invalid shapes for w3: {w3.shape} / b3: {b3.shape}")

    if not op.PACKED_WEIGHTS:
        return op(x, w1, b1, w2, b2, w3, b3).reshape([*batch_shape, -1])
    w1w2 = stack_or_none((w1, w2), dim=0)
    if b1 is not None and b2 is not None:
        b1b2: Optional[torch.Tensor] = stack_or_none((b1, b2), dim=0)
        if b1b2 is None:
            raise NotImplementedError("b1/b2 needs to be properly packed")
    else:
        b1b2 = None
        assert b1 is None and b2 is None

    if w1w2 is None:
        raise NotImplementedError("w1/w2 needs to be properly packed")
    return op(x, w1w2, b1b2, w3, b3).reshape([*batch_shape, -1])


def _glu_ffn_variant_packed(
    x: torch.Tensor,
    w1w2: torch.Tensor,
    b1b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    b3: Optional[torch.Tensor],
    *,
    op: GLUOpBase,
) -> torch.Tensor:
    """
    Computes one of the GLU FFN variants given the weights/bias of the 3
    linear layers.

    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = activation_function(x1) * x2
        return F.linear(hidden, w3, b3)

    :Supported hardware:

    This operator is only optimized on A100+ on ``torch.half`` or ``torch.bfloat16`` \
        (autocast is supported), and will fallback to a functional pytorch \
        implementation otherwise.
    """
    batch_shape = x.shape[:-1]
    x = x.reshape([-1, x.shape[-1]])

    if b3 is not None:
        if b3.ndim != 1 or b3.shape[0] != w3.shape[0]:
            raise ValueError(f"Invalid shapes for w3: {w3.shape} / b3: {b3.shape}")

    assert op.PACKED_WEIGHTS, "Not implemented PACKED_WEIGHTS"

    return op(x, w1w2, b1b2, w3, b3).reshape([*batch_shape, -1])


class GLUFFNBase(nn.Module, Generic[T_GLU_OP], ABC):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        *,
        _pack_weights: bool = True,
    ) -> None:
        """Common code of the constructor for all GLU FFN variants

        Args:
            in_features (int): Number of features of the input
            hidden_features (int): Number of hidden features
            out_features (Optional[int], optional): Number of features of the input. Defaults to None.
            bias (bool, optional): Whether linear layers also include a bias. Defaults to True.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w12: Optional[nn.Linear]
        if _pack_weights:
            self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

        self.hidden_features = hidden_features
        self.out_features = out_features
        self.in_features = in_features
        self.op: Optional[T_GLU_OP] = None

    def _packed_ordered_params(
        self,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        assert self.w12 is not None, "Packed weights are only available when using w12"

        """Used for testing - returns ordered arguments for packed operators"""
        w1w2 = self.w12.weight
        b1b2_param = self.w12.bias

        w1w2 = w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]])

        b1b2: Optional[torch.Tensor] = None
        if b1b2_param is not None:
            b1b2 = b1b2_param.view([2, b1b2_param.shape[0] // 2])

        return (
            w1w2,
            b1b2,
            self.w3.weight,
            self.w3.bias,
        )

    def _ordered_params(
        self,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Used for testing - returns ordered arguments for operators"""
        b1: Optional[torch.Tensor]
        b2: Optional[torch.Tensor]
        if self.w12 is not None:
            w1w2 = self.w12.weight
            b1b2 = self.w12.bias
            w1, w2 = unbind(
                w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]),
                dim=0,
            )
            if b1b2 is not None:
                b1, b2 = unbind(b1b2.view([2, b1b2.shape[0] // 2]), dim=0)
            else:
                b1, b2 = None, None
        else:
            w1, w2 = self.w1.weight, self.w2.weight
            b1, b2 = self.w1.bias, self.w2.bias

        return (
            w1,
            b1,
            w2,
            b2,
            self.w3.weight,
            self.w3.bias,
        )

    @abstractmethod
    def _forward_packed(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def _forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes one of the GLU variants with the module's weights

        Args:
            x (torch.Tensor): A Tensor of shape ``[..., in_features]``

        Returns:
            torch.Tensor: A Tensor of shape ``[..., out_features]``
        """
        if self.w12 is not None:
            if self.op is not None:
                assert (
                    self.op.PACKED_WEIGHTS
                ), "_pack_weights and self.op.PACKED_WEIGHTS should match"
                return self._forward_packed(
                    x, *self._packed_ordered_params(), op=self.op
                )

        return self._forward(x, *self._ordered_params(), op=self.op)
