# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from .unbind import stack_or_none, unbind


class _SwiGLUDecomposedFunc(torch.autograd.Function):
    """
    This is just an example implementation with all
    operations explicited. This implementation is worse
    than pytorch, because pytorch is able to fuse some operations
    (eg the linear forward ...) that are decomposed here.

    The time measurements were made on the ViT-Giant setting:
    - A100/f16
    - input: [4440, 1536]
    - hidden: [4440, 4096]
    """

    NAME = "decomposed"
    FORCE_BW_F32 = False

    def _silu_backward(dy, x):
        # https://github.com/pytorch/pytorch/blob/563b065f5a4b4055fa6b025c2514b566d5fd9439/aten/src/ATen/native/Activation.cpp#L483
        sigm = 1 / (1 + torch.exp(-x.float()))
        return (dy.float() * sigm * (1 + x.float() * (1 - sigm))).to(x.dtype)

    # 952us
    @classmethod
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1 = x @ w1.transpose(-2, -1) + b1  # 275us
        x2 = x @ w2.transpose(-2, -1) + b2  # 275us
        x3 = F.silu(x1)  # 62us
        x4 = x3 * x2  # 90us
        x5 = x4 @ w3.transpose(-2, -1) + b3  # 250us

        ctx.save_for_backward(x, w1, b1, w2, b2, w3, b3, x1, x2, x3, x4, x5)
        return x5

    # 1900us
    @classmethod
    def backward(cls, ctx, dx5):
        saved_tensors = ctx.saved_tensors
        if cls.FORCE_BW_F32:
            dx5 = dx5.float()
            saved_tensors = [t.float() for t in ctx.saved_tensors]
        x, w1, b1, w2, b2, w3, b3, x1, x2, x3, x4, x5 = saved_tensors
        dx4 = dx5 @ w3  # 255us (nn)
        dw3 = dx5.transpose(-2, -1) @ x4  # 247us (nt)
        db3 = dx5.sum(0)  # 25us
        dx3 = dx4 * x2  # 88us
        dx2 = dx4 * x3  # 88us
        dx1 = cls._silu_backward(dx3, x1)  # 90us
        dx = dx2 @ w2  # 260us (nn)
        dw2 = dx2.transpose(-2, -1) @ x  # 245us (nt)
        db2 = dx2.sum(0)  # 50us
        dx += dx1 @ w1  # 260us (nn)
        dw1 = dx1.transpose(-2, -1) @ x  # 245us (nt)
        db1 = dx1.sum(0)  # 50us
        return (dx, dw1, db1, dw2, db2, dw3, db3)


class SwiGLUOp:
    """Base class for any swiglu operator in :attr:`xformers.ops.swiglu`"""

    def __init__(self, op, packed_weights: bool, name: str, constraints):
        self.NAME = name
        self.PACKED_WEIGHTS = packed_weights
        self.op = op
        self.constraints = constraints

    def supports(self, op: "SwiGLUOpDispatch") -> bool:
        if self.PACKED_WEIGHTS and not op.packed_weights:
            return False
        return all(c(op) for c in self.constraints)

    def __call__(self, *args: Optional[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"SwiGLUOp:{self.NAME}"


class _ForwardToPythonAutogradFunc(SwiGLUOp):
    def supports(self, op: "SwiGLUOpDispatch") -> bool:
        return super().supports(op)

    def __call__(self, *args, **kwargs):
        return self.op.apply(*args, **kwargs)


class _ForwardToFunc(SwiGLUOp):
    def __call__(self, *args, **kwargs):
        return self.op(*args, **kwargs)

    def info(self):
        if self.op.__name__ == "no_such_operator":
            return "not built"
        return "available"


def _eager_functional_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    b3: Optional[torch.Tensor],
) -> torch.Tensor:
    x1 = F.linear(x, w1, b1)
    x2 = F.linear(x, w2, b2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3, b3)


@dataclass
class SwiGLUOpDispatch:
    """Dispatcher to automatically select
    the best operator in :attr:`xformers.ops.swiglu`
    """

    device: Union[torch.device, str]
    dtype: torch.dtype
    dtype_autocast_gpu: Optional[torch.dtype]
    packed_weights: bool
    bias_enabled: bool

    @property
    def op(self) -> SwiGLUOp:
        """Computes the best operator

        Returns:
            SwiGLUOp: The best operator for the configuration
        """
        return SwiGLUEagerOp

    @staticmethod
    def from_arguments(
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: Optional[torch.Tensor],
        w2: torch.Tensor,
        b2: Optional[torch.Tensor],
        w3: torch.Tensor,
        b3: Optional[torch.Tensor],
    ) -> "SwiGLUOpDispatch":
        return SwiGLUOpDispatch(
            device=x.device,
            dtype=x.dtype,
            packed_weights=stack_or_none((w1, w2), dim=0) is not None,
            dtype_autocast_gpu=(
                torch.get_autocast_gpu_dtype()
                if torch.is_autocast_enabled()
                else w1.dtype
            ),
            bias_enabled=b1 is not None and b2 is not None and b3 is not None,
        )


def _bias_enabled(op: SwiGLUOpDispatch) -> bool:
    return op.bias_enabled


_SwiGLUDecomposedOp = _ForwardToPythonAutogradFunc(
    _SwiGLUDecomposedFunc, False, "decomposed", constraints=[_bias_enabled]
)
SwiGLUEagerOp = _ForwardToFunc(
    _eager_functional_swiglu,
    False,
    "eager",
    constraints=[],
)


def swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    b3: Optional[torch.Tensor],
    *,
    op: Optional[SwiGLUOp] = None,
) -> torch.Tensor:
    """
    Computes a SwiGLU block given the weights/bias of the 3
    linear layers.

    - It is recommended to keep ``op=None`` so the best implementation \
    available for the inputs will be used.


    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = F.silu(x1) * x2
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

    if op is None:
        op = SwiGLUOpDispatch.from_arguments(x, w1, b1, w2, b2, w3, b3).op

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


def swiglu_packed(
    x: torch.Tensor,
    w1w2: torch.Tensor,
    b1b2: Optional[torch.Tensor],
    w3: torch.Tensor,
    b3: Optional[torch.Tensor],
    *,
    op: SwiGLUOp,
) -> torch.Tensor:
    """
    Computes a SwiGLU block given the weights/bias of the 3
    linear layers.

    :Equivalent pytorch code:

    .. code-block:: python

        x1 = F.linear(x, w1, b1)
        x2 = F.linear(x, w2, b2)
        hidden = F.silu(x1) * x2
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


class SwiGLU(nn.Module):
    """
    A Module that encapsulates the call to :attr:`xformers.ops.swiglu`,
    and holds the weights for the 3 linear layers
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
        *,
        _pack_weights: bool = True,
    ) -> None:
        """Create a SwiGLU module

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
        self.op: Optional[SwiGLUOp] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes :attr:`swiglu` with the module's weights

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
                return swiglu_packed(x, *self._packed_ordered_params(), op=self.op)

        return swiglu(x, *self._ordered_params(), op=self.op)

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
