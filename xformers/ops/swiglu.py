# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import nn

from .common import get_xformers_operator
from .unbind import stack_or_none, unbind


class _SwiGLUModule(nn.Module):
    """
    Reference implementation of a SwiGLU module
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        align_as: int = 8,
        pack_weights: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        swiglu_hidden_features = int(2 * hidden_features / 3)
        swiglu_hidden_features = (
            (swiglu_hidden_features + align_as - 1) // align_as * align_as
        )

        self.w12: Optional[nn.Linear]
        if pack_weights:
            self.w12 = nn.Linear(in_features, 2 * swiglu_hidden_features)
        else:
            self.w12 = None
            self.w1 = nn.Linear(in_features, swiglu_hidden_features)
            self.w2 = nn.Linear(in_features, swiglu_hidden_features)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features)

        self.swiglu_hidden_features = swiglu_hidden_features
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A baseline forward. Just replace it with the following
        to use xFormers's kernels
        ```python
        return xformers.ops.functional_swiglu(x,
            *self._ordered_params_for_op(),
            op=xformers.ops.SwiGLUPackedFusedOp)
        ```
        """
        if self.w12 is not None:
            x12 = self.w12(x).view([x.shape[0], 2, self.swiglu_hidden_features])
            x1, x2 = unbind(x12, dim=1)
        else:
            x1 = self.w1(x)
            x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

    def _ordered_params_for_op(self):
        """Used for testing - returns ordered arguments for operators"""
        if self.w12 is not None:
            w1w2 = self.w12.weight
            b1b2 = self.w12.bias
            w1, w2 = unbind(
                w1w2.view([2, w1w2.shape[0] // 2, w1w2.shape[1]]),
                dim=0,
            )
            b1, b2 = unbind(b1b2.view([2, b1b2.shape[0] // 2]), dim=0)
        else:
            w1, w2 = self.w1.weight, self.w2.weight
            b1, b2 = self.w1.bias, self.w2.bias
        return [
            w1,
            b1,
            w2,
            b2,
            self.w3.weight,
            self.w3.bias,
        ]


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


class _SwiGLUFusedFunc(torch.autograd.Function):
    NAME = "fused.py"

    @classmethod
    @torch.cuda.amp.custom_fwd
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1, x2, x4 = torch.ops.xformers.dual_gemm_silu_identity_mul(x, w1, b1, w2, b2)

        x5 = F.linear(x4, w3, b3)
        ctx.save_for_backward(x, w1, w2, w3, x1, x2)
        return x5

    @classmethod
    @torch.cuda.amp.custom_bwd
    def backward(cls, ctx, dx5):
        x, w1, w2, w3, x1, x2 = ctx.saved_tensors
        w1w2 = stack_or_none([w1, w2], dim=0)

        dx4 = dx5 @ w3  # 255us (nn)
        dx1dx2, x4 = torch.ops.xformers.silu_bw_fused(x1, x2, dx4)
        dx1, dx2 = dx1dx2.unbind(1)
        del x1, x2, dx4

        db3 = dx5.sum(0)  # 25us
        dw3 = dx5.transpose(-2, -1) @ x4  # 247us (nt)
        del x4, dx5
        if w1w2 is not None:
            assert dx1dx2.is_contiguous()
            assert w1w2.is_contiguous()
            w1w2 = w1w2.view([w1.shape[0] * 2, w1.shape[1]])
            dx = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]) @ w1w2

            # backward of linear1 + linear2 - packed
            dw1dw2 = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]).transpose(-2, -1) @ x
            db1db2 = dx1dx2.sum(0).view([2, dx1.shape[1]])
            dw1, dw2 = dw1dw2.view([2, *w1.shape]).unbind(0)
            db1, db2 = torch.unbind(db1db2, dim=0)
        else:
            dx = dx2 @ w2  # 260us (nn)
            torch.addmm(
                dx, dx1, w1.to(dx1.dtype), beta=1, alpha=1, out=dx
            )  # dx += dx1 @ w1
            dw2 = dx2.transpose(-2, -1) @ x  # 245us (nt)
            db2 = dx2.sum(0)  # 50us
            dw1 = dx1.transpose(-2, -1) @ x  # 245us (nt)
            db1 = dx1.sum(0)  # 50us
        return (dx, dw1, db1, dw2, db2, dw3, db3)


class SwiGLUOp:
    def __init__(self, op, packed_weights: bool, name: str, constraints):
        self.NAME = name
        self.PACKED_WEIGHTS = packed_weights
        self.op = op
        self.constraints = constraints

    def supports(self, op: "SwiGLUOpDispatch") -> bool:
        if self.PACKED_WEIGHTS and not op.packed_weights:
            return False
        return all(c(op) for c in self.constraints)

    def __call__(self, *args: torch.Tensor) -> torch.Tensor:
        pass

    def __str__(self) -> str:
        return f"SwiGLUOp:{self.NAME}"


class _ForwardToPythonAutogradFunc(SwiGLUOp):
    def supports(self, op: "SwiGLUOpDispatch") -> bool:
        # Let's disable autocast in bf16 until this issue is fixed
        # https://github.com/pytorch/pytorch/issues/87979
        if op.dtype_autocast_gpu == torch.bfloat16:
            return False
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
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
) -> torch.Tensor:
    x1 = F.linear(x, w1, b1)
    x2 = F.linear(x, w2, b2)
    hidden = F.silu(x1) * x2
    return F.linear(hidden, w3, b3)


@dataclass
class SwiGLUOpDispatch:
    device: Union[torch.device, str]
    dtype: torch.dtype
    dtype_autocast_gpu: Optional[torch.dtype]
    packed_weights: bool

    @property
    def op(self) -> SwiGLUOp:
        priorities: Sequence[SwiGLUOp] = [
            SwiGLUPackedFusedOp,
            SwiGLUFusedOp,
        ]
        for op in priorities:
            if op.supports(self):
                return op
        return SwiGLUEagerOp

    @staticmethod
    def from_arguments(
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        w3: torch.Tensor,
        b3: torch.Tensor,
    ) -> "SwiGLUOpDispatch":
        return SwiGLUOpDispatch(
            device=x.device,
            dtype=x.dtype,
            packed_weights=stack_or_none((w1, w2), dim=0) is not None,
            dtype_autocast_gpu=torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else w1.dtype,
        )


def _only_sm80(op: SwiGLUOpDispatch) -> bool:
    return (
        str(op.device) == "cuda" and torch.cuda.get_device_capability(op.device)[0] >= 8
    )


def _only_half_or_autocast(op: SwiGLUOpDispatch) -> bool:
    HALF_DTYPES = [torch.half, torch.bfloat16]
    return op.dtype in HALF_DTYPES or (
        op.dtype_autocast_gpu is not None and op.dtype_autocast_gpu in HALF_DTYPES
    )


_SwiGLUDecomposedOp = _ForwardToPythonAutogradFunc(
    _SwiGLUDecomposedFunc, False, "decomposed", constraints=[]
)
SwiGLUFusedOp = _ForwardToPythonAutogradFunc(
    _SwiGLUFusedFunc, False, "fused", constraints=[_only_sm80, _only_half_or_autocast]
)
SwiGLUPackedFusedOp = _ForwardToFunc(
    get_xformers_operator("swiglu_packedw"),
    True,
    "fused.p.cpp",
    constraints=[_only_sm80, _only_half_or_autocast],
)
SwiGLUEagerOp = _ForwardToFunc(
    _eager_functional_swiglu,
    False,
    "eager",
    constraints=[],
)


def _info() -> Dict[str, str]:
    return {op.NAME: op.info() for op in [SwiGLUPackedFusedOp]}


def functional_swiglu(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
    w3: torch.Tensor,
    b3: torch.Tensor,
    *,
    op: SwiGLUOp = None,
) -> torch.Tensor:
    """
    Computes a SwiGLU block given the weights/bias of the 3
    linear layers.
    It is recommended to keep `op=None` so the best implementation
    available for the inputs will be used.

    NOTE: It's better to provide w1/w2 and b1/b2 are packed together
        to allow for faster implementations
    """

    if x.ndim != 2:
        raise ValueError(f"Expected shape for x: [batch, n_input] but got {x.shape}")
    if w1.ndim != 2 or w1.shape != w2.shape:
        raise ValueError(f"Invalid shapes for w1: {w1.shape} / w2: {w2.shape}")
    if b1.ndim != 1 or b1.shape != b2.shape or b1.shape[0] != w1.shape[0]:
        raise ValueError(f"Invalid shapes for b1: {b1.shape} / b2: {b2.shape}")
    if (
        (w3.ndim, b3.ndim) != (2, 1)
        or w3.shape[1] != w2.shape[0]
        or b3.shape[0] != w3.shape[0]
    ):
        raise ValueError(f"Invalid shapes for w3: {w3.shape} / b3: {b3.shape}")

    if op is None:
        op = SwiGLUOpDispatch.from_arguments(x, w1, b1, w2, b2, w3, b3).op

    if not op.PACKED_WEIGHTS:
        return op(x, w1, b1, w2, b2, w3, b3)
    w1w2 = stack_or_none((w1, w2), dim=0)
    b1b2 = stack_or_none((b1, b2), dim=0)
    if w1w2 is None:
        raise NotImplementedError("w1/w2 needs to be properly packed")
    if b1b2 is None:
        raise NotImplementedError("b1/b2 needs to be properly packed")
    return op(x, w1w2, b1b2, w3, b3)
