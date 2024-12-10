# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.amp import custom_bwd, custom_fwd

from .common import BaseOperator, get_xformers_operator, register_operator
from .common_glu import (
    GLUFFNBase,
    GLUOpBase,
    GLUOpDispatchBase,
    _bias_enabled,
    _glu_ffn_variant,
    _glu_ffn_variant_packed,
    _only_half_or_autocast,
    _only_sm80,
)
from .unbind import stack_or_none

if torch.version.hip:

    @torch.library.register_kernel("xformers::dual_gemm_silu_identity_mul", "cuda")  # type: ignore
    def dual_gemm_silu_identity_mul_cuda(
        x: torch.Tensor,
        w1: torch.Tensor,
        b1: Optional[torch.Tensor],
        w2: torch.Tensor,
        b2: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = x @ w1.T
        if b1 is not None:
            x1 += b1

        x2 = x @ w2.T
        if b2 is not None:
            x2 += b2

        x4 = F.silu(x1) * x2
        return x1, x2, x4


@register_operator
class DualGemmSiluOp(BaseOperator):
    OPERATOR = get_xformers_operator("dual_gemm_silu_identity_mul")
    OPERATOR_CATEGORY = "swiglu"
    NAME = "dual_gemm_silu"


@register_operator
class GemmFusedSumOp(BaseOperator):
    OPERATOR = get_xformers_operator("gemm_fused_operand_sum")
    OPERATOR_CATEGORY = "swiglu"
    NAME = "gemm_fused_operand_sum"


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
    @custom_fwd(device_type="cuda")
    def forward(cls, ctx, x, w1, b1, w2, b2, w3, b3):
        x1, x2, x4 = DualGemmSiluOp.OPERATOR(x, w1, b1, w2, b2)

        x5 = F.linear(x4, w3, b3)
        ctx.save_for_backward(x, w1, w2, w3, x1, x2)
        ctx.bias = [b1 is not None, b2 is not None, b3 is not None]
        return x5

    @staticmethod
    def _linear_bw(
        dy: torch.Tensor, x: torch.Tensor, bias: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not bias:
            return (dy.transpose(-2, -1) @ x), None
        db = torch.empty([dy.shape[1]], dtype=dy.dtype, device=dy.device)
        dw = torch.empty([dy.shape[1], x.shape[1]], dtype=dy.dtype, device=dy.device)
        GemmFusedSumOp.OPERATOR(dy.transpose(-2, -1), x, dw, db)
        return dw, db

    @classmethod
    @custom_bwd(device_type="cuda")
    def backward(cls, ctx, dx5):
        x, w1, w2, w3, x1, x2 = ctx.saved_tensors
        w1w2 = stack_or_none([w1, w2], dim=0)

        dx4 = dx5 @ w3  # 255us (nn)
        dx1dx2, x4 = torch.ops.xformers.silu_bw_fused(x1, x2, dx4)
        dx1, dx2 = dx1dx2.unbind(1)
        del x1, x2, dx4

        dw3, db3 = cls._linear_bw(dx5, x4, bias=ctx.bias[2])
        del x4, dx5
        if w1w2 is not None:
            assert dx1dx2.is_contiguous()
            assert w1w2.is_contiguous()
            w1w2 = w1w2.view([w1.shape[0] * 2, w1.shape[1]])
            dx = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]) @ w1w2

            # backward of linear1 + linear2 - packed
            dw1dw2 = dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]).transpose(-2, -1) @ x
            dw1dw2, db1db2 = cls._linear_bw(
                dx1dx2.view([dx1.shape[0], 2 * dx1.shape[1]]), x, bias=ctx.bias[0]
            )
            dw1, dw2 = dw1dw2.view([2, *w1.shape]).unbind(0)
            if ctx.bias[0]:
                db1db2 = db1db2.view([2, dx1.shape[1]])
                db1, db2 = torch.unbind(db1db2, dim=0)
            else:
                db1 = db2 = None
        else:
            dx = dx2 @ w2  # 260us (nn)
            torch.addmm(
                dx, dx1, w1.to(dx1.dtype), beta=1, alpha=1, out=dx
            )  # dx += dx1 @ w1
            dw2, db2 = cls._linear_bw(dx2, x, bias=ctx.bias[1])
            dw1, db1 = cls._linear_bw(dx1, x, bias=ctx.bias[0])
        return (dx, dw1, db1, dw2, db2, dw3, db3)


class SwiGLUOp(GLUOpBase["SwiGLUOpDispatch"]):
    """Base class for any swiglu operator in :attr:`xformers.ops.swiglu`"""

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


class SwiGLUOpDispatch(GLUOpDispatchBase[SwiGLUOp]):
    """Dispatcher to automatically select
    the best operator in :attr:`xformers.ops.swiglu`
    """

    def get_op_priorities(self) -> Sequence[SwiGLUOp]:
        priorities: Sequence[SwiGLUOp] = [
            SwiGLUPackedFusedOp,
            SwiGLUFusedOp,
        ]
        return priorities

    def get_default_op(self) -> SwiGLUOp:
        return SwiGLUEagerOp


_SwiGLUDecomposedOp = _ForwardToPythonAutogradFunc(
    _SwiGLUDecomposedFunc, False, "decomposed", constraints=[_bias_enabled]
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
    if op is None:
        op = SwiGLUOpDispatch.from_arguments(x, w1, b1, w2, b2, w3, b3).op

    y = _glu_ffn_variant(x, w1, b1, w2, b2, w3, b3, op=op)
    return y


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
    y = _glu_ffn_variant_packed(x, w1w2, b1b2, w3, b3, op=op)
    return y


class SwiGLU(GLUFFNBase[SwiGLUOp]):
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
        super().__init__(
            in_features,
            hidden_features,
            out_features,
            bias,
            _pack_weights=_pack_weights,
        )

    def _forward_packed(self, *args, **kwargs) -> torch.Tensor:
        return swiglu_packed(*args, **kwargs)

    def _forward(self, *args, **kwargs) -> torch.Tensor:
        return swiglu(*args, **kwargs)
