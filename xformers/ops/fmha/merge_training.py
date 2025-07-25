# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, Tuple, Type, Union

import torch

from xformers.ops import fmha


"""
Friendly wrapper around merge_attentions which works with autograd.

Use as follows

```
partial1 = memory_efficient_attention_partial_autograd(q, k1, v1, ...)
partial2 = memory_efficient_attention_partial_autograd(q, k2, v2, ...)
attn_out = merge_attentions_autograd(partial1, partial2)
```

merge_attentions_autograd() can take any number of inputs. Note that
partial1 and partial2 are not tensors, but rather objects of type
`Partial`.

If you have partial1 and you changed your mind and don't
want to merge it with anything, you can do
```
attn_out = merge_attentions_autograd(partial1)
```

"""


class _PartialFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, fmha.AttentionBias]],
        p: float = 0.0,
        scale: Optional[float] = None,
        op: Optional[Union[fmha.AttentionOp, Type[fmha.AttentionFwOpBase]]] = None,
        output_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx.bias = attn_bias  # type: ignore
        ctx.save_for_backward(query, key, value)
        ctx.p = p  # type: ignore
        ctx.scale = scale  # type: ignore
        ctx.op = op[1] if isinstance(op, tuple) else None  # type: ignore
        attn, lse = fmha.memory_efficient_attention_partial(
            query, key, value, attn_bias, p, scale, op=op, output_dtype=output_dtype
        )
        placeholder = torch.empty_like(attn)
        return attn, lse, placeholder

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_attn: torch.Tensor,
        lse: torch.Tensor,
        out: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        query, key, value = ctx.saved_tensors  # type: ignore
        grad_q, grad_k, grad_v = fmha.memory_efficient_attention_backward(
            grad_attn,
            out,
            lse.contiguous(),
            query,
            key,
            value,
            ctx.bias,  # type: ignore
            ctx.p,  # type: ignore
            ctx.scale,  # type: ignore
            op=ctx.op,  # type: ignore
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None


class _MergeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        *inputs: torch.Tensor,
    ) -> torch.Tensor:
        ctx.length = len(inputs) // 3  # type: ignore
        if ctx.length > 1:  # type: ignore
            attns = inputs[::3]
            lses = inputs[1::3]
            out, lse = fmha.merge_attentions(attns, lses)
            assert lse is not None
        else:
            out, lse = inputs[:2]
        ctx.save_for_backward(out, lse)
        return out

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        out, lse = ctx.saved_tensors  # type: ignore
        return (grad_out, lse, out) * ctx.length  # type: ignore


class Partial:
    """
    This class is used to represent a partial attention output, which is
    returned by `memory_efficient_attention_partial_autograd`.

    Attributes: (Do not access them directly, use the methods instead.)

        _attn: torch.Tensor
        _lse: torch.Tensor . (Its grad is the full LSE to be used by
            the individual backward passes.)
        _placeholder: torch.Tensor, whose grad is used for passing the full attention
            output to the individual backward passes.
    """

    def __init__(
        self,
        attn: torch.Tensor,
        lse: torch.Tensor,
        placeholder: torch.Tensor,
    ) -> None:
        """
        Internal use only
        """
        self._attn = attn
        self._lse = lse
        self._placeholder = placeholder

    def is_bmghk(self) -> bool:
        return self._attn.ndim == 5

    def apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "Partial":
        """
        Applies fn to the attention output, as if we were a tensor.
        fn must expect tensors of shape BMGHK or BMHK, but cannot actually
        manipulate the K dimension (because the LSE doesn't have one).

        slice and pad are examples of how to use this. See also an undilation in
        a test case.
        """
        attn = fn(self._attn)
        if self.is_bmghk():
            rearranged = self._lse.permute(0, 3, 1, 2).unsqueeze(-1)
            lse = fn(rearranged).squeeze(-1).permute(0, 2, 3, 1)
        else:
            rearranged = self._lse.permute(0, 2, 1).unsqueeze(-1)
            lse = fn(rearranged).squeeze(-1).permute(0, 2, 1)
        placeholder = fn(self._placeholder)
        return self.__class__(attn, lse, placeholder)

    def _tuple(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._attn, self._lse, self._placeholder

    def do_slice(self, start: int, end: int) -> "Partial":
        """
        slice on sequence dim
        """
        return self.apply(lambda x: x[:, start:end])

    def pad(self, left: int, right: int) -> "Partial":
        """
        pad on sequence dim
        """
        pad2 = (0, 0) * (3 if self.is_bmghk() else 2) + (left, right)
        return self.apply(lambda x: torch.nn.functional.pad(x, pad2))


def memory_efficient_attention_partial_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[Union[torch.Tensor, fmha.AttentionBias]] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    *,
    op: Optional[Union[fmha.AttentionOp, Type[fmha.AttentionFwOpBase]]] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> Partial:
    """
    Wrapper around `memory_efficient_attention_partial` which works with autograd.
    Arguments are the same as for `memory_efficient_attention_partial`.
    """
    return Partial(
        *_PartialFunc.apply(query, key, value, attn_bias, p, scale, op, output_dtype)
    )


def merge_attentions_autograd(
    *partials: Partial,
) -> torch.Tensor:
    """
    Wrapper around merge_attentions which works with autograd.
    """
    args = [i for part in partials for i in part._tuple()]
    if len(args) == 0:
        raise ValueError("No partials to merge")
    return _MergeFunc.apply(*args)
