# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from typing import Any, List, Optional, Sequence

import torch
from torch.utils._pytree import tree_flatten


def _copy_seqlen_info(
    dst: "TensorWithSeqLen", src: "TensorWithSeqLen", deepcopy=False
) -> None:
    dst.max_seqlen = src.max_seqlen
    if deepcopy:
        dst.cu_seqlen = copy.deepcopy(src.cu_seqlen)
        dst.cu_seqlen_py = copy.deepcopy(src.cu_seqlen_py)
    else:
        dst.cu_seqlen = src.cu_seqlen
        dst.cu_seqlen_py = src.cu_seqlen_py
        if dst.device != dst.cu_seqlen.device:
            dst.cu_seqlen = dst.cu_seqlen.to(dst.device)


def _find_arg_to_copy_metadata(cls, func, args, kwargs) -> Optional["TensorWithSeqLen"]:
    all_elements, _ = tree_flatten(args)
    elements_kwargs: List[Any] = []
    if kwargs is not None:
        elements_kwargs, _ = tree_flatten(kwargs)
    first_cls = None
    for x in itertools.chain(all_elements, elements_kwargs):
        if isinstance(x, cls):
            if first_cls is not None:
                assert first_cls.cu_seqlen is not None
                assert first_cls.max_seqlen > 0
                assert first_cls.cu_seqlen.shape == x.cu_seqlen.shape, f"Op: {func}"
                assert first_cls.max_seqlen == x.max_seqlen, f"Op: {func}"
            first_cls = x
    assert first_cls is not None, f"can't find TensorWithSeqLen argument for op: {func}"
    return first_cls


class TensorWithSeqLen(torch.Tensor):
    max_seqlen: int
    cu_seqlen: torch.Tensor
    cu_seqlen_py: Sequence[int]

    def __new__(
        cls,
        data: torch.Tensor,
        max_seqlen: int,
        cu_seqlen: torch.Tensor,
        cu_seqlen_py: Sequence[int],
    ):
        t: TensorWithSeqLen = torch.Tensor._make_subclass(cls, data, require_grad=data.requires_grad)  # type: ignore
        t.cu_seqlen_py = cu_seqlen_py
        t.cu_seqlen = cu_seqlen
        t.max_seqlen = max_seqlen
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super().__torch_function__(func, types, args, kwargs)
        if func.__name__ == "efficient_attention_forward_cutlass":
            _copy_seqlen_info(result[0], kwargs["query"])  # output
            _copy_seqlen_info(result[1], kwargs["query"])  # LSE
            return result
        elif isinstance(result, cls):
            _copy_seqlen_info(
                result, _find_arg_to_copy_metadata(cls, func, args, kwargs)
            )
        elif isinstance(result, tuple):
            src = _find_arg_to_copy_metadata(cls, func, args, kwargs)
            for r in result:
                if not isinstance(r, cls):
                    continue
                _copy_seqlen_info(r, src)
        return result

    # new_empty() must be defined for deepcopy to work
    def new_empty(self, *args, **kwargs):
        out = type(self)(torch.empty(*args, **kwargs), -1, None, [])
        _copy_seqlen_info(out, self)
        return out

    @classmethod
    def from_tensor_list(
        cls, tensors: Sequence[torch.Tensor], dim: int = 0
    ) -> torch.Tensor:
        class FromTensorListOp(torch.autograd.Function):
            @staticmethod
            # type: ignore
            def forward(
                ctx, tensor: torch.Tensor, max_seqlen: int, cu_seqlen_py: Sequence[int]
            ) -> "TensorWithSeqLen":
                cu_seqlen = torch.tensor(
                    cu_seqlen_py, dtype=torch.int32, device=tensor.device
                )
                return TensorWithSeqLen(
                    tensor,
                    max_seqlen=max_seqlen,
                    cu_seqlen=cu_seqlen,
                    cu_seqlen_py=cu_seqlen_py,
                )

            @staticmethod
            @torch.autograd.function.once_differentiable
            def backward(ctx, grad: torch.Tensor):
                return grad, None, None

        c = torch.cat(tuple(tensors), dim=dim)
        cu_seqlen = [0]
        max_seqlen = -1
        for tensor in tensors:
            assert not isinstance(tensor, TensorWithSeqLen)
            seqlen = tensor.shape[dim]
            max_seqlen = max(max_seqlen, seqlen)
            cu_seqlen.append(cu_seqlen[len(cu_seqlen) - 1] + seqlen)
        return FromTensorListOp.apply(c, max_seqlen, cu_seqlen)

    def to_tensor_list(self, dim: int) -> List[torch.Tensor]:
        if self.cu_seqlen_py[-1] != self.shape[dim]:
            raise ValueError(
                f"Invalid `TensorWithSeqLen` of shape {self.shape}.\n"
                f" cu_seqlen: {self.cu_seqlen_py}"
            )
        out = []
        for begin, end in zip(self.cu_seqlen_py, self.cu_seqlen_py[1:]):
            s = self[(slice(None),) * dim + (slice(begin, end),)]
            out.append(s)
        return out
