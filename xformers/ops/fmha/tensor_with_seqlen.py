# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
from typing import List, Optional, Sequence

import torch


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
    first_cls = None
    for a in itertools.chain(args, kwargs.values() if kwargs is not None else []):
        if isinstance(a, cls):
            if first_cls is not None:
                assert first_cls.cu_seqlen is not None
                assert first_cls.max_seqlen > 0
                assert first_cls.cu_seqlen.shape == a.cu_seqlen.shape, f"Op: {func}"
                assert first_cls.max_seqlen == a.max_seqlen, f"Op: {func}"
            first_cls = a
    assert first_cls is not None, f"Op: {func}"
    return first_cls


class TensorWithSeqLen(torch.Tensor):
    max_seqlen: int
    cu_seqlen: torch.Tensor
    cu_seqlen_py: Sequence[int]

    def __new__(cls, data):
        t = torch.Tensor._make_subclass(cls, data)
        t.cu_seqlen = None
        t.cu_seqlen_py = []
        t.max_seqlen = -1
        t.extra_state = {
            "last_func_called": "new",
        }
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
        out = type(self)(torch.empty(*args, **kwargs))
        _copy_seqlen_info(out, self)
        return out

    @classmethod
    def from_tensor_list(
        cls, tensors: Sequence[torch.Tensor], dim: int = 0
    ) -> torch.Tensor:
        c = cls(torch.cat(tuple(tensors), dim=dim))
        cu_seqlen = [0]
        max_seqlen = -1
        for tensor in tensors:
            assert not isinstance(tensor, TensorWithSeqLen)
            seqlen = tensor.shape[dim]
            max_seqlen = max(max_seqlen, seqlen)
            cu_seqlen.append(cu_seqlen[len(cu_seqlen) - 1] + seqlen)
        c.cu_seqlen = torch.tensor(
            cu_seqlen, dtype=torch.int32, device=tensors[0].device
        )
        c.cu_seqlen_py = cu_seqlen
        c.max_seqlen = max_seqlen
        return c

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
