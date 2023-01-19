# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch


class AttentionBias:
    """Base class for a custom bias that can be applied \
        in :attr:`xformers.ops.memory_efficient_attention`.

    When using an :attr:`xformers.ops.AttentionBias`
    instead of a :attr:`torch.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.


    See:

    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`

    """

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Materializes the bias as a `torch.Tensor`. This is very slow
        and we don't attempt to make it fast. Only use for debugging/testing.

        Shape should be like `[*, q_seqlen, k_seqlen]`
        """
        raise NotImplementedError()


class LowerTriangularMask(AttentionBias):
    """A lower-triangular (aka causal) mask"""

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        create_as = dtype if dtype is not torch.bfloat16 else torch.float32
        tensor = torch.full(  # type: ignore
            shape,
            dtype=create_as,
            fill_value=float("-inf"),
            device=device,
        )
        return torch.triu(tensor, diagonal=1).to(dtype)  # type: ignore

    def add_bias(self, bias: torch.Tensor) -> "LowerTriangularMaskWithTensorBias":
        return LowerTriangularMaskWithTensorBias(bias)


class LowerTriangularMaskWithTensorBias(LowerTriangularMask):
    """A lower-triangular (aka causal) mask with an additive bias"""

    def __init__(self, bias: torch.Tensor) -> None:
        self._bias = bias

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return super().materialize(shape, dtype=dtype, device=device) + self._bias


@dataclass
class _SeqLenInfo:
    max_seqlen: int
    cu_seqlen: torch.Tensor
    cu_seqlen_py: List[int]

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        cu_seqlen_py = [0]
        max_seqlen = -1
        for seqlen in seqlens:
            max_seqlen = max(max_seqlen, seqlen)
            cu_seqlen_py.append(cu_seqlen_py[len(cu_seqlen_py) - 1] + seqlen)
        cu_seqlen = torch.tensor(cu_seqlen_py, dtype=torch.int32)
        return cls(
            max_seqlen=max_seqlen, cu_seqlen=cu_seqlen, cu_seqlen_py=cu_seqlen_py
        )

    def split(
        self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        if self.cu_seqlen_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(
                f"Invalid `torch.Tensor` of shape {x.shape}, expected format "
                f"(B, M, *) with B=1 and M={self.cu_seqlen_py[-1]}\n"
                f" cu_seqlen: {self.cu_seqlen_py}"
            )
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.cu_seqlen_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(
                self.cu_seqlen_py[it + batch_size] - self.cu_seqlen_py[it]
            )
            it += batch_size
        return [
            tensor.reshape([bs, -1, *tensor.shape[2:]])
            for bs, tensor in zip(batch_sizes, x.split(split_chunks, dim=1))
        ]


@dataclass
class BlockDiagonalMask(AttentionBias):
    """A block-diagonal mask - can be used to handle batch elements with different sequence length"""

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _SeqLenInfo
    _batch_sizes: Optional[Sequence[int]] = None

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        assert shape[-1] == self.k_seqinfo.cu_seqlen_py[-1]
        assert shape[-2] == self.q_seqinfo.cu_seqlen_py[-1]
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for q_start, q_end, k_start, k_end in zip(
            self.q_seqinfo.cu_seqlen_py,
            self.q_seqinfo.cu_seqlen_py[1:],
            self.k_seqinfo.cu_seqlen_py,
            self.k_seqinfo.cu_seqlen_py[1:],
        ):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask(
                (q_end - q_start, k_end - k_start),
                dtype=dtype,
                device=device,
            )
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_seqlen: Optional[Sequence[int]] = None,
    ) -> "BlockDiagonalMask":
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        if kv_seqlen is None or q_seqlen == kv_seqlen:
            k_seqinfo = q_seqinfo
        else:
            k_seqinfo = _SeqLenInfo.from_seqlens(kv_seqlen)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    @classmethod
    def from_tensor_list(
        cls,
        tensors: Sequence[torch.Tensor],
    ) -> Tuple["BlockDiagonalMask", torch.Tensor]:
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        seqlens = []
        for x in tensors:
            for _ in range(x.shape[0]):
                seqlens.append(x.shape[1])
        block_diag = cls.from_seqlens(seqlens)
        block_diag._batch_sizes = batch_sizes
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in tensors)
        concat_tensors = torch.cat(tensors_bs1, dim=1)
        return block_diag, concat_tensors

    @classmethod
    def from_tensor_lists_qkv(
        cls,
        tensors_q: Sequence[torch.Tensor],
        tensors_k: Sequence[torch.Tensor],
        tensors_v: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple["BlockDiagonalMask", torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        assert len(tensors_q) == len(tensors_k)
        assert tensors_v is None or len(tensors_v) == len(tensors_q)
        batch_sizes = [tensor.shape[0] for tensor in tensors_q]
        q_seqlens, kv_seqlens = [], []
        for i, (q, k) in enumerate(zip(tensors_q, tensors_k)):
            assert q.shape[0] == k.shape[0]
            q_seqlens += [q.shape[1]] * q.shape[0]
            kv_seqlens += [k.shape[1]] * k.shape[0]
            assert tensors_v is None or tensors_v[i].shape[:2] == k.shape[:2]
        block_diag = cls.from_seqlens(q_seqlens, kv_seqlens)
        block_diag._batch_sizes = batch_sizes
        return (
            block_diag,
            torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_q], dim=1),
            torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_k], dim=1),
            torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], dim=1)
            if tensors_v is not None
            else None,
        )

    def split_queries(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def split_kv(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        return self.k_seqinfo.split(tensor, self._batch_sizes)

    def split(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        assert self.q_seqinfo is self.k_seqinfo
        return self.q_seqinfo.split(tensor, self._batch_sizes)

    def make_causal(self) -> "BlockDiagonalCausalMask":
        """Makes each block causal"""
        return BlockDiagonalCausalMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
        )


@dataclass
class BlockDiagonalCausalMask(BlockDiagonalMask):
    """Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal"""

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return LowerTriangularMask().materialize(
            shape,
            dtype=dtype,
            device=device,
        )
