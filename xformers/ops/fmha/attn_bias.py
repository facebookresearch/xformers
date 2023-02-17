# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import math
from dataclasses import dataclass, field
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
    cu_seqlen: torch.Tensor
    max_seqlen: int = field(init=False)
    cu_seqlen_py: List[int] = field(init=False)

    def __post_init__(self):
        self.cu_seqlen_py = self.cu_seqlen.tolist()
        self.max_seqlen = max(self.cu_seqlen_py)

    def to(self, device: torch.device) -> None:
        self.cu_seqlen = self.cu_seqlen.to(device, non_blocking=True)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.cu_seqlen.tolist(), self.cu_seqlen.tolist()[1:])

    @classmethod
    def from_seqlens(cls, seqlens: Union[Iterable[int], torch.Tensor]) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        If padding is not None, seqstart = padding * torch.arange(batch_size)
        """
        t_seqlens = (
            seqlens
            if isinstance(seqlens, torch.Tensor)
            else torch.tensor(seqlens, dtype=torch.int32)
        )
        cu_seqlen = torch.cumsum(
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=t_seqlens.device),
                    t_seqlens.int(),
                ],
                dim=0,
            ),
            dim=0,
        )
        return cls(cu_seqlen=cu_seqlen.int())

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
class _PaddedSeqLenInfo(_SeqLenInfo):
    seqlen: torch.Tensor

    def to(self, device: torch.device) -> None:
        if self.seqlen is not None:
            self.seqlen = self.seqlen.to(device, non_blocking=True)
        super().to(device)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        assert self.seqlen is not None
        for (start, _), length in zip(super().intervals(), self.seqlen):
            yield start, start + length

    @classmethod
    def from_seqlens(cls, seqlens: Union[Iterable[int], torch.Tensor]) -> "_SeqLenInfo":
        raise RuntimeError(
            "Use either `_SeqLenInfo.from_seqlens` or `_PaddedSeqLenInfo.from_seqlens_padded`"
        )

    @classmethod
    def from_seqlens_padded(
        cls, seqlens: Union[Iterable[int], torch.Tensor], padding: int
    ) -> "_PaddedSeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        seqstart = padding * torch.arange(batch_size)
        """
        t_seqlens = (
            seqlens
            if isinstance(seqlens, torch.Tensor)
            else torch.tensor(seqlens, dtype=torch.int32)
        )
        cu_seqlen = (
            # + 1 because expected to be same shape as cu_seqlen_k
            torch.arange(
                t_seqlens.shape[0] + 1, dtype=torch.int32, device=t_seqlens.device
            )
            * padding
        )
        return cls(
            cu_seqlen=cu_seqlen,
            seqlen=t_seqlens,
        )

    def split(
        self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        raise NotImplementedError("_PaddedSeqLenInfo.split")


@dataclass
class BlockDiagonalMask(AttentionBias):
    """
    A block-diagonal mask that can be passed as ``attn_bias``
    argument to :attr:`xformers.ops.memory_efficient_attention`.

    .. figure:: /_static/block_diag_bias.png

        This bias can be used to handle a batch of sequences of
        different lengths, via :attr:`BlockDiagonalMask.from_tensor_list`

    :Example:

    .. code-block:: python

        import torch
        from xformers.ops import fmha

        K = 16
        dtype = torch.float16
        device = "cuda"
        list_x = [
            torch.randn([1, 3, 1, K], dtype=dtype, device=device),
            torch.randn([1, 6, 1, K], dtype=dtype, device=device),
            torch.randn([1, 2, 1, K], dtype=dtype, device=device),
        ]
        attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(list_x)
        linear = torch.nn.Linear(K, K * 3).to(device=device, dtype=dtype)

        q, k, v = linear(x).reshape([1, -1, 1, 3, K]).unbind(-2)
        out = fmha.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        list_out = attn_bias.split(out)
        print(list_out[0].shape)  # [1, 3, 1, K]
        assert tuple(list_out[0].shape) == (1, 3, 1, K)

    """

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
        """Materialize the attention bias - for debugging & testing"""
        assert shape[-1] == self.k_seqinfo.cu_seqlen_py[-1], (
            shape[-1],
            self.k_seqinfo.cu_seqlen_py[-1],
        )
        assert shape[-2] == self.q_seqinfo.cu_seqlen_py[-1], (
            shape[-2],
            self.q_seqinfo.cu_seqlen_py[-1],
        )
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
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
        q_seqlen: Union[Sequence[int], torch.Tensor],
        kv_seqlen: Optional[Union[Sequence[int], torch.Tensor]] = None,
    ) -> "BlockDiagonalMask":
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors lengths for query and key/value.

        Args:
            q_seqlen (Union[Sequence[int], torch.Tensor]): List or tensor of sequence lengths for query tensors
            kv_seqlen (Union[Sequence[int], torch.Tensor], optional): List or tensor of sequence lengths for key/value.
            Defaults to ``q_seqlen``.
        Returns:
            BlockDiagonalMask
        """
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
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors, and returns the tensors
        concatenated on the sequence length dimension

        .. figure:: /_static/block_diag_cat_split.png

            See also :attr:`BlockDiagonalMask.split` to split the returned
            :attr:`torch.Tensor` back to a list of tensors of varying sequence length

        Args:
            tensors (Sequence[torch.Tensor]): A list of tensors of shape ``[B, M_i, *]``.
                All tensors should have the same dimension and the same batch size ``B``, but
                they can have different sequence length ``M``.

        Returns:
            Tuple[BlockDiagonalMask, torch.Tensor]: The corresponding bias for the attention
            along with `tensors` concatenated on the sequence length dimension, with shape ``[1, sum_i{M_i}, *]``
        """
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
        """The inverse operation of :attr:`BlockDiagonalCausalMask.from_tensor_list`

        Args:
            tensor (torch.Tensor): Tensor of tokens of shape ``[1, sum_i{M_i}, *]``

        Returns:
            Sequence[torch.Tensor]: A list of tokens with possibly different sequence lengths
        """
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


@dataclass
class BlockDiagonalCausalWithOffsetPaddedKeysMask(AttentionBias):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`,
    except an offset on causality is allowed for each block and we support padding for k/v
    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _PaddedSeqLenInfo
    causal_diagonal: Optional[torch.Tensor] = None

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        offset: int = 0,
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
        return torch.triu(tensor, diagonal=1 + offset).to(dtype)  # type: ignore

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        assert shape[-1] == self.k_seqinfo.cu_seqlen_py[-1]
        assert shape[-2] == self.q_seqinfo.cu_seqlen_py[-1]
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
        ):
            mask[q_start:q_end, k_start:k_end] = self._create_block_mask(
                (q_end - q_start, k_end - k_start),
                offset=0
                if self.causal_diagonal is None
                else int(self.causal_diagonal[i].item()),
                dtype=dtype,
                device=device,
            )
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Union[Sequence[int], torch.Tensor],
        kv_padding: int,
        kv_seqlen: Union[Sequence[int], torch.Tensor],
        causal_diagonal: Optional[torch.Tensor] = None,
    ) -> "BlockDiagonalCausalWithOffsetPaddedKeysMask":
        """Creates a :attr:`BlockDiagonalCausalWithOffsetPaddedKeysMask` from a list of tensors lengths for query and key/value.

        Args:
            q_seqlen (Union[Sequence[int], torch.Tensor]): List or tensor of sequence lengths for query tensors
            kv_seqlen (Union[Sequence[int], torch.Tensor], optional): List or tensor of sequence lengths for key/value.
            Defaults to ``q_seqlen``.
            causal_diagonal (torch.Tensor, optional): tensor of sequence positions for causal masking
            padding (int, optional): offset for start positions in kv
        Returns:
            BlockDiagonalCausalWithOffsetPaddedKeysMask
        """
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, kv_padding)
        return cls(
            q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, causal_diagonal=causal_diagonal
        )
