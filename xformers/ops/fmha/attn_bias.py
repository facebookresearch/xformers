# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""
This file contains biases that can be used as the `attn_bias` argument in
:attr:`xformers.ops.memory_efficient_attention`.
Essentially, a bias is a Tensor which will be added to the ``Q @ K.t`` before
computing the ``softmax``.


The goal of having custom made classes (instead of dense tensors) is that
we want to avoid having to load the biases from memory in the kernel, for
performance reasons. We also want to be able to know before-hand which
parts of the attention matrix we will need to compute (eg causal masks).


Some very common biases are LowerTriangularMask and BlockDiagonalMask.
"""

import math
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    ClassVar,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import torch


def _to_device(t: torch.Tensor, device: torch.device):
    if t.device == device:
        return t
    if device == torch.device("cpu"):
        return t.to(device)

    return t.to(device, non_blocking=True)


def _to_device_tensor(seq: Sequence[int], dtype: torch.dtype, device: torch.device):
    if device == torch.device("cpu"):
        return torch.tensor(seq, dtype=dtype)

    return torch.tensor(seq, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


class AttentionBias:
    """Base class for a custom bias that can be applied \
        as the attn_bias argument in
    :attr:`xformers.ops.memory_efficient_attention`.

    That function has the ability to add a tensor, the
    attention bias, to the QK^T matrix before it is used
    in the softmax part of the attention calculation.
    The attention bias tensor with shape
    (B or 1, n_queries, number of keys)
    can be given as the attn_bias input.
    The most common use case is for an attention bias is
    to contain only zeros and negative infinities, which forms
    a mask so that some queries only attend to some keys.

    Children of this class define alternative things which can
    be used as the attn_bias input to define an attention bias which
    forms such a mask, for some common cases.

    When using an :attr:`xformers.ops.AttentionBias`
    instead of a :attr:`torch.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.

    See:

    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularFromBottomRightMask`
    - :attr:`xformers.ops.fmha.attn_bias.LowerTriangularMaskWithTensorBias`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`
    - :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`

    """

    HOLDS_DENSE_TENSOR = False

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


def _get_default_bias_device(device: Optional[torch.device] = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.mtia.is_available():
            return torch.device("mtia")
        return torch.device("cpu")
    return device


def _materialize_causal_mask(
    shape: Tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    *,
    window_size: Optional[int] = None,
    from_bottomright: bool = False,
) -> torch.Tensor:
    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    tensor = torch.full(  # type: ignore
        shape,
        dtype=create_as,
        fill_value=1,
        device=device,
    )

    num_queries, num_keys = shape[-2:]
    shift = 0
    if from_bottomright:
        shift = num_keys - num_queries

    mask = torch.tril(tensor, diagonal=shift).to(dtype)  # type: ignore
    if window_size is not None:
        mask = torch.triu(mask, diagonal=shift - window_size + 1)
    mask = torch.log(mask)
    return mask.to(dtype)


@dataclass
class LocalAttentionFromBottomRightMask(AttentionBias):
    """
    A local attention mask

    The query at position :math:`q` can attend the key at position :math:`k` if
    :math:`q - window\\_left <= k + s <= q + window\\_right`

    With :math:`s = num\\_queries - num\\_keys`

    :Example:

    .. code-block:: python

        import torch
        from xformers.ops import fmha

        bias = fmha.attn_bias.LocalAttentionFromBottomRightMask(window_left=1, window_right=2)
        print(bias.materialize(shape=(4, 4)).exp())
        print(bias.materialize(shape=(4, 5)).exp())

    .. code-block:: text

        # 4x4
        tensor([[1., 1., 1., 0.],
                [1., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 0., 1., 1.]])

        # 4x5
        tensor([[1., 1., 1., 1., 0.],
                [0., 1., 1., 1., 1.],
                [0., 0., 1., 1., 1.],
                [0., 0., 0., 1., 1.]])

    :Illustration:

    .. figure:: /_static/local_attn.png
        :width: 240px

        The total window size is :math:`window\\_left + 1 + window\\_right`
    """

    window_left: int
    window_right: int

    def to(self, device) -> "LocalAttentionFromBottomRightMask":
        return self

    def __post_init__(self) -> None:
        if self.window_left < 0:
            raise ValueError(
                "Invalid window value passed to "
                "`LocalAttentionFromBottomRightMask`: expected"
                f"`window_left > 0` but got window_left={self.window_left}"
            )
        if self.window_right < 0:
            raise ValueError(
                "Invalid window value passed to "
                "`LocalAttentionFromBottomRightMask`: expected"
                f"`window_right > 0` but got window_right={self.window_right}"
            )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        create_as = dtype if dtype is not torch.bfloat16 else torch.float32
        mask = torch.full(  # type: ignore
            shape,
            dtype=create_as,
            fill_value=1,
            device=device,
        )

        num_queries, num_keys = shape[-2:]
        shift = num_keys - num_queries

        mask = torch.triu(mask, diagonal=shift - self.window_left)
        mask = torch.tril(mask, diagonal=shift + self.window_right)
        mask = torch.log(mask)
        return mask.to(dtype)


class LowerTriangularFromBottomRightMask(AttentionBias):
    """
    A causal masking.

    This mask is exactly the same as :attr:`LowerTriangularMask` when there is
    the same number of queries and keys.
    When the number of queries is different from the number of keys,
    it is a triangular mask shifted so that the last query can attend to
    the last key.
    In other words, a query Q cannot attend to a key which is nearer the
    final key than Q is to the final query.


    .. figure:: /_static/causal_bottom_right.png

        The difference between :attr:`LowerTriangularMask` (left) and
        :attr:`LowerTriangularFromBottomRightMask` (right). They become
        equivalent if the number of queries equals the number of keys.
    """

    def to(self, device: torch.device) -> "LowerTriangularFromBottomRightMask":
        assert (
            type(self) is LowerTriangularFromBottomRightMask
        ), "Please implement in subclass"
        return self

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask(
            shape, dtype=dtype, device=device, from_bottomright=True
        )

    def make_local_attention(
        self, window_size: int
    ) -> "LowerTriangularFromBottomRightLocalAttentionMask":
        """
        Create a new bias which combines local + causal attention.

        See :attr:`LowerTriangularFromBottomRightLocalAttentionMask`
        """
        return LowerTriangularFromBottomRightLocalAttentionMask(window_size)


@dataclass
class LowerTriangularFromBottomRightLocalAttentionMask(
    LowerTriangularFromBottomRightMask
):
    """
    A mask that combines both :attr:`LowerTriangularFromBottomRightMask` and
    local attention.

    A query whose distance from the final query is X cannot attend to a key
    whose distance to the final key is either of:

    * less than X (i.e. "causal attention", same as :attr:`LowerTriangularFromBottomRightMask`)
    * greater than or equal to X + window_size (i.e. "local attention")


    .. figure:: /_static/causal_bottom_right_local.png

        The mask from :attr:`LowerTriangularFromBottomRightLocalAttentionMask`.
        The green area is calculated, and the grey area is masked out.
    """

    _window_size: int

    def to(
        self, device: torch.device
    ) -> "LowerTriangularFromBottomRightLocalAttentionMask":
        assert (
            type(self) is LowerTriangularFromBottomRightLocalAttentionMask
        ), "Please implement in subclass"
        return self

    def __post_init__(self) -> None:
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask(
            shape,
            dtype=dtype,
            device=device,
            window_size=self._window_size,
            from_bottomright=True,
        )


@dataclass
class _SeqLenInfo:
    """
    (Internal) Represents the division of a dimension into blocks.

    For example, to represents a dimension of length 7 divided into
    three blocks of lengths 2, 3 and 2, use `from_seqlength([2, 3, 2])`.
    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 2, 5, 7]
        seqstart: torch.IntTensor([0, 2, 5, 7])
    """

    seqstart: torch.Tensor
    max_seqlen: int
    min_seqlen: int
    seqstart_py: List[int]

    def to(self, device: torch.device) -> "_SeqLenInfo":
        assert type(self) is _SeqLenInfo, "Please implement in subclass"
        if self.seqstart.device == device:
            return self
        return _SeqLenInfo(
            seqstart=_to_device(self.seqstart, device),
            max_seqlen=self.max_seqlen,
            min_seqlen=self.min_seqlen,
            seqstart_py=self.seqstart_py,
        )

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def _get_seqstart(
        cls, seqlens: Iterable[int], *, device: torch.device
    ) -> Tuple[int, int, List[int], torch.Tensor]:
        """
        Given sequence lengths, returns the min/max value and the sequence start
        positions (offsets), with first element being 0 (returned in list and Tensor).
        """

        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = [0]
        max_seqlen = -1
        min_seqlen = -1
        for seqlen in seqlens:
            min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
        seqstart = _to_device_tensor(seqstart_py, dtype=torch.int32, device=device)

        return (min_seqlen, max_seqlen, seqstart_py, seqstart)

    @classmethod
    def from_seqlens(
        cls, seqlens: Iterable[int], *, device: Optional[torch.device] = None
    ) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        device = _get_default_bias_device(device)
        min_seqlen, max_seqlen, seqstart_py, seqstart = cls._get_seqstart(
            seqlens, device=device
        )

        return cls(
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=seqstart,
            seqstart_py=seqstart_py,
        )

    def from_seqlens_inplace(self, seqlens: Iterable[int]) -> None:
        """
        Perform in-place update. You can only update with the same shape.
        Can be useful with CUDA graphs.
        """
        min_seqlen, max_seqlen, seqstart_py, seqstart = self._get_seqstart(
            seqlens, device=self.seqstart.device
        )

        assert len(seqstart_py) == len(self.seqstart_py), (
            f"Old / New len {len(self.seqstart_py)} / {len(seqstart)}, "
            f"Contents {self.seqstart_py} / {seqstart}"
        )
        assert self.max_seqlen >= max_seqlen, (
            f"For inplace update, new max_seqlen {max_seqlen} "
            f"cannot exceed the previous max_seqlen {self.max_seqlen}"
        )
        for i in range(len(seqstart_py)):
            self.seqstart_py[i] = seqstart_py[i]
        self.seqstart.copy_(seqstart, non_blocking=True)

    def split(
        self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        if self.seqstart_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(
                f"Invalid `torch.Tensor` of shape {x.shape}, expected format "
                f"(B, M, *) with B=1 and M={self.seqstart_py[-1]}\n"
                f" seqstart: {self.seqstart_py}"
            )
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.seqstart_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(
                self.seqstart_py[it + batch_size] - self.seqstart_py[it]
            )
            it += batch_size
        return [
            tensor.reshape([bs, -1, *tensor.shape[2:]])
            for bs, tensor in zip(batch_sizes, x.split(split_chunks, dim=1))
        ]


@dataclass
class _PaddedSeqLenInfo(_SeqLenInfo):
    """
    (Internal)  Represents the division of a dimension into blocks which are
    padded out to the same total length.

    For example, to represent a dimension of length 12 with space for
    three blocks of length 4, but where the occupied lengths are
    2, 3 and 2, use `from_seqlens_padded([2, 3, 2], 4)`.

    The layout along the dimension is

     0 ─►  block 0
           block 0
           <space>
           <space>
     4 ─►  block 1
           block 1
           block 1
           <space>
     8 ─►  block 2
           block 2
           <space>
           <space>
    12 ─►

    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 4, 8, 12]
        seqstart: torch.IntTensor([0, 4, 8, 12])
        seqlen_py: [2, 3, 2]
        seqlen: torch.IntTensor([2, 3, 2])
        padding: 4
    """

    seqlen: torch.Tensor
    seqlen_py: List[int]
    padding: int
    # From parent: seqstart[i] contains the start position
    # of the i-th sequence
    # seqstart: torch.Tensor

    def __post_init__(self) -> None:
        assert len(self.seqstart_py) == len(self.seqlen_py) + 1

    def to(self, device: torch.device) -> "_PaddedSeqLenInfo":
        assert type(self) is _PaddedSeqLenInfo, "Please implement in subclass"
        if self.seqlen.device == device:
            return self
        return _PaddedSeqLenInfo(
            # _SeqLenInfo
            seqstart=_to_device(self.seqstart, device),
            max_seqlen=self.max_seqlen,
            min_seqlen=self.min_seqlen,
            seqstart_py=self.seqstart_py,
            # _PaddedSeqLenInfo
            seqlen=_to_device(self.seqlen, device),
            seqlen_py=self.seqlen_py,
            padding=self.padding,
        )

    def intervals(self) -> Iterable[Tuple[int, int]]:
        for (start, _), length in zip(super().intervals(), self.seqlen_py):
            yield start, start + length

    @classmethod
    def from_seqlens(
        cls, seqlens: Iterable[int], *, device: Optional[torch.device] = None
    ) -> "_SeqLenInfo":
        raise RuntimeError(
            "Use either `_SeqLenInfo.from_seqlens` or `_PaddedSeqLenInfo.from_seqlens_padded`"
        )

    @classmethod
    def from_seqlens_padded(
        cls,
        seqlens: Sequence[int],
        padding: int,
        *,
        device: Optional[torch.device] = None,
    ) -> "_PaddedSeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        seqstart = padding * torch.arange(batch_size)
        """
        assert not isinstance(seqlens, torch.Tensor)
        assert all(
            seqlen <= padding for seqlen in seqlens
        ), f"Seqlens {seqlens} Padding {padding}"
        device = _get_default_bias_device(device)
        seqstart_py = list(range(0, len(seqlens) * padding + 1, padding))
        seqlen = _to_device_tensor(seqlens, dtype=torch.int32, device=device)
        return cls(
            seqlen=seqlen,
            seqlen_py=seqlens if isinstance(seqlens, list) else list(seqlens),
            max_seqlen=max(seqlens),
            min_seqlen=min(seqlens),
            seqstart=_to_device_tensor(seqstart_py, dtype=torch.int32, device=device),
            seqstart_py=seqstart_py,
            padding=padding,
        )

    def from_seqlens_padded_inplace(self, seqlens: Sequence[int]) -> None:
        """
        Perform in-place update. You can only update with the same shape.
        Can be useful with CUDA graphs.
        Note: we don't update padding because they would have been already baked
        into CUDA graphs during the generation.
        """
        assert not isinstance(seqlens, torch.Tensor)
        assert all(
            seqlen <= self.padding for seqlen in seqlens
        ), f"Seqlens {seqlens} Padding {self.padding}"
        seqlen_tensor = torch.tensor(seqlens, dtype=torch.int32)

        assert len(self.seqlen_py) == len(seqlens), (
            f"Old/New len {len(self.seqlen_py)} / {len(seqlens)}, "
            f"Contents {self.seqlen_py} / {seqlens}"
        )
        assert self.max_seqlen >= max(seqlens), (
            f"For inplace update, new max_seqlen {max(seqlens)} "
            f"cannot exceed the previous max_seqlen {self.max_seqlen}"
        )

        for i in range(len(self.seqlen_py)):
            self.seqlen_py[i] = seqlens[i]

        self.seqlen.copy_(seqlen_tensor, non_blocking=True)

    def split(
        self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        raise NotImplementedError("_PaddedSeqLenInfo.split")


@dataclass
class _GappySeqInfo(_SeqLenInfo):
    """
    (Internal) Flexible equivalent of _PaddedSeqLenInfo. There are two
    distinct semantics.

    (1) For non-paged masks:
    Represents the division of a dimension into blocks which are
    anywhere. Each just has a start and a length. The final start is the total
    length of the dimension.

    For example, to represent a dimension of length 14 like follows with
    three occupied lengths of
    6, 3 and 1, use `from_seqlens_padded([0, 7, 12, 14], [6, 3, 1])`.

    The layout along the dimension is

     0 ─►  block 0
           block 0
           block 0
           block 0
     4 ─►  block 0
           block 0
           <space>
           block 1
     8 ─►  block 1
           block 1
           <space>
           <space>
     12 ─► block 2
           <space>

    The members will be:
        max_seqlen: 6
        min_seqlen: 1
        seqstart_py: [0, 7, 12, 14]
        seqstart: torch.IntTensor([0, 7, 12, 14])
        seqlen_py: [6, 3, 1]
        seqlen: torch.IntTensor([6, 3, 1])

    (2) For paged masks:
    The notional space is divided into batch-size-many blocks.
    seqstart and seqstart_py is an offset in the block, not in
    the whole space, and the extra last element is not important.
    And seqlen is the index of the last key in the block.
    Otherwise as above.
    """

    seqlen: torch.Tensor
    seqlen_py: Sequence[int]
    # From parent: seqstart[i] contains the start position
    # of the i-th sequence
    # seqstart: torch.Tensor

    def to(self, device: torch.device) -> "_GappySeqInfo":
        assert type(self) is _GappySeqInfo, "Please implement in subclass"
        if self.seqlen.device == device:
            return self
        return _GappySeqInfo(
            # _SeqLenInfo
            seqstart=_to_device(self.seqstart, device),
            max_seqlen=self.max_seqlen,
            min_seqlen=self.min_seqlen,
            seqstart_py=self.seqstart_py,
            # _GappySeqInfo
            seqlen=_to_device(self.seqlen, device),
            seqlen_py=self.seqlen_py,
        )

    def intervals(self) -> Iterable[Tuple[int, int]]:
        for (start, _), length in zip(super().intervals(), self.seqlen_py):
            yield start, start + length

    @classmethod
    def from_seqlens(
        cls, seqlens: Iterable[int], *, device: Optional[torch.device] = None
    ) -> "_SeqLenInfo":
        raise NotImplementedError()

    @classmethod
    def from_seqlens_gappy(
        cls,
        seqstarts: Sequence[int],
        seqlens: Sequence[int],
        paged: bool,
        *,
        device: torch.device,
    ) -> "_GappySeqInfo":
        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = list(seqstarts)
        if len(seqlens) == 0:
            raise ValueError("No elements")
        if len(seqstarts) - len(seqlens) != 1:
            raise ValueError(
                f"len(seqstarts)={seqstarts} should be len(seqlens)={seqlens}"
            )
        max_seqlen = max(seqlens)
        min_seqlen = min(seqlens)
        if paged:
            seqstart_py.append(-1)
            seqlens = [i + j for i, j in zip(seqstart_py, seqlens)]
        seqlen = _to_device_tensor(seqlens, dtype=torch.int32, device=device)
        return cls(
            seqlen=seqlen,
            seqlen_py=seqlens,
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=_to_device_tensor(seqstart_py, dtype=torch.int32, device=device),
            seqstart_py=seqstart_py,
        )

    def split(
        self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        raise NotImplementedError("_GappySeqInfo.split")


@dataclass
class BlockDiagonalMask(AttentionBias):
    """
    A block-diagonal mask that can be passed as ``attn_bias``
    argument to :attr:`xformers.ops.memory_efficient_attention`.

    Queries and Keys are each divided into the same number of blocks.
    Queries in block i only attend to keys in block i.

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

    def to(self, device) -> "BlockDiagonalMask":
        assert type(self) is BlockDiagonalMask, "Please implement in subclass"
        return BlockDiagonalMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            _batch_sizes=self._batch_sizes,
        )

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
        assert shape[-1] == self.k_seqinfo.seqstart_py[-1], (
            shape[-1],
            self.k_seqinfo.seqstart_py[-1],
        )
        assert shape[-2] == self.q_seqinfo.seqstart_py[-1], (
            shape[-2],
            self.q_seqinfo.seqstart_py[-1],
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
        q_seqlen: Sequence[int],
        kv_seqlen: Optional[Sequence[int]] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> "BlockDiagonalMask":
        """Creates a :attr:`BlockDiagonalMask` from a list of tensors lengths for query and key/value.

        Args:
            q_seqlen (Union[Sequence[int], torch.Tensor]): List or tensor of sequence lengths for query tensors
            kv_seqlen (Union[Sequence[int], torch.Tensor], optional): List or tensor of sequence lengths for key/value.
                    (Defaults to ``q_seqlen``.)
        Returns:
            BlockDiagonalMask
        """
        device = _get_default_bias_device(device)
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen, device=device)
        if kv_seqlen is None or q_seqlen == kv_seqlen:
            k_seqinfo = q_seqinfo
        else:
            k_seqinfo = _SeqLenInfo.from_seqlens(kv_seqlen, device=device)
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
            (
                torch.cat([x.reshape([1, -1, *x.shape[2:]]) for x in tensors_v], dim=1)
                if tensors_v is not None
                else None
            ),
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

    def make_causal_from_bottomright(self) -> "BlockDiagonalCausalFromBottomRightMask":
        """Makes each block causal with a possible non-causal prefix"""
        return BlockDiagonalCausalFromBottomRightMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
        )

    def make_local_attention(
        self, window_size: int
    ) -> "BlockDiagonalCausalLocalAttentionMask":
        """Experimental: Makes each block causal with local attention"""
        return BlockDiagonalCausalLocalAttentionMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
            _window_size=window_size,
        )

    def make_local_attention_from_bottomright(
        self, window_size: int
    ) -> "BlockDiagonalCausalLocalAttentionFromBottomRightMask":
        """Experimental: Makes each block causal with local attention, start from bottom right"""
        return BlockDiagonalCausalLocalAttentionFromBottomRightMask(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=self.k_seqinfo,
            _batch_sizes=self._batch_sizes,
            _window_size=window_size,
        )


@dataclass
class BlockDiagonalCausalMask(BlockDiagonalMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal.

    Queries and Keys are each divided into the same number of blocks.
    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is farther from the initial key in block i than Q
    is from the initial query in block i.
    """

    def to(self, device) -> "BlockDiagonalCausalMask":
        assert type(self) is BlockDiagonalCausalMask, "Please implement in subclass"
        return BlockDiagonalCausalMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            _batch_sizes=self._batch_sizes,
        )

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
class BlockDiagonalCausalFromBottomRightMask(BlockDiagonalMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`, except that each block is causal.
    This mask allows for a non-causal prefix
    NOTE: Each block should have `num_keys >= num_queries` otherwise the forward pass is not
    defined (softmax of vector of `-inf` in the attention)

    Queries and keys are each divided into the same number of blocks.
    A query Q in block i cannot attend to a key which is not in block i,
    nor one which nearer the final key in block i than Q is to the
    final query in block i.
    """

    def to(self, device) -> "BlockDiagonalCausalFromBottomRightMask":
        assert (
            type(self) is BlockDiagonalCausalFromBottomRightMask
        ), "Please implement in subclass"
        return BlockDiagonalCausalFromBottomRightMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            _batch_sizes=self._batch_sizes,
        )

    def __post_init__(self) -> None:
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
        ):
            num_queries = q_end - q_start
            num_keys = k_end - k_start
            if num_keys < num_queries:
                raise ValueError(
                    f"Block #{i} has num_keys={num_keys} and num_queries={num_queries}."
                    " Expected `num_keys >= num_queries`"
                )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return LowerTriangularFromBottomRightMask().materialize(
            shape=shape, dtype=dtype, device=device
        )


@dataclass
class BlockDiagonalPaddedKeysMask(AttentionBias):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`,
    except we support padding for k/v

    The keys and values are divided into blocks which are padded out to
    the same total length.
    For example, if there is space for 12 keys, for three blocks of
    max length 4, but we only want to use the first 2, 3 and 2
    of each block, use `kv_padding=4` and `kv_seqlens=[2, 3, 2]`.
    The queries are divided into blocks, without padding, of lengths given by
    q_seqlen.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is not in use (i.e. in the padded area).
    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _PaddedSeqLenInfo

    def to(self, device) -> "BlockDiagonalPaddedKeysMask":
        assert type(self) is BlockDiagonalPaddedKeysMask, "Please implement in subclass"
        return BlockDiagonalPaddedKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
        )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return torch.zeros([1], device=device, dtype=dtype)

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        if shape[-1] != self.k_seqinfo.seqstart_py[-1]:
            raise ValueError("k shapes wrong")
        if shape[-2] != self.q_seqinfo.seqstart_py[-1]:
            raise ValueError("q shapes wrong")
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
        q_seqlen: Sequence[int],
        kv_padding: int,
        kv_seqlen: Sequence[int],
        causal_diagonal: Any = None,
        *,
        device: Optional[torch.device] = None,
    ) -> "BlockDiagonalPaddedKeysMask":
        """Creates a :attr:`BlockDiagonalPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            BlockDiagonalPaddedKeysMask
        """
        device = _get_default_bias_device(device)
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen, device=device)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(
            kv_seqlen, kv_padding, device=device
        )
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    def make_paged(
        self,
        block_tables: torch.Tensor,
        page_size: int,
        paged_type: Type["PagedBlockDiagonalPaddedKeysMask"],
    ) -> "PagedBlockDiagonalPaddedKeysMask":
        paged_bias = paged_type(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=_PaddedSeqLenInfo(
                seqstart=self.k_seqinfo.seqstart,
                seqstart_py=self.k_seqinfo.seqstart_py,
                seqlen=self.k_seqinfo.seqlen,
                seqlen_py=self.k_seqinfo.seqlen_py,
                padding=block_tables.shape[1] * page_size,
                max_seqlen=self.k_seqinfo.max_seqlen,
                min_seqlen=self.k_seqinfo.min_seqlen,
            ),
            block_tables=block_tables,
            page_size=page_size,
        )
        return paged_bias


@dataclass
class BlockDiagonalCausalWithOffsetPaddedKeysMask(BlockDiagonalPaddedKeysMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`,
    except an offset on causality is allowed for each block and we support padding for k/v

    The keys and values are divided into blocks which are padded out to
    the same total length.
    For example, if there is space for 12 keys, for three blocks of
    max length 4, but we only want to use the first 2, 3 and 2
    of each block, use `kv_padding=4` and `kv_seqlens=[2, 3, 2]`.
    The queries are divided into blocks, without padding, of lengths given by
    q_seqlen.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is not in use (i.e. in the padded area),
    nor one which is nearer to the final key in block i
    than Q is to the final query in block i.
    """

    causal_diagonal: Any = None  # unused. Exists for BC only.

    def to(self, device) -> "BlockDiagonalCausalWithOffsetPaddedKeysMask":
        assert (
            type(self) is BlockDiagonalCausalWithOffsetPaddedKeysMask
        ), "Please implement in subclass"
        return BlockDiagonalCausalWithOffsetPaddedKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
        )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return LowerTriangularFromBottomRightMask().materialize(
            shape=shape, dtype=dtype, device=device
        )

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_padding: int,
        kv_seqlen: Sequence[int],
        causal_diagonal: Any = None,
        *,
        device: Optional[torch.device] = None,
    ) -> "BlockDiagonalCausalWithOffsetPaddedKeysMask":
        """Creates a :attr:`BlockDiagonalCausalWithOffsetPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            BlockDiagonalCausalWithOffsetPaddedKeysMask
        """
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        device = _get_default_bias_device(device)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen, device=device)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(
            kv_seqlen, kv_padding, device=device
        )
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)


@dataclass
class BlockDiagonalLocalAttentionPaddedKeysMask(BlockDiagonalPaddedKeysMask):
    """
    Like :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalLocalAttentionPaddedKeysMask`,
    except that this is non-causal.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is not in use (i.e. in the padded area),
    nor one whose distance to the final key in block i
    is more than window_left further or window_right nearer
    than Q is to the final query in block i.

    A query attends to at most window_left + window_right - 1 keys.

    NOTE that if window_right is 0, then this is like a
    BlockDiagonalCausalLocalAttentionPaddedKeysMask whose window_size is equal to
    window_left - 1.
    """

    window_left: int
    window_right: int

    def to(self, device) -> "BlockDiagonalLocalAttentionPaddedKeysMask":
        assert (
            type(self) is BlockDiagonalLocalAttentionPaddedKeysMask
        ), "Please implement in subclass"
        return BlockDiagonalLocalAttentionPaddedKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            window_left=self.window_left,
            window_right=self.window_right,
        )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return LocalAttentionFromBottomRightMask(
            window_left=self.window_left, window_right=self.window_right
        ).materialize(shape=shape, dtype=dtype, device=device)

    @classmethod
    def from_seqlens_local(
        cls,
        q_seqlen: Sequence[int],
        kv_padding: int,
        kv_seqlen: Sequence[int],
        window_left: int,
        window_right: int,
    ) -> "BlockDiagonalLocalAttentionPaddedKeysMask":
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, kv_padding)
        return cls(
            q_seqinfo=q_seqinfo,
            k_seqinfo=k_seqinfo,
            window_left=window_left,
            window_right=window_right,
        )


@dataclass
class BlockDiagonalCausalLocalAttentionPaddedKeysMask(BlockDiagonalPaddedKeysMask):
    """
    Like :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask`,
    except with a window size.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is not in use (i.e. in the padded area),
    nor one which is nearer to the final key in block i
    than Q is to the final query in block i, nor one that is at least
    window_size further from the final key in block i than Q is
    to the final query in block i.
    """

    _window_size: int

    def to(self, device) -> "BlockDiagonalCausalLocalAttentionPaddedKeysMask":
        assert (
            type(self) is BlockDiagonalCausalLocalAttentionPaddedKeysMask
        ), "Please implement in subclass"
        return BlockDiagonalCausalLocalAttentionPaddedKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            _window_size=self._window_size,
        )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask(
            shape=shape,
            dtype=dtype,
            device=device,
            window_size=self._window_size,
            from_bottomright=True,
        )

    @classmethod
    def from_seqlens_local(
        cls,
        q_seqlen: Sequence[int],
        kv_padding: int,
        kv_seqlen: Sequence[int],
        window_size: int,
    ) -> "BlockDiagonalCausalLocalAttentionPaddedKeysMask":
        assert kv_seqlen is None or len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(kv_seqlen, kv_padding)
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo, _window_size=window_size)


@dataclass
class PagedBlockDiagonalPaddedKeysMask(AttentionBias):
    """
    Same as BlockDiagonalPaddedKeysMask, but for paged attention.
    block_tables has shape [batch_size, max_num_pages] and K/V have shape
    [1, max_num_pages * page_size, num_heads, head_dim]
    or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _PaddedSeqLenInfo
    block_tables: torch.Tensor
    page_size: int

    _UNPAGED_TYPE: ClassVar[Type[BlockDiagonalPaddedKeysMask]] = (
        BlockDiagonalPaddedKeysMask
    )

    def to(self, device: torch.device) -> "PagedBlockDiagonalPaddedKeysMask":
        assert (
            type(self) is PagedBlockDiagonalPaddedKeysMask
        ), "Please implement in subclass"
        return PagedBlockDiagonalPaddedKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            block_tables=_to_device(self.block_tables, device),
            page_size=self.page_size,
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        # First create a non-paged mask, then cut individual pages and
        # copy them to their places in the physical mask, using block tables

        max_row_len = self.block_tables.shape[1] * self.page_size
        bias_nonpaged = self._UNPAGED_TYPE(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=_PaddedSeqLenInfo.from_seqlens_padded(
                self.k_seqinfo.seqlen_py, max_row_len
            ),
        )
        mask_nonpaged = bias_nonpaged.materialize(shape, dtype, device)

        n_used_blocks = cast(int, self.block_tables.max().item() + 1)
        max_physical_len = n_used_blocks * self.page_size
        mask_paged = torch.empty(
            mask_nonpaged.shape[:-1] + (max_physical_len,), dtype=dtype, device=device
        )
        mask_paged.fill_(-math.inf)
        for b, (q_start, q_end) in enumerate(self.q_seqinfo.intervals()):
            for logical_page_idx in range(self.block_tables.shape[1]):
                physical_page_idx = cast(
                    int, self.block_tables[b][logical_page_idx].item()
                )
                k_logical_start = b * max_row_len + logical_page_idx * self.page_size
                k_logical_end = k_logical_start + self.page_size
                k_physical_start = physical_page_idx * self.page_size
                k_physical_end = k_physical_start + self.page_size
                mask_paged[..., q_start:q_end, k_physical_start:k_physical_end] = (
                    mask_nonpaged[..., q_start:q_end, k_logical_start:k_logical_end]
                )
        return mask_paged

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_seqlen: Sequence[int],
        block_tables: torch.Tensor,
        page_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> "PagedBlockDiagonalPaddedKeysMask":
        """Creates a :attr:`PagedBlockDiagonalPaddedKeysMask` from a list of tensor
        lengths for query and key/value.

        Args:
            q_seqlen (Sequence[int]): List or tensor of sequence lengths for query tensors
            kv_padding (int): Padding for k/v - also an upperbound on each individual key length
            kv_seqlen (Sequence[int]): List or tensor of sequence lengths for key/value.
            causal_diagonal: unused, for BC only
        Returns:
            PagedBlockDiagonalPaddedKeysMask
        """
        assert len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        device = _get_default_bias_device(device)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen, device=device)
        k_seqinfo = _PaddedSeqLenInfo.from_seqlens_padded(
            kv_seqlen, padding=block_tables.shape[1] * page_size, device=device
        )
        return cls(
            q_seqinfo=q_seqinfo,
            k_seqinfo=k_seqinfo,
            block_tables=block_tables,
            page_size=page_size,
        )


@dataclass
class PagedBlockDiagonalCausalWithOffsetPaddedKeysMask(
    PagedBlockDiagonalPaddedKeysMask
):
    """
    Same as BlockDiagonalCausalWithOffsetPaddedKeysMask, but for paged attention.
    block_tables has shape [batch_size, max_num_pages] and K/V have shape
    [1, max_num_pages * page_size, num_heads, head_dim]
    or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
    """

    _UNPAGED_TYPE = BlockDiagonalCausalWithOffsetPaddedKeysMask

    def to(
        self, device: torch.device
    ) -> "PagedBlockDiagonalCausalWithOffsetPaddedKeysMask":
        assert (
            type(self) is PagedBlockDiagonalCausalWithOffsetPaddedKeysMask
        ), "Please implement in subclass"
        return PagedBlockDiagonalCausalWithOffsetPaddedKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            block_tables=_to_device(self.block_tables, device),
            page_size=self.page_size,
        )


@dataclass
class BlockDiagonalGappyKeysMask(AttentionBias):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalMask`,
    except k/v is gappy.

    A query Q in block i only attends to a key which is in block i.
    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _GappySeqInfo

    def to(self, device: torch.device) -> "BlockDiagonalGappyKeysMask":
        assert type(self) is BlockDiagonalGappyKeysMask, "Please implement in subclass"
        return BlockDiagonalGappyKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        if shape[-1] != self.k_seqinfo.seqstart_py[-1]:
            raise ValueError("k shapes wrong", (shape, self.k_seqinfo))
        if shape[-2] != self.q_seqinfo.seqstart_py[-1]:
            raise ValueError("q shapes wrong", (shape, self.q_seqinfo))
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for (q_start, q_end), (k_start, k_end) in zip(
            self.q_seqinfo.intervals(),
            self.k_seqinfo.intervals(),
        ):
            mask[q_start:q_end, k_start:k_end] = 0
        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_seqstarts: Sequence[int],
        kv_seqlen: Sequence[int],
        *,
        device: Optional[torch.device] = None,
    ) -> "BlockDiagonalGappyKeysMask":
        """Creates a :attr:`BlockDiagonalGappyKeysMask` from a list of tensor
        lengths for query and key/value.
        """
        assert len(q_seqlen) == len(kv_seqlen), (
            q_seqlen,
            kv_seqlen,
        )
        device = _get_default_bias_device(device)
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen, device=device)
        k_seqinfo = _GappySeqInfo.from_seqlens_gappy(
            kv_seqstarts, kv_seqlen, False, device=device
        )
        return cls(q_seqinfo=q_seqinfo, k_seqinfo=k_seqinfo)

    def make_paged(
        self,
        block_tables: torch.Tensor,
        page_size: int,
        notional_padding: int,
        paged_type: Type["PagedBlockDiagonalGappyKeysMask"],
    ) -> AttentionBias:
        """
        Assuming our keys actually live in separate blocks of length
        notional_padding, convert to a Paged version, avoiding GPU syncs.
        """
        if notional_padding % page_size:
            raise ValueError(
                "Notional padding should be divisible by the page size,"
                f" but got {notional_padding=}, {page_size=}."
            )
        max_row_len = block_tables.shape[1] * page_size
        new_seqstarts_py = [
            start - i * notional_padding
            for i, start in enumerate(self.k_seqinfo.seqstart_py[:-1])
        ]
        new_seqstarts_py.append(-1)
        assert all(
            0 <= i < max_row_len for i in new_seqstarts_py[:-1]
        ), f"{max_row_len=} {new_seqstarts_py=}"

        # Sequence info is duplicated on CPU and GPU,
        # but we process them independently to avoid GPU sync.
        batch_size = len(self.k_seqinfo.seqlen_py)
        notional_starts = notional_padding * torch.arange(
            batch_size + 1,
            device=block_tables.device,
            dtype=torch.int32,
        )
        new_seqstarts = self.k_seqinfo.seqstart - notional_starts

        new_seqlens_py = [
            i + j for i, j in zip(new_seqstarts_py, self.k_seqinfo.seqlen_py)
        ]
        new_seqlens = self.k_seqinfo.seqlen + new_seqstarts[:-1]

        k_seqinfo = _GappySeqInfo(
            seqlen=new_seqlens,
            seqlen_py=new_seqlens_py,
            max_seqlen=self.k_seqinfo.max_seqlen,
            min_seqlen=self.k_seqinfo.min_seqlen,
            seqstart=new_seqstarts,
            seqstart_py=new_seqstarts_py,
        )
        assert self.k_seqinfo.max_seqlen <= max_row_len
        paged_bias = paged_type(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=k_seqinfo,
            block_tables=block_tables,
            page_size=page_size,
        )
        return paged_bias


@dataclass
class BlockDiagonalCausalWithOffsetGappyKeysMask(BlockDiagonalGappyKeysMask):
    """
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`,
    except k/v is gappy.

    A query Q in block i cannot attend to a key which is not in block i,
    nor one which is nearer to the final key in block i
    than Q is to the final query in block i.
    """

    def to(self, device: torch.device) -> "BlockDiagonalCausalWithOffsetGappyKeysMask":
        assert (
            type(self) is BlockDiagonalCausalWithOffsetGappyKeysMask
        ), "Please implement in subclass"
        return BlockDiagonalCausalWithOffsetGappyKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        if shape[-1] != self.k_seqinfo.seqstart_py[-1]:
            raise ValueError("k shapes wrong")
        if shape[-2] != self.q_seqinfo.seqstart_py[-1]:
            raise ValueError("q shapes wrong")
        mask = torch.empty(shape[-2:], dtype=dtype, device=device)
        mask.fill_(-math.inf)
        for i, ((q_start, q_end), (k_start, k_end)) in enumerate(
            zip(
                self.q_seqinfo.intervals(),
                self.k_seqinfo.intervals(),
            )
        ):
            mask[
                q_start:q_end, k_start:k_end
            ] = LowerTriangularFromBottomRightMask().materialize(
                shape=(q_end - q_start, k_end - k_start), dtype=dtype, device=device
            )

        for _ in range(len(shape) - 2):
            mask = mask.unsqueeze(0)
        return mask.expand(shape)


@dataclass
class PagedBlockDiagonalGappyKeysMask(AttentionBias):
    """
    Equivalent BlockDiagonalGappyKeysMask, but for paged attention.
    block_tables has shape [batch_size, max_num_pages] and K/V have shape
    [1, max_num_pages * page_size, num_heads, head_dim]
    or [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
    """

    q_seqinfo: _SeqLenInfo
    k_seqinfo: _GappySeqInfo
    block_tables: torch.Tensor
    page_size: int

    _UNPAGED_TYPE: ClassVar[Type[BlockDiagonalGappyKeysMask]] = (
        BlockDiagonalGappyKeysMask
    )

    def to(self, device: torch.device) -> "PagedBlockDiagonalGappyKeysMask":
        assert (
            type(self) is PagedBlockDiagonalGappyKeysMask
        ), "Please implement in subclass"
        return PagedBlockDiagonalGappyKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            block_tables=_to_device(self.block_tables, device),
            page_size=self.page_size,
        )

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """Materialize the attention bias - for debugging & testing"""
        # First create a non-paged mask, then cut individual pages and
        # copy them to their places in the physical mask, using block tables

        max_row_len = self.block_tables.shape[1] * self.page_size
        new_seqstarts = [
            start + i * max_row_len
            for i, start in enumerate(self.k_seqinfo.seqstart_py[:-1])
        ] + [shape[-1]]
        new_seqlens = [
            end - start
            for start, end in zip(self.k_seqinfo.seqstart_py, self.k_seqinfo.seqlen_py)
        ]
        bias_nonpaged = self._UNPAGED_TYPE(
            q_seqinfo=self.q_seqinfo,
            k_seqinfo=_GappySeqInfo.from_seqlens_gappy(
                new_seqstarts,
                new_seqlens,
                False,
                device=torch.device(device),
            ),
        )
        mask_nonpaged = bias_nonpaged.materialize(shape, dtype, device)

        n_used_blocks = cast(int, self.block_tables.max().item() + 1)
        max_physical_len = n_used_blocks * self.page_size
        mask_paged = torch.empty(
            mask_nonpaged.shape[:-1] + (max_physical_len,), dtype=dtype, device=device
        )
        mask_paged.fill_(-math.inf)
        for b, (q_start, q_end) in enumerate(self.q_seqinfo.intervals()):
            for logical_page_idx in range(self.block_tables.shape[1]):
                physical_page_idx = cast(
                    int, self.block_tables[b][logical_page_idx].item()
                )
                k_logical_start = b * max_row_len + logical_page_idx * self.page_size
                k_logical_end = k_logical_start + self.page_size
                k_physical_start = physical_page_idx * self.page_size
                k_physical_end = k_physical_start + self.page_size
                mask_paged[..., q_start:q_end, k_physical_start:k_physical_end] = (
                    mask_nonpaged[..., q_start:q_end, k_logical_start:k_logical_end]
                )
        return mask_paged

    @classmethod
    def from_seqlens(
        cls,
        q_seqlen: Sequence[int],
        kv_seqstarts: Sequence[int],
        kv_seqlen: Sequence[int],
        block_tables: torch.Tensor,
        page_size: int,
        *,
        device: Optional[torch.device] = None,
    ) -> "PagedBlockDiagonalGappyKeysMask":
        """Creates a :attr:`PagedBlockDiagonalGappyKeysMask` from a list of tensor
        lengths for query and key/value.

        Note that unlike :attr:`BlockDiagonalGappyKeysMask`, kv_seqstarts is
        addressing in a different space for each batch element. For example
        if you were doing a BlockDiagonalPaddedKeysMask with two batch
        elements and padding=100, but wanted to change it so that the first
        key is ignored, then you would use BlockDiagonalGappyKeysMask with kv_seqstarts
        [1, 101, 200]. But if you were using PagedBlockDiagonalPaddedKeysMask
        but wanted to ignore the first key, you would provide this function with
        kv_seqstarts = [1, 1].
        """
        assert len(q_seqlen) == len(kv_seqlen) == len(kv_seqstarts), (
            q_seqlen,
            kv_seqlen,
            kv_seqstarts,
        )
        device = block_tables.device if device is None else device
        q_seqinfo = _SeqLenInfo.from_seqlens(q_seqlen, device=device)
        k_seqinfo = _GappySeqInfo.from_seqlens_gappy(
            kv_seqstarts, kv_seqlen, True, device=device
        )
        return cls(
            q_seqinfo=q_seqinfo,
            k_seqinfo=k_seqinfo,
            block_tables=block_tables,
            page_size=page_size,
        )


@dataclass
class PagedBlockDiagonalCausalWithOffsetGappyKeysMask(PagedBlockDiagonalGappyKeysMask):
    """
    Same as BlockDiagonalCausalWithOffsetGappyKeysMask, but for paged attention.
    block_tables has shape [batch_size, max_num_pages] and K/V have shape
    [1, max_num_pages * page_size, num_heads, head_dim] or
    [1, max_num_pages * page_size, num_groups, num_heads, head_dim]
    """

    _UNPAGED_TYPE = BlockDiagonalCausalWithOffsetGappyKeysMask

    def to(
        self, device: torch.device
    ) -> "PagedBlockDiagonalCausalWithOffsetGappyKeysMask":
        assert (
            type(self) is PagedBlockDiagonalCausalWithOffsetGappyKeysMask
        ), "Please implement in subclass"
        return PagedBlockDiagonalCausalWithOffsetGappyKeysMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            block_tables=_to_device(self.block_tables, device),
            page_size=self.page_size,
        )


@dataclass
class BlockDiagonalCausalLocalAttentionMask(BlockDiagonalCausalMask):
    """
    (Experimental feature)
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`.
    This makes the mask "local" and the attention pattern banded.

    The ith query in a block only attends to keys in its block with index
    greater than i - window_size and less than or equal to i.
    """

    _window_size: int = 0  # forced due to inheritance and default arguments

    def to(self, device) -> "BlockDiagonalCausalLocalAttentionMask":
        assert (
            type(self) is BlockDiagonalCausalLocalAttentionMask
        ), "Please implement in subclass"
        return BlockDiagonalCausalLocalAttentionMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            _batch_sizes=self._batch_sizes,
            _window_size=self._window_size,
        )

    def __post_init__(self):
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )
        q_seqlen = [
            y - x
            for x, y in zip(
                self.q_seqinfo.seqstart_py[:-1], self.q_seqinfo.seqstart_py[1:]
            )
        ]
        kv_seqlen = [
            y - x
            for x, y in zip(
                self.k_seqinfo.seqstart_py[:-1], self.k_seqinfo.seqstart_py[1:]
            )
        ]
        for q, k in zip(q_seqlen, kv_seqlen):
            if q - self._window_size >= k:
                # Each query only attends to keys no further than window_size back.
                # When q > k + window_size, there will be a query for which the window doesn't reach any key.
                raise RuntimeError(
                    f"No keys are attended in q_seqlen {q} k_seqlen {k} with sliding window {self._window_size}"
                )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask(
            shape,
            dtype=dtype,
            device=device,
            window_size=self._window_size,
        )


@dataclass
class BlockDiagonalCausalLocalAttentionFromBottomRightMask(
    BlockDiagonalCausalFromBottomRightMask
):
    """
    (Experimental feature)
    Same as :attr:`xformers.ops.fmha.attn_bias.BlockDiagonalCausalMask`.
    This makes the mask "local" and the attention pattern banded.

    A query with distance j from the last query in its block only attends to
    keys in the same block, and only those whose distance to the last key
    in the block is greater than or equal to j and less than window_size + j.
    """

    _window_size: int = 0  # forced due to inheritance and default arguments

    def to(self, device) -> "BlockDiagonalCausalLocalAttentionFromBottomRightMask":
        assert (
            type(self) is BlockDiagonalCausalLocalAttentionFromBottomRightMask
        ), "Please implement in subclass"
        return BlockDiagonalCausalLocalAttentionFromBottomRightMask(
            q_seqinfo=self.q_seqinfo.to(device),
            k_seqinfo=self.k_seqinfo.to(device),
            _batch_sizes=self._batch_sizes,
            _window_size=self._window_size,
        )

    def __post_init__(self):
        super().__post_init__()
        if self._window_size <= 0:
            raise ValueError(
                f"Expected `window_size > 0`, but window_size={self._window_size}"
            )

    def _create_block_mask(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask(
            shape,
            dtype=dtype,
            device=device,
            window_size=self._window_size,
            from_bottomright=True,
        )


class AttentionBiasSubTensor(torch.Tensor, AttentionBias):
    HOLDS_DENSE_TENSOR = False

    _subtensor: torch.Tensor

    @staticmethod
    def __new__(cls, *, _subtensor=None, device=None, **kwargs):
        raise NotImplementedError()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._overloadpacket in [
            torch.ops.aten.clone,
            torch.ops.aten.detach,
            torch.ops.aten._to_copy,
            torch.ops.aten.to,
        ]:
            return cls(_subtensor=func(args[0]._subtensor, *args[1:], **kwargs))
        return NotImplemented

    def __tensor_flatten__(self):
        return ["_subtensor"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        assert meta is None
        return cls(_subtensor=inner_tensors["_subtensor"])

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


class _AddDenseBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, causal_bias, tensor):
        assert type(causal_bias) is LowerTriangularMask
        return LowerTriangularMaskWithTensorBias(tensor)

    @staticmethod
    def backward(ctx, grad_out):
        return None, grad_out


class LowerTriangularMask(AttentionBiasSubTensor):
    """
    A lower-triangular (aka causal) mask

    A query Q cannot attend to a key which is farther from the
    initial key than Q is from the initial query.

    See also :attr:`LowerTriangularFromBottomRightMask` if the number
    of queries is not equal to the number of keys/values.
    """

    HOLDS_DENSE_TENSOR = False

    @staticmethod
    def __new__(cls, *, _subtensor=None, device="cpu", **kwargs):
        """
        Note: create on CPU by default to avoid initializing CUDA context
        by mistake.
        """
        if _subtensor is None:
            _subtensor = torch.empty((0,), device=device)
        tensor = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            [],
            device=_subtensor.device,
            dtype=_subtensor.dtype,
            requires_grad=False,
        )
        tensor._subtensor = _subtensor
        return tensor

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return _materialize_causal_mask(shape, dtype=dtype, device=device)

    def add_bias(self, bias: torch.Tensor) -> "LowerTriangularMaskWithTensorBias":
        """
        Creates a new causal mask with an arbitrary ``torch.Tensor`` bias
        """
        return _AddDenseBias.apply(self, bias)


class LowerTriangularMaskWithTensorBias(LowerTriangularMask):
    """A lower-triangular (aka causal) mask with an additive bias"""

    HOLDS_DENSE_TENSOR = True

    @staticmethod
    def __new__(cls, bias):
        tensor = torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            bias.shape,
            device=bias.device,
            dtype=bias.dtype,
            requires_grad=bias.requires_grad,
        )
        tensor._subtensor = bias
        return tensor

    def materialize(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        return super().materialize(shape, dtype=dtype, device=device) + self._subtensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func._overloadpacket in [
            torch.ops.aten.unsqueeze,
            torch.ops.aten.select,
            torch.ops.aten.slice,
            torch.ops.aten.clone,
            torch.ops.aten.detach,
            torch.ops.aten._to_copy,
            torch.ops.aten.to,
            torch.ops.aten.view,
        ]:
            output = func(
                *[a._subtensor if isinstance(a, cls) else a for a in args],
                **kwargs,
            )
            return cls(output)
        return NotImplemented


torch._dynamo.allow_in_graph(LowerTriangularMask)
torch._dynamo.allow_in_graph(LowerTriangularMaskWithTensorBias)

VARLEN_BIASES = (
    BlockDiagonalMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
)
