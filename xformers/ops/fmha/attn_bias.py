# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class TensorCreateInfo:
    shape: Tuple[int, ...]
    dtype: torch.dtype = torch.float32
    device: torch.device = "cpu"

    def __post_init__(self) -> None:
        assert len(self.shape) in [2, 3]


@dataclass
class SeqLenInfo:
    max_seqlen: int
    cu_seqlen: torch.Tensor
    cu_seqlen_py: List[int]

    @staticmethod
    def cat_for_attention(
        tensors: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, "SeqLenInfo"]:
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        tensors_bs1 = tuple(x.reshape([1, -1, *x.shape[2:]]) for x in tensors)
        c = torch.cat(tensors_bs1, dim=1)
        cu_seqlen_py = [0]
        max_seqlen = -1
        for tensor in tensors:
            seqlen = tensor.shape[1]
            max_seqlen = max(max_seqlen, seqlen)
            for _ in range(tensor.shape[0]):
                cu_seqlen_py.append(cu_seqlen_py[len(cu_seqlen_py) - 1] + seqlen)
        cu_seqlen = torch.tensor(cu_seqlen_py, dtype=torch.int32, device=tensor.device)
        return c, SeqLenInfo(
            max_seqlen=max_seqlen, cu_seqlen=cu_seqlen, cu_seqlen_py=cu_seqlen_py
        )

    def split_after_attention(
        self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None
    ) -> List[torch.Tensor]:
        if self.cu_seqlen_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(
                f"Invalid `torch.Tensor` of shape {x.shape}, expected format "
                f"(B, M, *) with B=1 and M={self.cu_seqlen_py[-1]}\n"
                f" cu_seqlen: {self.cu_seqlen_py}"
            )
        if batch_sizes is None:
            batch_sizes = range(len(self.cu_seqlen_py) - 1)
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
class AttentionBiasBlockDiagonal:
    q_seqinfo: SeqLenInfo
    k_seqinfo: SeqLenInfo
    _batch_sizes: Optional[Sequence[int]] = None

    def __post_init__(self) -> None:
        assert len(self.q_seqinfo.cu_seqlen_py) == len(self.k_seqinfo.cu_seqlen_py)

    def materialize(self, create_info: TensorCreateInfo) -> torch.Tensor:
        tensors: List[torch.Tensor] = []
        for q_start, q_end, k_start, k_end in zip(
            self.q_seqinfo.cu_seqlen_py,
            self.q_seqinfo.cu_seqlen_py[1:],
            self.k_seqinfo.cu_seqlen_py,
            self.k_seqinfo.cu_seqlen_py[1:],
        ):
            tensors.append(
                torch.ones(
                    [q_end - q_start, k_end - k_start],
                    dtype=create_info.dtype,
                    device=create_info.device,
                )
            )
        mask = torch.block_diag(*tensors)
        for _ in range(len(create_info.shape) - 2):
            mask = mask.unsqueeze(0)
        return mask

    @staticmethod
    def cat_for_attention(
        tensors: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, "AttentionBiasBlockDiagonal"]:
        batch_sizes = [tensor.shape[0] for tensor in tensors]
        concat_tensors, seqinfo = SeqLenInfo.cat_for_attention(tensors)
        return concat_tensors, AttentionBiasBlockDiagonal(
            q_seqinfo=seqinfo,
            k_seqinfo=seqinfo,
            _batch_sizes=batch_sizes,
        )

    def split_after_attention(self, tensor: torch.Tensor) -> Sequence[torch.Tensor]:
        assert self.q_seqinfo is self.k_seqinfo
        return self.q_seqinfo.split_after_attention(tensor, self._batch_sizes)


def _create_causal_mask(create_info: TensorCreateInfo) -> torch.Tensor:
    dtype = create_info.dtype
    create_as = dtype if dtype is not torch.bfloat16 else torch.float32
    tensor = torch.full(  # type: ignore
        create_info.shape,
        dtype=create_as,
        fill_value=float("-inf"),
        device=create_info.device,
    )
    return torch.triu(tensor, diagonal=1).to(dtype)  # type: ignore


class AttentionBias:
    """A custom bias that can be applied \
        in :attr:`xformers.ops.memory_efficient_attention`.

    When using an :attr:`xformers.ops.AttentionBias`
    instead of a :attr:`torch.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.
    """

    def __init__(self, *args, **kwargs) -> None:
        create_info: Optional[TensorCreateInfo] = None
        if len(args) + len(kwargs):
            create_info = TensorCreateInfo(*args, **kwargs)
        self.causal = False
        self._bias: Optional[torch.Tensor] = None
        self._block_diag: Optional[AttentionBiasBlockDiagonal] = None
        self.create_info = create_info

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self._bias

    @bias.setter
    def bias(self, bias: Optional[torch.Tensor]) -> None:
        """
        Returns a new :attr:`AttentionBias` with a custom `torch.Tensor` mask added

        NOTE: This is not compatible with block-diagonal biases
        """
        if bias is None:
            self._bias = None
            return
        # Make sure we don't add an `AttentionBias` for instance
        assert isinstance(bias, torch.Tensor)
        if self.block_diag:
            raise ValueError(
                "Block-diagonal bias and additive bias can't be used together"
            )

        self._bias = self._bias + bias if self._bias is not None else bias
        if self.create_info is None:
            self.create_info = TensorCreateInfo(
                self.bias.shape, dtype=self.bias.dtype, device=self.bias.device
            )

    @property
    def block_diag(self) -> Optional[AttentionBiasBlockDiagonal]:
        return self._block_diag

    @block_diag.setter
    def block_diag(self, value: Optional[AttentionBiasBlockDiagonal]) -> None:
        if self.bias is not None:
            raise ValueError(
                "Block-diagonal bias and additive bias can't be used together"
            )
        self._block_diag = value

    def materialize(self) -> torch.Tensor:
        """
        Materializes the bias as a `torch.Tensor`. This is very slow
        and we don't attempt to make it fast. Only use for debugging/testing.
        Returned tensor has shape [..., Mq, Mk]
        """
        if self.create_info is None:
            raise ValueError(
                "Can't create a causal mask if no dimension/shape/device provided"
            )
        tensor = torch.zeros(
            self.create_info.shape,
            dtype=self.create_info.dtype,
            device=self.create_info.device,
        )
        if self.bias is not None:
            tensor += self.bias
        if self.causal:
            tensor = tensor + _create_causal_mask(self.create_info)
        if self.block_diag is not None:
            tensor = tensor + self.block_diag.materialize(self.create_info)
        if tensor is None:
            raise ValueError("This mask is empty")
        return tensor


def LowerTriangularMask(*args, **kwargs) -> AttentionBias:
    attn_bias = AttentionBias(*args, **kwargs)
    attn_bias.causal = True
    return attn_bias


def cat_for_attention(
    tensors: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, AttentionBias]:
    """
    Input tensors are assumed to be in shape [B, M, *]
    """
    tensor_out, block_diag = AttentionBiasBlockDiagonal.cat_for_attention(tensors)
    attn_bias = AttentionBias(
        [tensor_out.shape[1], tensor_out.shape[1]],
        dtype=tensor_out.dtype,
        device=tensor_out.device,
    )
    attn_bias.block_diag = block_diag
    return tensor_out, attn_bias


def split_after_attention(
    tensor: torch.Tensor, attn_bias: AttentionBias
) -> Sequence[torch.Tensor]:
    assert attn_bias is not None
    if attn_bias.block_diag is None:
        raise ValueError("Provided bias is not block-diagonal")
    return attn_bias.block_diag.split_after_attention(tensor)
