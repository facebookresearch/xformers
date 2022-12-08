# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Set, Tuple, Type, Union

import torch


class AttentionMask:
    """Base class for custom masks that can be applied \
        in :attr:`xformers.ops.memory_efficient_attention`.

    When using an :attr:`xformers.ops.AttentionMask`
    instead of a :attr:`torch.Tensor`, the mask matrix does
    not need to be materialized, and can be
    hardcoded into some kernels for better performance.

    See also :attr:`xformers.ops.LowerTriangularMask`
    """

    def to_tensor(self) -> torch.Tensor:
        """Materializes the mask tensor

        Returns:
            torch.Tensor
        """
        raise NotImplementedError()


class LowerTriangularMask(AttentionMask):
    """A lower triangular mask that can be used for causal attention"""

    def __init__(self, *tensor_args, **tensor_kwargs) -> None:
        """Creates a Lower triangular mask.
        It is not requires to specify any parameter, as they are only \
            used when calling :attr:`LowerTriangularMask.to_tensor`

        The mask will not be materialized by default, and hence does not use \
            any additional memory, but acts as an option for the MHA kernel.
        """
        self._tensor: Optional[torch.Tensor] = None
        self._tensor_kwargs = tensor_kwargs
        self._tensor_args = tensor_args

    def to_tensor(self) -> torch.Tensor:
        """Materializes the mask tensor

        Returns:
            torch.Tensor
        """
        if self._tensor is None:
            # Work around for "triu_tril_cuda_template" not implemented for 'BFloat16'
            dtype = self._tensor_kwargs.pop("dtype", torch.float)
            create_as = dtype if dtype is not torch.bfloat16 else torch.float32
            self._tensor = torch.full(  # type: ignore
                *self._tensor_args,
                **self._tensor_kwargs,
                dtype=create_as,
                fill_value=float("-inf"),
            )
            self._tensor = torch.triu(self._tensor, diagonal=1).to(dtype)  # type: ignore
        return self._tensor


@dataclass
class Inputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None
    p: float = 0.0
    scale: Optional[float] = None

    @property
    def device(self) -> torch.device:
        return self.query.device

    @property
    def scale_float(self) -> float:
        return self.query.shape[-1] ** (-0.5) if self.scale is None else self.scale

    def normalize_bmhk(self) -> Tuple[int, ...]:
        if self.query.ndim not in [3, 4]:
            raise ValueError(
                f"Invalid shape for query: {self.query.shape}. "
                "Expected shape [batch, seqlen, num_heads, K], or [batch, seqlen, K]."
            )
        output_shape = (self.query.shape[:-1]) + (self.value.shape[-1],)
        # Convert from legacy format
        if self.query.ndim == 3:
            self.query = self.query.unsqueeze(2)
            self.key = self.key.unsqueeze(2)
            self.value = self.value.unsqueeze(2)
        return output_shape


@dataclass
class Context:
    lse: torch.Tensor
    out: torch.Tensor

    def get_padded_lse(self, pad_to: int) -> torch.Tensor:
        pad_amount = (pad_to - (self.lse.shape[2] % pad_to)) % pad_to
        lse = self.lse
        if pad_amount > 0:
            lse = torch.nn.functional.pad(self.lse, [0, pad_amount], value=math.inf)
        return lse


@dataclass
class Gradients:
    dq: torch.Tensor
    dk: torch.Tensor
    dv: torch.Tensor


class AttentionOpBase:
    """Base class for any attention operator in xFormers

    See:

    - :attr:`xformers.ops.MemoryEfficientAttentionOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionCutlassOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionFlashAttentionOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp`
    """

    OPERATOR: Any
    SUPPORTED_DEVICES: Set[str]
    SUPPORTED_DTYPES: Set[torch.dtype]
    SUPPORTED_MAX_K: float
    SUPPORTED_ATTN_BIAS_TYPES: Set[Any] = {type(None)}
    SUPPORTS_DROPOUT: bool
    SUPPORTS_CUSTOM_SCALE: bool = False
    SUPPORTS_DIFFERENT_VALUE_EMBED: bool = False
    NAME: str

    _TEST_BATCH_SIZES: List[int] = [1, 300]
    _TEST_K: List[int] = [32, 128]

    @classmethod
    def info(cls):
        if cls.OPERATOR.__name__ == "no_such_operator":
            return "not built"
        return "available"

    @classmethod
    def supports(cls, d: Inputs) -> bool:
        device_type = d.query.device.type
        dtype = d.query.dtype
        if device_type not in cls.SUPPORTED_DEVICES:
            return False
        if dtype not in cls.SUPPORTED_DTYPES:
            return False
        if (
            not cls.SUPPORTS_DIFFERENT_VALUE_EMBED
            and d.query.shape[-1] != d.value.shape[-1]
        ):
            return False
        if max(d.query.shape[-1], d.value.shape[-1]) > cls.SUPPORTED_MAX_K:
            return False
        if type(d.attn_bias) not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            return False
        if (d.p != 0.0) and not cls.SUPPORTS_DROPOUT:
            return False
        if d.scale is not None and not cls.SUPPORTS_CUSTOM_SCALE:
            return False
        # bfloat16 is only supported on A100+
        # ... although the kernels can still run and give the
        # correct result
        if dtype is torch.bfloat16 and (
            not device_type.startswith("cuda")
            or torch.cuda.get_device_capability(d.query.device)[0] < 8
        ):
            return False
        return True


class AttentionFwOpBase(AttentionOpBase):
    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 4e-3,
        torch.bfloat16: 2e-2,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 4e-4,
        torch.bfloat16: 5e-3,
    }

    @classmethod
    def apply(
        cls, inp: Inputs, needs_gradient: bool
    ) -> Tuple[torch.Tensor, Optional[Context]]:
        raise NotImplementedError()


class AttentionBwOpBase(AttentionOpBase):
    ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 5e-4,
        torch.half: 9e-2,
        torch.bfloat16: 0.7,
    }
    ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 1e-4,
        torch.half: 2e-2,
        torch.bfloat16: 0.1,
    }

    @classmethod
    def apply(cls, ctx: Context, inp: Inputs, grad: torch.Tensor) -> Gradients:
        raise NotImplementedError()


AttentionOp = Tuple[
    Optional[Type[AttentionFwOpBase]], Optional[Type[AttentionBwOpBase]]
]


@dataclass
class AttentionOpDispatch:
    """Dispatcher to automatically select
    the best operator to run memory-efficient attention.

    :Deprecated:

        This class is deprecated and will be removed in a later version
    """

    op: AttentionOp

    @classmethod
    def from_arguments(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]] = None,
        p: float = 0.0,
        scale: Optional[float] = None,
    ) -> "AttentionOpDispatch":
        """Here for backward compatibility"""
        from .dispatch import _dispatch_bw, _dispatch_fw

        inp = Inputs(
            query=query,
            key=key,
            value=value,
            attn_bias=attn_bias,
            p=p,
            scale=scale,
        )
        return AttentionOpDispatch(op=(_dispatch_fw(inp), _dispatch_bw(inp)))


def bmk2bmhk(tensor, num_heads: int) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor
    return tensor.reshape([-1, num_heads, tensor.shape[1], tensor.shape[2]]).permute(
        (0, 2, 1, 3)
    )
