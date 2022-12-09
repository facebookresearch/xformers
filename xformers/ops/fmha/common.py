# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Set, Type, Union

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


class AttentionOpBase(torch.autograd.Function):
    """Base class for any attention operator in xFormers

    See:

    - :attr:`xformers.ops.MemoryEfficientAttentionOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionCutlassOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionFlashAttentionOp`

    - :attr:`xformers.ops.MemoryEfficientAttentionCutlassFwdFlashBwOp`
    """

    FORWARD_OPERATOR: Any
    FORWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 3e-4,
        torch.half: 4e-3,
        torch.bfloat16: 2e-2,
    }
    FORWARD_ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 2e-5,
        torch.half: 4e-4,
        torch.bfloat16: 5e-3,
    }
    BACKWARD_ERROR_ATOL: Mapping[torch.dtype, float] = {
        torch.float: 5e-4,
        torch.half: 9e-2,
        torch.bfloat16: 0.7,
    }
    BACKWARD_ERROR_RTOL: Mapping[torch.dtype, float] = {
        torch.float: 1e-4,
        torch.half: 2e-2,
        torch.bfloat16: 0.1,
    }
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
        if cls.FORWARD_OPERATOR.__name__ == "no_such_operator":
            return "not built"
        return "available"

    @classmethod
    def forward_no_grad(
        cls,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: Optional[Union[torch.Tensor, AttentionMask]],
        p: float,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    def forward(cls, ctx, query, key, value, attn_bias, p, scale):
        raise NotImplementedError()

    @classmethod
    def backward(cls, ctx, grad):
        raise NotImplementedError()

    @classmethod
    def supports(cls, d: "AttentionOpDispatch") -> bool:
        device_type = d.device if isinstance(d.device, str) else d.device.type
        if device_type not in cls.SUPPORTED_DEVICES:
            return False
        if d.dtype not in cls.SUPPORTED_DTYPES:
            return False
        if not cls.SUPPORTS_DIFFERENT_VALUE_EMBED and d.k != d.kv:
            return False
        if max(d.k, d.kv) > cls.SUPPORTED_MAX_K:
            return False
        if d.attn_bias_type not in cls.SUPPORTED_ATTN_BIAS_TYPES:
            return False
        if d.has_dropout and not cls.SUPPORTS_DROPOUT:
            return False
        if d.has_custom_scale and not cls.SUPPORTS_CUSTOM_SCALE:
            return False
        # bfloat16 is only supported on A100+
        # ... although the kernels can still run and give the
        # correct result
        if d.dtype is torch.bfloat16 and (
            not device_type.startswith("cuda")
            or torch.cuda.get_device_capability(d.device)[0] < 8
        ):
            return False
        return True


AttentionOp = Type[AttentionOpBase]


@dataclass
class AttentionOpDispatch:
    """Dispatcher to automatically select
    the best operator to run memory-efficient attention.
    """

    dtype: torch.dtype
    device: Union[torch.device, str]
    k: int
    has_dropout: bool
    attn_bias_type: Any
    kv_len: int
    q_len: int
    kv: int = -1
    batch_size: int = -1
    num_heads: int = 1
    has_custom_scale: bool = False
    requires_grad: bool = True

    def __post_init__(self):
        if self.kv == -1:
            self.kv = self.k

    def _is_cutlass_fwd_faster_than_flash(self) -> bool:
        # Very small batch sizes - if batch size specified
        if self.batch_size > 0:
            threads_flash = self.batch_size * self.num_heads
            threads_cutlass = threads_flash * (self.q_len // 64)
            if threads_flash < 60 and (threads_cutlass // 2) >= threads_flash:
                return True
        # Large values of K
        return max(self.k, self.kv) == 128

    def _is_triton_fwd_faster_than_cutlass(self) -> bool:
        # TODO: fill out
        return False

    @property
    def op(self) -> AttentionOp:
        """Computes the best operator

        Raises:
            NotImplementedError: if not operator was found

        Returns:
            AttentionOp: The best operator for the configuration
        """
        from .cutlass import Op as MemoryEfficientAttentionCutlassOp
        from .flash import Op as MemoryEfficientAttentionFlashAttentionOp
        from .mixed import (
            MemoryEfficientAttentionCutlassFwdFlashBwOp,
            MemoryEfficientAttentionTritonFwdFlashBwOp,
        )
        from .small_k import Op as MemoryEfficientAttentionOp

        priority_list_ops: List[AttentionOp] = [
            MemoryEfficientAttentionFlashAttentionOp,
            # TODO: remove once triton_faster_than_cutlass method complete
            MemoryEfficientAttentionTritonFwdFlashBwOp,
            MemoryEfficientAttentionCutlassOp,
            MemoryEfficientAttentionOp,
        ]
        if self.requires_grad and self._is_cutlass_fwd_faster_than_flash():
            priority_list_ops.insert(0, MemoryEfficientAttentionCutlassFwdFlashBwOp)
        if self.requires_grad and self._is_triton_fwd_faster_than_cutlass():
            priority_list_ops.insert(0, MemoryEfficientAttentionTritonFwdFlashBwOp)
        for op in priority_list_ops:
            if op.supports(self):
                return op
        raise NotImplementedError(f"No operator found for this attention: {self}")

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
        """Creates an :attr:`xformers.ops.AttentionOpDispatch` from :attr:`xformers.ops.memory_efficient_attention`'s
        arguments

        Args:
            query (torch.Tensor)
            key (torch.Tensor)
            value (torch.Tensor)
            attn_bias (Optional[Union[torch.Tensor, xformers.ops.AttentionMask]], optional): Defaults to None.
            p (float, optional): Defaults to 0.0.
            scale (float, optional): Custom scale. Default to None (use q.shape[-1]**-0.5).

        Returns:
            AttentionOpDispatch
        """
        B, H = query.shape[0], 1
        if query.ndim == 4:
            H = query.shape[2]
        return AttentionOpDispatch(
            dtype=query.dtype,
            device=query.device,
            k=query.shape[-1],
            kv=value.shape[-1],
            has_dropout=p > 0.0,
            has_custom_scale=scale is not None,
            attn_bias_type=type(attn_bias),
            kv_len=value.shape[1],
            q_len=query.shape[1],
            batch_size=B,
            num_heads=H,
            requires_grad=any(x.requires_grad for x in [query, key, value]),
        )
