# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Callable, List, Optional, Sequence, Union, overload

import torch
import torch.distributed as dist
from torch.distributed._symmetric_memory import (
    _pipelined_all_gather_and_consume,
    _pipelined_produce_and_all2all,
)


def _is_fp8_dtype(dt: torch.dtype):
    # Detect if it's float8_e4m3fn or float8_e5m2 without mentioning them in
    # order to support old versions of PyTorch that don't define them.
    return dt.is_floating_point and torch.finfo(dt).bits == 8


def _can_ranks_communicate_all_to_all_over_nvlink(group: dist.ProcessGroup) -> bool:
    # FIXME This is currently overly simplistic, must be improved. The following
    # should be enough:
    # - ensure that all ranks are running on the same machine (by exchanging
    #   their /proc/sys/kernel/random/boot_id value)
    # - ensure there's P2P between all pairs of ranks (can_device_access_peer
    #   could help here but it's unclear what happens if target devices aren't
    #   visible? maybe just trying to exchange IPC handles and catching errors
    #   would work? note that in any case some ranks might succeed while some
    #   might fail so we need a barrier to have them all make the same decision)
    return group.size() <= 8


def _should_use_fallback(group: dist.ProcessGroup) -> bool:
    world_size = group.size()
    if int(os.environ.get("DISABLE_FUSED_SEQUENCE_PARALLEL", "0")):
        return True
    elif world_size == 1:
        return True
    elif not _can_ranks_communicate_all_to_all_over_nvlink(group):
        return True
    return False


def _default_stream_factory() -> torch.cuda.Stream:
    return torch.cuda.current_stream()


@overload
def fused_allgather_and_linear(
    scattered_input: torch.Tensor,
    weight: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    out: Optional[torch.Tensor] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> torch.Tensor:
    ...


@overload
def fused_allgather_and_linear(
    scattered_input: torch.Tensor,
    weight: List[torch.Tensor],
    *,
    group: dist.ProcessGroup,
    out: Optional[List[torch.Tensor]] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> List[torch.Tensor]:
    ...


def fused_allgather_and_linear(
    scattered_input: torch.Tensor,
    weight: Union[torch.Tensor, List[torch.Tensor]],
    *,
    group: dist.ProcessGroup,
    out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    timeout_s: int = 60 * 60,
    scale_scattered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Performs a fused all-gather followed by a linear op

    It is equivalent to the following plain PyTorch code:

    # like scattered_input but with first dim multiplied by group's world size
    gathered_input = scattered_input.new_empty(...)
    dist.all_gather_into_tensor(gathered_input, scattered_input, group=group)
    return torch.nn.functional.linear(gathered_input, weight)

    It achieves this by breaking down the matmul into smaller partial ops (as
    many as the world size), each needing as input a different "contribution"
    to the all-gather (by a different rank), and writing to a different chunk of
    the output. Then, on one stream, it sends the local contribution to all
    other ranks (first one rank over, then two, ...) while, on another stream,
    it launches the sub-matmuls in the order in which the remote contributions
    (which are the sub-matmuls' inputs) are supposed to arrive, so that ideally
    none of the sub-matmuls will ever have to wait.

    The idea comes from this paper: https://arxiv.org/abs/2302.05442

    This method uses a staging buffer, which persists across calls, of the same
    size as the all-gathered input tensor (i.e., the input's size times the
    world size). If multiple inputs of multiple sizes are used, the staging
    buffer will be the maximum needed by any of them. Each call, when it starts,
    must first wait for the previous call to finish using the staging buffer. In
    normal conditions, where there's some other operation between two calls,
    this isn't an issue.

    Supports FP8 gemm for tensor-wise quantized weight and input tensors.
    To enable FP8 gemm:
    1. pass scattered_input and weight as quantized FP8 datatype
    2. pass scale_scattered_input and scale_weight, the scales used to
    quantize input and weight, respectively.
    3. set out_dtype, if not specified, will be inferred from scattered_input type.

    """
    world_size = group.size()
    weights = weight if isinstance(weight, list) else [weight]
    assert (scale_scattered_input is None) == (scale_weight is None)
    if scale_weight is not None:
        assert isinstance(weight, list) == isinstance(scale_weight, list)
        scales_weights: Sequence[Optional[torch.Tensor]] = (
            scale_weight if isinstance(scale_weight, list) else [scale_weight]
        )
        assert len(weights) == len(scales_weights)
        assert _is_fp8_dtype(scattered_input.dtype)
        assert all(_is_fp8_dtype(w.dtype) for w in weights)
        assert out_dtype is not None, "output_dtype is required with FP8"
    else:
        scales_weights = [None] * len(weights)
    assert all(w.ndim == 2 for w in weights)
    assert scattered_input.ndim >= 2
    assert all(scattered_input.shape[-1] == w.shape[-1] for w in weights)
    assert scattered_input.is_contiguous()
    gathered_input_shape = (world_size,) + scattered_input.shape
    gathered_output_shapes = [gathered_input_shape[:-1] + w.shape[:-1] for w in weights]
    if out is not None:
        assert isinstance(out, list) == isinstance(weight, list)
        gathered_outputs = out if isinstance(out, list) else [out]
        assert len(gathered_outputs) == len(gathered_output_shapes)
        assert all(
            go.shape == gos for go, gos in zip(gathered_outputs, gathered_output_shapes)
        )
        assert all(go.is_contiguous() for go in gathered_outputs)
        if out_dtype is not None:
            if isinstance(out, list):
                for o in out:
                    assert o.dtype == out_dtype
            else:
                assert out.dtype == out_dtype
    else:
        gathered_outputs = [
            scattered_input.new_empty(
                gos,
                dtype=out_dtype if out_dtype is not None else scattered_input.dtype,
            )
            for gos in gathered_output_shapes
        ]

    torch.ops.xformers_python._fused_allgather_and_linear_impl(
        scattered_input,
        weights,
        group.group_name,
        gathered_outputs,
        timeout_s=timeout_s,
        scale_scattered_input=scale_scattered_input,
        scales_weights=scales_weights,
    )

    if isinstance(weight, list):
        return [go.flatten(0, 1) for go in gathered_outputs]
    else:
        return gathered_outputs[0].flatten(0, 1)


@torch.library.custom_op(
    "xformers_python::_fused_allgather_and_linear_impl",
    mutates_args={"gathered_outputs"},
    device_types="cuda",
)
def _fused_allgather_and_linear_custom_op(
    scattered_input: torch.Tensor,
    weights: List[torch.Tensor],
    process_group_name: str,
    gathered_outputs: List[torch.Tensor],
    timeout_s: int,
    scale_scattered_input: torch.Tensor,
    scales_weights: Sequence[Optional[torch.Tensor]],
) -> None:
    process_group = dist.distributed_c10d._resolve_process_group(process_group_name)

    def my_matmul(
        input_: torch.Tensor,
        src_rank: int,
        stream_factory: Callable[[], torch.cuda.Stream],
    ) -> None:
        for w, scale_weight, go in zip(weights, scales_weights, gathered_outputs):
            with torch.cuda.stream(stream_factory()):
                if scale_scattered_input is not None and scale_weight is not None:
                    torch._scaled_mm(
                        input_,
                        w.t(),
                        out_dtype=go[src_rank].dtype,
                        scale_a=scale_scattered_input,
                        scale_b=scale_weight,
                        out=go[src_rank],
                    )
                else:
                    torch.matmul(input_, w.t(), out=go[src_rank])

    fused_allgather_and_anything(
        scattered_input,
        my_matmul,
        group=process_group,
        timeout_s=timeout_s,
    )


def fused_allgather_and_anything(
    scattered_input: torch.Tensor,
    my_matmul: Callable[[torch.Tensor, int, Callable[[], torch.cuda.Stream]], None],
    *,
    group: dist.ProcessGroup,
    timeout_s: int = 60 * 60,
    **private_args_DO_NOT_USE,
) -> None:
    world_size = group.size()

    assert scattered_input.is_contiguous()

    gathered_input_shape = (world_size,) + scattered_input.shape

    if world_size == 1:
        my_matmul(scattered_input, 0, _default_stream_factory)

    # Fallback
    elif _should_use_fallback(group):
        gathered_input = scattered_input.new_empty(gathered_input_shape)
        dist.all_gather_into_tensor(
            output_tensor=gathered_input, input_tensor=scattered_input, group=group
        )
        for src_rank in range(world_size):
            my_matmul(
                gathered_input[src_rank],
                src_rank,
                _default_stream_factory,
            )

    # Fast path
    else:
        def my_wrapper(t, rank):
            my_matmul(t.squeeze(0), rank, _default_stream_factory)

        gathered_input = scattered_input.new_empty(gathered_input_shape)
        _pipelined_all_gather_and_consume(
            scattered_input,
            my_wrapper,
            gathered_input,
            group.group_name,
        )


def fused_linear_and_reducescatter(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    out: Optional[torch.Tensor] = None,
    timeout_s: int = 60 * 60,
    scale_gathered_input: Optional[torch.Tensor] = None,
    scale_weight: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    **private_args_DO_NOT_USE,
) -> torch.Tensor:
    """Performs a fused linear op followed by a reduce-scatter

    It is equivalent to the following plain PyTorch code:

    gathered_output = torch.nn.functional.linear(gathered_input, weight)
    # like gathered_output but with first dim divided by group's world size
    scattered_output = gathered_output.new_empty(...)
    dist.reduce_scatter_tensor(scattered_output, gathered_output, group=group)

    Supports FP8 gemm with tensor-wise quantized weights. To enable FP8 gemm:
    1. pass weight and gathered_input as FP8 tensors
    2. Set `scale_gathered_input` and `scale_weight` to the scales used to quantize
    inputs and weight, respectively.
    3. Set out_dtype to the desired output dtype. If not specified, it will be inferred from
    gathered_input datatype.
    """
    world_size = group.size()
    assert (scale_gathered_input is None) == (scale_weight is None)
    if scale_weight is not None:
        assert _is_fp8_dtype(gathered_input.dtype)
        assert _is_fp8_dtype(weight.dtype)
        assert out_dtype is not None, "output_dtype is required with FP8"
    assert weight.ndim == 2
    assert gathered_input.ndim >= 2
    assert gathered_input.shape[-1] == weight.shape[-1]
    assert gathered_input.is_contiguous()
    assert gathered_input.shape[0] % world_size == 0
    gathered_input = gathered_input.view(
        (world_size, gathered_input.shape[0] // world_size) + gathered_input.shape[1:]
    )
    gathered_output_shape = gathered_input.shape[:-1] + weight.shape[:-1]
    scattered_output_shape = gathered_output_shape[1:]
    if out is not None:
        scattered_output = out
        assert scattered_output.device == gathered_input.device
        assert scattered_output.dtype == gathered_input.dtype
        assert scattered_output.shape == scattered_output_shape
        if out_dtype is not None:
            assert scattered_output.dtype == out_dtype
    else:
        scattered_output = gathered_input.new_empty(
            scattered_output_shape,
            dtype=out_dtype if out_dtype is not None else gathered_input.dtype,
        )

    torch.ops.xformers_python._fused_linear_and_reducescatter_impl(
        gathered_input,
        weight,
        group.group_name,
        scattered_output,
        timeout_s=timeout_s,
        scale_gathered_input=scale_gathered_input,
        scale_weight=scale_weight,
    )

    return scattered_output


@torch.library.custom_op(
    "xformers_python::_fused_linear_and_reducescatter_impl",
    mutates_args={"scattered_output"},
    device_types="cuda",
)
def _fused_linear_and_reducescatter_custom_op(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    process_group_name: str,
    scattered_output: torch.Tensor,
    timeout_s: int,
    scale_gathered_input: Optional[torch.Tensor],
    scale_weight: Optional[torch.Tensor],
) -> None:
    process_group = dist.distributed_c10d._resolve_process_group(process_group_name)

    def my_matmul(
        output: torch.Tensor,
        dst_rank: int,
        stream_factory: Callable[[], torch.cuda.Stream],
    ) -> None:
        with torch.cuda.stream(stream_factory()):
            if scale_gathered_input is not None and scale_weight is not None:
                torch._scaled_mm(
                    gathered_input[dst_rank],
                    weight.t(),
                    out_dtype=output.dtype,
                    scale_a=scale_gathered_input,
                    scale_b=scale_weight,
                    out=output,
                )
            else:
                torch.matmul(gathered_input[dst_rank], weight.t(), out=output)

    fused_anything_and_reducescatter(
        my_matmul,
        scattered_output,
        group=process_group,
        timeout_s=timeout_s,
    )


def fused_anything_and_reducescatter(
    my_matmul: Callable[[torch.Tensor, int, Callable[[], torch.cuda.Stream]], None],
    scattered_output: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    timeout_s: int = 60 * 60,
    **private_args_DO_NOT_USE,
) -> None:
    world_size = group.size()

    assert scattered_output.is_contiguous()

    gathered_output_shape = (world_size,) + scattered_output.shape

    if world_size == 1:
        my_matmul(scattered_output, 0, _default_stream_factory)

    # Fallback
    elif _should_use_fallback(group):
        gathered_output = scattered_output.new_empty(gathered_output_shape)
        for dst_rank in range(world_size):
            my_matmul(
                gathered_output[dst_rank],
                dst_rank,
                _default_stream_factory,
            )
        dist.reduce_scatter_tensor(
            output=scattered_output, input=gathered_output, group=group
        )

    # Fast path
    else:
        def my_wrapper(rank, t):
            my_matmul(t.squeeze(0), rank, _default_stream_factory)

        gathered_output = scattered_output.new_empty(gathered_output_shape)
        _pipelined_produce_and_all2all(
            my_wrapper,
            gathered_output,
            group.group_name,
        )

        torch.sum(gathered_output, dim=0, out=scattered_output)
