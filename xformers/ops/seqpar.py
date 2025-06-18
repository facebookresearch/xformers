# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Callable, List, Tuple

import torch
from torch.distributed.distributed_c10d import _resolve_process_group

from .differentiable_collectives import (
    gather_along_first_dim,
    gather_along_first_dim_async,
    reduce_scatter_along_first_dim,
    reduce_scatter_along_first_dim_async,
)
from .sequence_parallel_fused_ops import (
    fused_allgather_and_anything,
    fused_allgather_and_linear,
    fused_anything_and_reducescatter,
    fused_linear_and_reducescatter,
)
from .tiled_matmul import tiled_matmul, tiled_matmul_out


@torch.library.custom_op(
    "xformers_python::sequence_parallel_leading_matmul_fwd",
    mutates_args=(),
    device_types="cuda",
)
def sequence_parallel_leading_matmul_fwd(
    scattered_input: torch.Tensor,
    weights: List[torch.Tensor],
    fuse: bool,
    process_group_name: str,
) -> List[torch.Tensor]:
    process_group = _resolve_process_group(process_group_name)

    if fuse:
        gathered_outputs = fused_allgather_and_linear(
            scattered_input, [w.t() for w in weights], group=process_group
        )
    else:
        gathered_input = gather_along_first_dim(
            scattered_input, process_group=process_group
        )
        (gathered_outputs,) = tiled_matmul(
            [[gathered_input]],
            [[w for w in weights]],
        )
    return gathered_outputs


@torch.library.register_fake("xformers_python::sequence_parallel_leading_matmul_fwd")
def sequence_parallel_leading_matmul_fwd_fake(
    scattered_input: torch.Tensor,
    weights: List[torch.Tensor],
    fuse: bool,
    process_group_name: str,
) -> List[torch.Tensor]:
    mp_size = _resolve_process_group(process_group_name).size()
    return [
        scattered_input.new_empty((scattered_input.shape[0] * mp_size, w.shape[1]))
        for w in weights
    ]


@torch.library.custom_op(
    "xformers_python::sequence_parallel_leading_matmul_bwd",
    mutates_args=(),
    device_types="cuda",
)
def sequence_parallel_leading_matmul_bwd(
    scattered_input: torch.Tensor,
    weights: List[torch.Tensor],
    grad_gathered_outputs: List[torch.Tensor],
    fuse: bool,
    process_group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    process_group = _resolve_process_group(process_group_name)
    mp_size = process_group.size()

    # torch.library.opcheck gives us gradients whose strides are zero.
    # See https://github.com/pytorch/pytorch/issues/132857.
    grad_gathered_outputs = [
        grad_go.clone() if any(s == 0 for s in grad_go.stride()) else grad_go
        for grad_go in grad_gathered_outputs
    ]

    if fuse:
        grad_scattered_input = torch.empty_like(scattered_input)
        grad_weights = [torch.zeros_like(w) for w in weights]

        grad_gathered_outputss = [
            grad_go.tensor_split(mp_size, dim=0) for grad_go in grad_gathered_outputs
        ]

        def my_si_matmul(
            grad_gathered_inputs: List[torch.Tensor],
            dst_rank: int,
            stream_factory: Callable[[], torch.cuda.Stream],
        ) -> None:
            (grad_gi,) = grad_gathered_inputs
            with torch.cuda.stream(stream_factory()):
                tiled_matmul_out(
                    [[grad_gos[dst_rank] for grad_gos in grad_gathered_outputss]],
                    [[w.t()] for w in weights],
                    out=[[grad_gi]],
                )

        fused_anything_and_reducescatter(
            my_si_matmul,
            [grad_scattered_input],
            group=process_group,
        )

        # Each pair of shards of input and grad_output accumulates into the same
        # grad_weight. Thus we need to make sure that the in-place addmms are
        # sequenced correctly for each of the grad_weights.
        events = [torch.cuda.Event() for _ in weights]

        def my_w_matmul(
            gathered_inputs_shard: List[torch.Tensor],
            src_rank: int,
            stream_factory: Callable[[], torch.cuda.Stream],
        ) -> None:
            (gi_shard,) = gathered_inputs_shard
            for grad_gos, grad_w, event in zip(
                grad_gathered_outputss, grad_weights, events
            ):
                with torch.cuda.stream(stream_factory()):
                    event.wait()
                    grad_w.t().addmm_(grad_gos[src_rank].t(), gi_shard)
                    event.record()

        fused_allgather_and_anything(
            [scattered_input],
            my_w_matmul,
            group=process_group,
        )
    else:
        gathered_input, handle = gather_along_first_dim_async(
            scattered_input, process_group=process_group
        )
        ((grad_gathered_input,),) = tiled_matmul(
            [[grad_go for grad_go in grad_gathered_outputs]],
            [[w.t()] for w in weights],
        )
        if handle is not None:
            handle.wait()

        grad_scattered_input, handle = reduce_scatter_along_first_dim_async(
            grad_gathered_input, process_group=process_group
        )

        grad_weights_tuples = tiled_matmul(
            [[grad_go.t()] for grad_go in grad_gathered_outputs],
            [[gathered_input]],
        )
        if handle is not None:
            handle.wait()

        grad_weights = [grad_w.t() for (grad_w,) in grad_weights_tuples]

    return grad_scattered_input, grad_weights


@torch.library.register_fake("xformers_python::sequence_parallel_leading_matmul_bwd")
def sequence_parallel_leading_matmul_bwd_fake(
    scattered_input: torch.Tensor,
    weights: List[torch.Tensor],
    grad_gathered_outputs: List[torch.Tensor],
    fuse: bool,
    process_group_name: str,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    return (torch.empty_like(scattered_input), [torch.empty_like(w) for w in weights])


def sequence_parallel_leading_matmul_setup_context(ctx, inputs, output):
    scattered_input, weights, fuse, process_group_name = inputs
    ctx.save_for_backward(scattered_input, *weights)
    ctx.fuse = fuse
    ctx.process_group_name = process_group_name


def sequence_parallel_leading_matmul_bwd_bridge(ctx, grad_gathered_outputs):
    scattered_input, *weights = ctx.saved_tensors
    (
        grad_scattered_input,
        grad_weights,
    ) = sequence_parallel_leading_matmul_bwd(
        scattered_input,
        list(weights),
        list(grad_gathered_outputs),
        ctx.fuse,
        ctx.process_group_name,
    )
    return grad_scattered_input, grad_weights, None, None


torch.library.register_autograd(
    "xformers_python::sequence_parallel_leading_matmul_fwd",
    sequence_parallel_leading_matmul_bwd_bridge,
    setup_context=sequence_parallel_leading_matmul_setup_context,
)


def sequence_parallel_leading_matmul(
    x: torch.Tensor,
    ws: List[torch.Tensor],
    *,
    fuse: bool,
    process_group: torch.distributed.ProcessGroup,
) -> List[torch.Tensor]:
    os = sequence_parallel_leading_matmul_fwd(
        x.flatten(0, -2), ws, fuse, process_group.group_name
    )
    return [o.view(-1, *x.shape[1:-1], w.shape[1]) for o, w in zip(os, ws)]


@torch.library.custom_op(
    "xformers_python::sequence_parallel_trailing_matmul_fwd",
    mutates_args=(),
    device_types="cuda",
)
def sequence_parallel_trailing_matmul_fwd(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    fuse: bool,
    process_group_name: str,
) -> torch.Tensor:
    process_group = _resolve_process_group(process_group_name)

    if fuse:
        scattered_output = fused_linear_and_reducescatter(
            gathered_input, weight.t(), group=process_group
        )
    else:
        gathered_output = torch.matmul(gathered_input, weight)
        scattered_output = reduce_scatter_along_first_dim(
            gathered_output, process_group=process_group
        )
    return scattered_output


@torch.library.register_fake("xformers_python::sequence_parallel_trailing_matmul_fwd")
def sequence_parallel_trailing_matmul_fwd_fake(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    fuse: bool,
    process_group_name: str,
) -> torch.Tensor:
    mp_size = _resolve_process_group(process_group_name).size()
    return gathered_input.new_empty(
        (gathered_input.shape[0] // mp_size, weight.shape[1])
    )


@torch.library.custom_op(
    "xformers_python::sequence_parallel_trailing_matmul_bwd",
    mutates_args=(),
    device_types="cuda",
)
def sequence_parallel_trailing_matmul_bwd(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    grad_scattered_output: torch.Tensor,
    fuse: bool,
    process_group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    process_group = _resolve_process_group(process_group_name)
    mp_size = process_group.size()

    # torch.library.opcheck gives us gradients whose strides are zero.
    # See https://github.com/pytorch/pytorch/issues/132857.
    if any(s == 0 for s in grad_scattered_output.stride()):
        grad_scattered_output = grad_scattered_output.clone()

    if fuse:
        grad_gathered_input = torch.empty_like(gathered_input)
        grad_weight = torch.zeros_like(weight)

        gathered_inputs = gathered_input.tensor_split(mp_size, dim=0)
        grad_gathered_inputs = grad_gathered_input.tensor_split(mp_size, dim=0)

        def my_gi_and_w_matmul(
            grad_gathered_outputs_shard: List[torch.Tensor],
            src_rank: int,
            stream_factory: Callable[[], torch.cuda.Stream],
        ) -> None:
            (grad_go_shard,) = grad_gathered_outputs_shard
            with torch.cuda.stream(stream_factory()):
                torch.matmul(
                    grad_go_shard, weight.t(), out=grad_gathered_inputs[src_rank]
                )
            with torch.cuda.stream(stream_factory()):
                grad_weight.t().addmm_(grad_go_shard.t(), gathered_inputs[src_rank])

        fused_allgather_and_anything(
            [grad_scattered_output],
            my_gi_and_w_matmul,
            group=process_group,
        )
    else:
        grad_gathered_output = gather_along_first_dim(
            grad_scattered_output, process_group=process_group
        )
        grad_gathered_input = torch.matmul(grad_gathered_output, weight.t())
        grad_weight = torch.matmul(grad_gathered_output.t(), gathered_input).t()

    return grad_gathered_input, grad_weight


@torch.library.register_fake("xformers_python::sequence_parallel_trailing_matmul_bwd")
def sequence_parallel_trailing_matmul_bwd_fake(
    gathered_input: torch.Tensor,
    weight: torch.Tensor,
    grad_scattered_output: torch.Tensor,
    fuse: bool,
    process_group_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (torch.empty_like(gathered_input), torch.empty_like(weight))


def sequence_parallel_trailing_matmul_setup_context(ctx, inputs, output):
    gathered_input, weight, fuse, process_group_name = inputs
    ctx.save_for_backward(gathered_input, weight)
    ctx.fuse = fuse
    ctx.process_group_name = process_group_name


def sequence_parallel_trailing_matmul_bwd_bridge(ctx, grad_scattered_output):
    gathered_input, weight = ctx.saved_tensors
    (
        grad_gathered_input,
        grad_weight,
    ) = sequence_parallel_trailing_matmul_bwd(
        gathered_input,
        weight,
        grad_scattered_output,
        ctx.fuse,
        ctx.process_group_name,
    )
    return grad_gathered_input, grad_weight, None, None


torch.library.register_autograd(
    "xformers_python::sequence_parallel_trailing_matmul_fwd",
    sequence_parallel_trailing_matmul_bwd_bridge,
    setup_context=sequence_parallel_trailing_matmul_setup_context,
)


def sequence_parallel_trailing_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    fuse: bool,
    process_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    o = sequence_parallel_trailing_matmul_fwd(
        x.flatten(0, -2), w, fuse, process_group.group_name
    )
    return o.view(-1, *x.shape[1:-1], w.shape[1])
