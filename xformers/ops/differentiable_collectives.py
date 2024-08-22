# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

import torch
import torch.distributed


def all_reduce(
    x: torch.Tensor, *, process_group: torch.distributed.ProcessGroup
) -> None:
    mp_size = process_group.size()
    if mp_size == 1:
        return

    torch.distributed.all_reduce(
        tensor=x, op=torch.distributed.ReduceOp.SUM, group=process_group
    )


def gather_along_first_dim_async(
    input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    mp_size = process_group.size()
    if mp_size == 1:
        return input_, None

    output = input_.new_empty((input_.shape[0] * mp_size,) + input_.shape[1:])
    handle = torch.distributed.all_gather_into_tensor(
        output_tensor=output,
        input_tensor=input_,
        group=process_group,
        async_op=True,
    )

    return output, handle


def reduce_scatter_along_first_dim_async(
    input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    mp_size = process_group.size()
    if mp_size == 1:
        return input_, None

    assert input_.shape[0] % mp_size == 0
    output = input_.new_empty((input_.shape[0] // mp_size,) + input_.shape[1:])
    handle = torch.distributed.reduce_scatter_tensor(
        output=output,
        input=input_,
        op=torch.distributed.ReduceOp.SUM,
        group=process_group,
        async_op=True,
    )

    return output, handle


def gather_along_first_dim(
    input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    output, handle = gather_along_first_dim_async(input_, process_group=process_group)
    if handle is not None:
        handle.wait()
    return output


def reduce_scatter_along_first_dim(
    input_: torch.Tensor, *, process_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    output, handle = reduce_scatter_along_first_dim_async(
        input_, process_group=process_group
    )
    if handle is not None:
        handle.wait()
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx, input_: torch.Tensor, process_group: torch.distributed.ProcessGroup
    ) -> torch.Tensor:
        ctx.process_group = process_group
        return input_

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        all_reduce(grad_output, process_group=ctx.process_group)
        return grad_output, None


def copy_to_model_parallel_region(
    x: torch.Tensor, process_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    return _CopyToModelParallelRegion.apply(x, process_group)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx, input_: torch.Tensor, process_group: torch.distributed.ProcessGroup
    ) -> torch.Tensor:
        all_reduce(input_, process_group=process_group)
        ctx.mark_dirty(input_)
        return input_

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        return grad_output, None


def reduce_from_model_parallel_region(
    x: torch.Tensor, process_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    return _ReduceFromModelParallelRegion.apply(x, process_group)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, process_group: torch.distributed.ProcessGroup
    ) -> torch.Tensor:
        ctx.process_group = process_group
        return gather_along_first_dim(x, process_group=process_group)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        return (
            reduce_scatter_along_first_dim(
                grad_output, process_group=ctx.process_group
            ),
            None,
        )


def gather_from_sequence_parallel_region(
    x: torch.Tensor, process_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    return _GatherFromSequenceParallelRegion.apply(x, process_group)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx, x: torch.Tensor, process_group: torch.distributed.ProcessGroup
    ) -> torch.Tensor:
        ctx.process_group = process_group
        return reduce_scatter_along_first_dim(x, process_group=process_group)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        return (
            gather_along_first_dim(grad_output, process_group=ctx.process_group),
            None,
        )


def scatter_to_sequence_parallel_region(
    x: torch.Tensor, process_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    return _ScatterToSequenceParallelRegion.apply(x, process_group)
