# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List

import torch
import torch.distributed

from .differentiable_collectives import (
    copy_to_model_parallel_region,
    reduce_from_model_parallel_region,
)
from .seqpar import sequence_parallel_leading_matmul, sequence_parallel_trailing_matmul


def _init_2d_weight(
    weight: torch.Tensor,
    init_method: Callable[[torch.Tensor], torch.Tensor],
    process_group: torch.distributed.ProcessGroup,
    partition_dim: int,
) -> None:
    # Mimick FairScale's _initialize_affine_weight, for backwards compatibility.
    # The reason we initialize the full unpartitioned/gathered weight is so that
    # different ranks get different initial values and thus "break the symmetry"
    # and in order to achieve the same init for any value of model parallelism.
    rank = process_group.rank()
    world_size = process_group.size()

    nrows, ncols = weight.shape
    if partition_dim == 0:
        full_weight = weight.new_empty(nrows * world_size, ncols)
        my_weight_slice = full_weight[rank::world_size, :]
    else:
        full_weight = weight.new_empty(nrows, ncols * world_size)
        my_weight_slice = full_weight[:, rank::world_size]

    init_method(full_weight)

    with torch.no_grad():
        weight.copy_(my_weight_slice)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: List[int],
        *,
        process_group: torch.distributed.ProcessGroup,
        bias: bool = True,
        gather_output: bool = True,
        init_method: Callable[
            [torch.Tensor], torch.Tensor
        ] = torch.nn.init.xavier_normal_,
        sequence_parallel: bool = False,
        fuse_sequence_parallel: bool = True,
    ) -> None:
        super(ColumnParallelLinear, self).__init__()

        if not isinstance(out_features, list):
            raise TypeError(
                "xFormers's implementation of ColumnParallelLinear requires out_features to be a list"
            )
        if bias:
            raise ValueError(
                "xFormers's implementation of ColumnParallelLinear requires bias=False"
            )
        if gather_output:
            raise ValueError(
                "xFormers's implementation of ColumnParallelLinear requires gather_output=False"
            )

        self.in_features = in_features
        self.global_out_features = out_features
        self.sequence_parallel = sequence_parallel
        self.fuse_sequence_parallel = fuse_sequence_parallel
        self.process_group = process_group
        mp_size = process_group.size()
        assert all(dim % mp_size == 0 for dim in out_features)
        self.my_out_features = [dim // mp_size for dim in out_features]

        self.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty((dim, in_features)))
                for dim in self.my_out_features
            ]
        )

        for w in self.weights:
            _init_2d_weight(w, init_method, process_group, partition_dim=0)

    def forward(self, input_: torch.Tensor) -> List[torch.Tensor]:
        if self.sequence_parallel:
            outputs = sequence_parallel_leading_matmul(
                input_,
                [w.t() for w in self.weights],
                fuse=self.fuse_sequence_parallel,
                process_group=self.process_group,
            )
        else:
            input_ = copy_to_model_parallel_region(input_, self.process_group)
            outputs = [torch.matmul(input_, w.t()) for w in self.weights]
        return outputs


class RowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        process_group: torch.distributed.ProcessGroup,
        bias: bool = True,
        input_is_parallel: bool = False,
        init_method: Callable[
            [torch.Tensor], torch.Tensor
        ] = torch.nn.init.xavier_normal_,
        sequence_parallel: bool = False,
        fuse_sequence_parallel: bool = True,
    ):
        super(RowParallelLinear, self).__init__()

        if bias:
            raise ValueError(
                "xFormers's implementation of RowParallelLinear requires bias=False"
            )
        if not input_is_parallel:
            raise ValueError(
                "xFormers's implementation of RowParallelLinear requires input_is_parallel=True"
            )

        self.global_in_features = in_features
        self.out_features = out_features
        self.sequence_parallel = sequence_parallel
        self.fuse_sequence_parallel = fuse_sequence_parallel
        self.process_group = process_group
        mp_size = process_group.size()
        assert in_features % mp_size == 0
        self.my_in_features = in_features // mp_size

        self.weight = torch.nn.Parameter(
            torch.empty((out_features, self.my_in_features))
        )

        _init_2d_weight(self.weight, init_method, process_group, partition_dim=1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            output = sequence_parallel_trailing_matmul(
                input_,
                self.weight.t(),
                fuse=self.fuse_sequence_parallel,
                process_group=self.process_group,
            )
        else:
            output = torch.matmul(input_, self.weight.t())
            output = reduce_from_model_parallel_region(output, self.process_group)
        return output
