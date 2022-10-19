# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, Union

import torch

from xformers.components import Activation
from xformers.components.feedforward import (
    Feedforward,
    FeedforwardConfig,
    register_feedforward,
)

logger = logging.getLogger("xformers")


_is_fairscale_available = True

try:
    import torch.distributed as dist
    from fairscale.nn import MOELayer, Top2Gate  # type: ignore

    from xformers.components.feedforward import MLP

except ImportError:
    logger.warning(
        "Either FairScale or torch distributed is not available, MixtureOfExperts will not be exposed."
        " Please install them if you would like to use MoE"
    )
    _is_fairscale_available = False


if _is_fairscale_available:

    # Credits: initially implemented in FairScale for sanity checking
    class RoundRobinGate(torch.nn.Module):
        def __init__(self, model_dim, num_experts):
            super().__init__()
            self.model_dim = model_dim
            self.num_experts = num_experts

        def forward(self, input):
            s = input.shape[0]
            assert s % self.num_experts == 0, f"{s} % {self.num_experts} != 0"
            capacity = 2 * s // self.num_experts
            output = torch.zeros(
                s, self.num_experts, capacity, dtype=input.dtype, device=input.device
            )
            for i in range(s):
                output[i, i % self.num_experts, i // self.num_experts] = 1.0
            return 0.0, output, output.bool()

    class GateConfig(str, Enum):
        RoundRobin = "round_robin"
        Top2 = "top_2"
        # Other gating techniques could be exposed here

    @dataclass
    class MoEConfig(FeedforwardConfig):
        number_of_experts: int
        gate: GateConfig
        number_of_local_experts: Optional[int] = None
        expert_constructor: Optional[Any] = None
        hidden_layer_multiplier: Optional[int] = None
        group: Optional[Any] = None

    @register_feedforward("MixtureOfExperts", MoEConfig)
    class MixtureOfExperts(Feedforward):
        """
        A MLP variant which uses the "Mixture of Experts" paradigm, as described in Gshard_.
        xFormers uses the FairScale_ implementation under the hood.

        .. warning: Please note that most of the benefits of MoE are present in a distributed training environmentt

        .. _Gshard: https://arxiv.org/pdf/2006.16668.pdf
        .. _FairScale: https://github.com/facebookresearch/fairscale/
        """

        def __init__(
            self,
            dim_model: int,
            dropout: float,
            activation: Activation,
            number_of_experts: int,
            gate: Union[GateConfig, torch.nn.Module],
            number_of_local_experts: Optional[int] = None,
            expert_constructor: Optional[Callable[[], torch.nn.Module]] = None,
            hidden_layer_multiplier: Optional[int] = None,
            group: Optional[Any] = None,
            *_,
            **__,
        ):
            super().__init__()

            # Handle a possibly uninitialized process group
            assert (
                dist.is_initialized()
            ), "Mixture of Experts require torch distributed to be initialized"

            if number_of_local_experts is not None:
                assert number_of_experts >= number_of_local_experts
            else:
                if dist.get_world_size() == 1:
                    logger.warning("Local experts no specified but world size of 1")
                    logger.warning("Assuming that all experts are local")
                    number_of_local_experts = number_of_experts
                else:
                    number_of_local_experts = 1

            # Programatically handle the gating technique
            if not isinstance(gate, torch.nn.Module):
                gate_constructor = {
                    GateConfig.RoundRobin: RoundRobinGate,
                    GateConfig.Top2: Top2Gate,
                }[gate]

                self.gate = gate_constructor(dim_model, number_of_experts)
            else:
                self.gate = gate

            # Programatically handle the experts
            if expert_constructor is None:

                multiplier = (
                    hidden_layer_multiplier
                    if hidden_layer_multiplier is not None
                    else 4
                )

                def expert_constructor() -> torch.nn.Module:
                    return MLP(dim_model, dropout, activation, multiplier)

                assert expert_constructor is not None

            local_experts = torch.nn.ModuleList(
                [expert_constructor() for _ in range(number_of_local_experts)]
            )

            self.moe = MOELayer(gate=self.gate, experts=local_experts, group=group)

            self.requires_cuda = True

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            # FairScale MoE assumes that the dimensions are [S, B, E]
            # xFormers assumes [B, S, E]
            return self.moe(inputs.movedim(0, 1)).movedim(0, 1)
