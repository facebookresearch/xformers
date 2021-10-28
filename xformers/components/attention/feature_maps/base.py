# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar

import torch

"""
Feature maps allow for a given query or key to be encoded in a different space.
"""

Self = TypeVar("Self", bound="FeatureMap")


@dataclass
class FeatureMapConfig:
    name: str
    dim_features: int
    iter_before_redraw: Optional[int]
    normalize_inputs: Optional[bool]
    epsilon: Optional[float]


class FeatureMap(torch.nn.Module):
    def __init__(
        self,
        dim_features: int,
        iter_before_redraw: Optional[int] = None,
        normalize_inputs: bool = False,
        epsilon: float = 1e-6,
    ):
        super().__init__()

        self.dim_features = dim_features
        self.dim_feature_map = dim_features

        self.iter_before_redraw = iter_before_redraw
        self.features: Optional[torch.Tensor] = None
        self.epsilon = epsilon
        self.normalize_inputs = normalize_inputs

        self._iter_counter = 0

    @abstractmethod
    def _get_feature_map(self, dim_input: int, dim_features: int, device: torch.device):
        raise NotImplementedError()

    @classmethod
    def from_config(cls: Type[Self], config: FeatureMapConfig) -> Self:
        # Generate the class inputs from the config
        fields = asdict(config)

        # Skip all Nones so that default values are used
        fields = {k: v for k, v in fields.items() if v is not None}

        return cls(**fields)
