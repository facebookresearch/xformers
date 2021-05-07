from abc import abstractmethod
from typing import Optional

import torch

from xformers.utils import ExtensibleConfig

"""
Feature maps allow for a given query or key to be encoded in a different space.
"""


class FeatureMapConfig(ExtensibleConfig):
    name: str
    dim_features: int
    iter_before_redraw: Optional[int]


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
    def from_config(cls, config: FeatureMapConfig) -> "FeatureMap":
        return cls(**FeatureMapConfig.as_patchy_dict(config))
