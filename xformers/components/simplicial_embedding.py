# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar

import torch

from xformers import _is_triton_available

Self = TypeVar("Self", bound="SimplicialEmbedding")


@dataclass
class SimplicialEmbeddingConfig:
    L: int
    temperature: float


class SimplicialEmbedding(torch.nn.Module):
    """
    An implementation of the "Simplicial Embeddings"_, as proposed by Lavoie et. al

    Arguments:
        - L: the number of embedding chunks
        - temperature: optional scaling parameter for the softmax operation.
            A small (<1.) temperature will lead to a sparse representation (up to one-hot),
            while a large (>1.) temperature will make the vector more uniform

    _"Simplicial Embeddings": https://arxiv.org/pdf/2204.00616.pdf
    """

    def __init__(self, L: int, temperature: Optional[float] = None) -> None:
        super().__init__()
        self.L = L
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] % self.L == 0
        ), f"The embedding dimension {x.shape[-1]} is not divisible by the chosen L parameter {self.L}"

        # Seperate the input tensor into V chunks
        B, C, E = x.shape
        V = E // self.L

        Vs = x.reshape(B, C, self.L, V)

        # Softmax normalize them, with the proposed temperature
        # This is done over the last dimension, so only within Vs
        if self.temperature is not None:
            Vs /= self.temperature

        if _is_triton_available():
            from xformers.triton.softmax import softmax as triton_softmax

            Vs = triton_softmax(
                Vs, mask=None, causal=False
            )  # the softmax is on the last dimension
        else:
            Vs = torch.nn.functional.softmax(Vs, dim=-1)

        # Concatenate back and return
        return Vs.reshape(B, C, E)

    @classmethod
    def from_config(cls: Type[Self], config: SimplicialEmbeddingConfig) -> Self:
        # Generate the class inputs from the config
        fields = asdict(config)

        return cls(**fields)
