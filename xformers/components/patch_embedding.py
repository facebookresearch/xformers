# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from enum import Enum

import torch


class PoolType(str, Enum):
    Conv2D = "CONV_2D"
    # ...
    # TODO: Support more cases ?


@dataclass
class PatchEmbeddingConfig:
    """
    The configuration for the patch embedding layer, which takes the raw token passed in
    and returns a pooled representation along a given embedding dimension.

    This typically trades the spatial (context length) representation with the embedding size

    This is canonicaly used by ViT, but other papers (like MetaFormer or other hierarchical transformers)
    propose a more general use case for this
    """

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int = 0
    pool_type: PoolType = PoolType.Conv2D


class ConditionalReshape(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        if x.ndim == 3:
            B, HW, C = x.shape
            # NOTE: We're assuming a square sample here
            H = int(math.sqrt(HW))
            assert H * H == HW, f"{H, HW}"
            x = x.transpose(1, 2).reshape(B, C, H, H)

        return x


class PatchToSequence(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.flatten(2, 3).transpose(1, 2).contiguous()  # B HW C


def build_patch_embedding(config: PatchEmbeddingConfig):
    if not isinstance(config, PatchEmbeddingConfig):
        config = PatchEmbeddingConfig(**config)

    if config.pool_type == PoolType.Conv2D:
        pool = torch.nn.Conv2d(
            config.in_channels,
            config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
        )
    else:
        raise NotImplementedError

    # The patch embedding supposes that the input really is 2D in essence
    # If this block is in the middle of a stack, we need to reshape
    return torch.nn.Sequential(ConditionalReshape(), pool, PatchToSequence())
