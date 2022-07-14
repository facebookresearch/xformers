# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from xformers.components import NormalizationType, ResidualNormStyle
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, FeedforwardConfig
from xformers.components.positional_embedding import (
    POSITION_EMBEDDING_REGISTRY,
    PositionEmbeddingConfig,
)
from xformers.utils import generate_matching_config


class LayerPositionBitmask(int, Enum):
    First = 0b01
    Last = 0b10
    Default = 0b11


class LayerPosition:
    """Bitmask to mark this layer as first, last, nothing or both"""

    def __init__(self):
        self.bitmask = LayerPositionBitmask.Default

    def is_first(self):
        return bool(self.bitmask & LayerPositionBitmask.First)

    def is_last(self):
        return bool(self.bitmask & LayerPositionBitmask.Last)

    def mark_not_first(self):
        self.bitmask &= ~LayerPositionBitmask.First

    def mark_not_last(self):
        self.bitmask &= ~LayerPositionBitmask.Last


class BlockType(str, Enum):
    Encoder = "encoder"
    Decoder = "decoder"


@dataclass(init=False)  # handle constructors explicitly to force type changes
class xFormerBlockConfig:
    """
    The configuration structure to define a Transformer block.
    This base class is applicable to both encoder and decoder definitions.

    This completely defines each of the blocks, for instance in terms of dimensions,
    position encoding, pre or post layer norms or reversibility.
    """

    dim_model: int
    feedforward_config: FeedforwardConfig
    position_encoding_config: Optional[PositionEmbeddingConfig]
    block_type: BlockType
    residual_norm_style: ResidualNormStyle
    normalization: NormalizationType
    layer_position: LayerPosition
    use_triton: bool
    reversible: bool
    num_layers: int

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]],
        block_type: BlockType,
        residual_norm_style: ResidualNormStyle = ResidualNormStyle("post"),
        normalization: NormalizationType = NormalizationType.LayerNorm,
        reversible: bool = False,
        num_layers: int = 1,
        layer_position: Optional[LayerPosition] = None,
    ):

        self.dim_model = dim_model
        self.block_type = block_type
        self.residual_norm_style = residual_norm_style
        self.reversible = reversible
        self.num_layers = num_layers
        self.normalization = normalization

        # Fill in possible gaps in the config for subparts of the block
        self.feedforward_config = generate_matching_config(
            feedforward_config,
            FEEDFORWARD_REGISTRY[feedforward_config["name"]].config,
        )

        self.position_encoding_config = (
            generate_matching_config(
                position_encoding_config,
                POSITION_EMBEDDING_REGISTRY[position_encoding_config["name"]].config,
            )
            if position_encoding_config is not None
            else None
        )

        # Default is that this layer is the only one, so both first and last
        if layer_position:
            self.layer_position = layer_position
        else:
            self.layer_position = LayerPosition()


@dataclass(init=False)
class xFormerEncoderConfig(xFormerBlockConfig):
    """
    The configuration structure for an encoder block
    """

    multi_head_config: Dict[str, Any]
    use_triton: bool
    simplicial_embeddings: Optional[Dict[str, Any]]
    patch_embedding_config: Optional[Dict[str, Any]]

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        multi_head_config: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]] = None,
        residual_norm_style: str = "post",
        normalization: NormalizationType = NormalizationType.LayerNorm,
        use_triton: bool = True,
        simplicial_embeddings: Optional[Dict[str, Any]] = None,
        patch_embedding_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # Convenience, fill in duplicated fields
        try:
            if "dim_model" not in multi_head_config.keys():
                multi_head_config["dim_model"] = dim_model

            if "dim_model" not in feedforward_config.keys():
                feedforward_config["dim_model"] = dim_model

            if (
                position_encoding_config is not None
                and "dim_model" not in position_encoding_config.keys()
            ):
                position_encoding_config["dim_model"] = dim_model

            if (
                patch_embedding_config is not None
                and "out_channels" not in patch_embedding_config.keys()
            ):
                patch_embedding_config["out_channels"] = dim_model

        except AttributeError:
            # A config instance was passed in, this is fine
            pass
        if "block_type" in kwargs:
            assert kwargs["block_type"] == "encoder"
        kwargs["block_type"] = BlockType("encoder")
        super().__init__(
            dim_model=dim_model,
            feedforward_config=feedforward_config,
            position_encoding_config=position_encoding_config,
            residual_norm_style=ResidualNormStyle(residual_norm_style),
            normalization=NormalizationType(normalization),
            **kwargs,
        )

        self.multi_head_config = multi_head_config
        self.use_triton = use_triton
        self.simplicial_embeddings = simplicial_embeddings
        self.patch_embedding_config = patch_embedding_config


@dataclass(init=False)
class xFormerDecoderConfig(xFormerBlockConfig):
    """
    The configuration structure for a decoder block.

    This specifically defines the masked and cross attention mechanisms,
    on top of the settings defining all blocks.
    """

    multi_head_config_masked: Dict[str, Any]  # prior to encoder output
    multi_head_config_cross: Dict[str, Any]  # cross attention, takes encoder output

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        multi_head_config_masked: Dict[str, Any],
        multi_head_config_cross: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]] = None,
        residual_norm_style: str = "post",
        normalization: NormalizationType = NormalizationType.LayerNorm,
        use_triton: bool = True,
        **kwargs,
    ):

        # Convenience, fill in duplicated field
        try:
            if "dim_model" not in multi_head_config_masked.keys():
                multi_head_config_masked["dim_model"] = dim_model

            if "dim_model" not in multi_head_config_cross.keys():
                multi_head_config_cross["dim_model"] = dim_model

            if "dim_model" not in feedforward_config.keys():
                feedforward_config["dim_model"] = dim_model

            if (
                position_encoding_config is not None
                and "dim_model" not in position_encoding_config.keys()
            ):
                position_encoding_config["dim_model"] = dim_model
        except AttributeError:
            # A config instance was passed in, this is fine
            pass
        if "block_type" in kwargs.keys():
            assert kwargs["block_type"] == "decoder"
        kwargs["block_type"] = BlockType("decoder")

        super().__init__(
            dim_model=dim_model,
            feedforward_config=feedforward_config,
            position_encoding_config=position_encoding_config,
            residual_norm_style=ResidualNormStyle(residual_norm_style),
            normalization=NormalizationType(normalization),
            **kwargs,
        )

        self.multi_head_config_masked = multi_head_config_masked
        self.multi_head_config_cross = multi_head_config_cross
        self.use_triton = use_triton
