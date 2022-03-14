# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from xformers.components import (
    LayerNormStyle,
    PostNorm,
    PreNorm,
    Residual,
    build_multi_head_attention,
)
from xformers.components.feedforward import (
    FEEDFORWARD_REGISTRY,
    FeedforwardConfig,
    build_feedforward,
)
from xformers.components.positional_embedding import (
    POSITION_EMBEDDING_REGISTRY,
    PositionEmbeddingConfig,
    build_positional_embedding,
)
from xformers.components.residual import get_deepnorm_coefficients
from xformers.utils import generate_matching_config


class LayerPositionBitmask(int, Enum):
    First = 0b01
    Last = 0b10
    Default = 0b11


class LayerPosition:
    """ Bitmask to mark this layer as first, last, nothing or both"""

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


def _get_ln_factory(
    d_model: int,
    layer_norm_style: Optional[LayerNormStyle],
    use_triton: bool,
    residual: bool,
    residual_scale: float = 1.0,
):
    """
    Handle all the supported residual path configurations.

    ..Note: we return the appropriate constructor, not an actual layer
    """

    def get_layer_wrapper(
        d_model: int,
        sublayer: nn.Module,
        layer_norm_style: Optional[LayerNormStyle],
        residual: bool,
        residual_scale: float,
    ):
        if residual:
            if layer_norm_style == LayerNormStyle.Pre:
                return Residual(
                    layer=PreNorm(d_model, sublayer, use_triton), scale=None
                )
            elif layer_norm_style == LayerNormStyle.Post:
                return PostNorm(
                    d_model, Residual(layer=sublayer, scale=None), use_triton
                )
            elif layer_norm_style == LayerNormStyle.DeepNorm:
                return PostNorm(
                    d_model,
                    Residual(layer=sublayer, scale=residual_scale),
                    use_triton=use_triton,
                )
            else:
                raise ValueError

        return (
            PreNorm(d_model, sublayer, use_triton)
            if layer_norm_style == LayerNormStyle.Pre
            else PostNorm(d_model, sublayer, use_triton)
        )

    def ln_factory(sublayer: nn.Module):
        return get_layer_wrapper(
            d_model, sublayer, layer_norm_style, residual, residual_scale
        )

    return ln_factory


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
    layer_norm_style: LayerNormStyle
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
        layer_norm_style: LayerNormStyle = LayerNormStyle("post"),
        reversible: bool = False,
        num_layers: int = 1,
        layer_position: Optional[LayerPosition] = None,
    ):

        self.dim_model = dim_model
        self.block_type = block_type
        self.layer_norm_style = layer_norm_style
        self.reversible = reversible
        self.num_layers = num_layers

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

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        multi_head_config: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]] = None,
        layer_norm_style: str = "post",
        use_triton: bool = True,
        **kwargs,
    ):
        # Convenience, fill in duplicated field
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
            layer_norm_style=LayerNormStyle(layer_norm_style),
            **kwargs,
        )

        self.multi_head_config = multi_head_config
        self.use_triton = use_triton


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
        layer_norm_style: str = "post",
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
            layer_norm_style=LayerNormStyle(layer_norm_style),
            **kwargs,
        )

        self.multi_head_config_masked = multi_head_config_masked
        self.multi_head_config_cross = multi_head_config_cross
        self.use_triton = use_triton


class xFormerEncoderBlock(torch.nn.Module):
    r""" A vanilla Transformer Encoder block """

    def __init__(self, config: xFormerEncoderConfig, **kwargs):
        super().__init__()

        self.reversible_f = None
        self.reversible_g = None
        self.layer_norm_style = config.layer_norm_style
        self.dim_model = config.dim_model

        # If this layer is the first one, and a pose encoding has been requested
        self.pose_encoding = (
            build_positional_embedding(asdict(config.position_encoding_config))
            if config.position_encoding_config and config.layer_position.is_first()
            else None
        )

        if config.layer_norm_style == LayerNormStyle.DeepNorm:
            # Just use the layer norm coefficient here,
            # the init will be handled at the xformers level (knows about encoder and decoder blocks)
            deep_norm_coefficients, _ = get_deepnorm_coefficients(
                encoder_layers=config.num_layers, decoder_layers=0
            )
            assert deep_norm_coefficients is not None
            residual_scale = deep_norm_coefficients.alpha
        else:
            residual_scale = 1.0

        # mini helper, builds a LayerNorm with the right Pre/Post config, residuals, and the right dimensions
        ln_factory = _get_ln_factory(
            config.dim_model,
            config.layer_norm_style,
            use_triton=config.use_triton,
            residual=True,
            residual_scale=residual_scale,
        )

        self.mha = build_multi_head_attention(config.multi_head_config)
        self.feedforward = build_feedforward(asdict(config.feedforward_config))

        # Wrappers handle the different layer norm styles (pre- and post-) and the residual path
        self.wrap_att = ln_factory(self.mha)
        self.wrap_ff: Union[Residual, PostNorm] = ln_factory(self.feedforward)
        if (
            config.layer_norm_style == LayerNormStyle.Pre
            and config.layer_position.is_last()
        ):
            self.wrap_ff = PostNorm(config.dim_model, self.wrap_ff)

    @classmethod
    def from_config(cls, config: xFormerEncoderConfig):
        return cls(config)

    @staticmethod
    def get_reversible_layer(config) -> Tuple[nn.Module, nn.Module]:
        ln_factory = _get_ln_factory(
            config.dim_model,
            config.layer_norm_style,
            residual=False,
            use_triton=config.use_triton,
        )

        mha = build_multi_head_attention(config.multi_head_config)
        feedforward = build_feedforward(asdict(config.feedforward_config))

        reversible_f = ln_factory(mha)
        reversible_g = ln_factory(feedforward)
        return reversible_f, reversible_g

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if self.pose_encoding:
            x = self.pose_encoding(x)

        # Handle the optional input masking, differs on Q, K, V
        if input_mask is not None:
            q = x
            k = x * input_mask.unsqueeze(-1)
            v = k
        else:
            q, k, v = x, x, x

        # Pre/Post norms and residual paths are already handled
        x = self.wrap_att(inputs=[q, k, v], att_mask=att_mask)
        x = self.wrap_ff(inputs=[x])

        return x


class xFormerDecoderBlock(torch.nn.Module):
    r"""A vanilla Transformer Decoder block

    ... note: this implementation is not (yet ?) reversible"""

    def __init__(self, config: xFormerDecoderConfig, **kwargs):
        super().__init__()

        # If this layer is the first one, and a pose encoding as been requested
        self.pose_encoding = (
            build_positional_embedding(config.position_encoding_config)
            if config.position_encoding_config and config.layer_position.is_first()
            else None
        )

        if config.layer_norm_style == LayerNormStyle.DeepNorm:
            # Just use the layer norm coefficient here,
            # the init will be handled at the xformers level (knows about encoder and decoder blocks)
            _, deep_norm_coefficients = get_deepnorm_coefficients(
                encoder_layers=0, decoder_layers=config.num_layers
            )
            assert deep_norm_coefficients is not None
            residual_scale = deep_norm_coefficients.alpha
        else:
            residual_scale = 1.0

        # mini helper, builds a LayerNorm with the right Pre/Post config and the right dimensions
        ln_factory = _get_ln_factory(
            config.dim_model,
            config.layer_norm_style,
            use_triton=config.use_triton,
            residual=True,
            residual_scale=residual_scale,
        )

        self.mha = build_multi_head_attention(config.multi_head_config_masked)
        self.cross_mha = build_multi_head_attention(config.multi_head_config_cross)
        self.feedforward = build_feedforward(config.feedforward_config)

        self.wrap_att = ln_factory(self.mha)
        self.wrap_cross = ln_factory(self.cross_mha)
        self.wrap_ff: Union[Residual, PostNorm] = ln_factory(self.feedforward)
        if (
            config.layer_norm_style == LayerNormStyle.Pre
            and config.layer_position.is_last()
        ):
            self.wrap_ff = PostNorm(config.dim_model, self.wrap_ff)

    @classmethod
    def from_config(cls, config: xFormerDecoderConfig):
        return cls(config)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        encoder_att_mask: Optional[torch.Tensor] = None,
        decoder_att_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if self.pose_encoding:
            target = self.pose_encoding(target)

        # Handle the optional input masking, differs on Q, K, V
        if input_mask is not None:
            target_q = target
            target_k = target * input_mask.unsqueeze(-1)
            target_v = target_k
        else:
            target_q, target_k, target_v = target, target, target

        x = self.wrap_att(
            inputs=[target_q, target_k, target_v], att_mask=decoder_att_mask
        )
        x = self.wrap_cross(inputs=[x, memory, memory], att_mask=encoder_att_mask)
        x = self.wrap_ff(inputs=[x])

        return x
