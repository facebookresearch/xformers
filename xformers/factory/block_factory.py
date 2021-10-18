# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from xformers.components import build_multi_head_attention
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
from xformers.utils import generate_matching_config

# NOTE: The Triton layernorm can be activated/deactivated from here
_is_triton_available = False

if _is_triton_available:
    try:
        from xformers.triton.layer_norm import FusedLayerNorm
    except ImportError as e:
        import logging

        logging.warning(
            f"Triton is not available, some optimizations will not be enabled.\n{e}"
        )
        _is_triton_available = False


def _to_tensor_list(
    inputs: Union[torch.Tensor, List[torch.Tensor]]
) -> List[torch.Tensor]:
    if not isinstance(inputs, list):
        inputs = [inputs]
    return inputs


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


class LayerNormStyle(str, Enum):
    """Support different layer norm styles.
    See "On Layer Normalization in the Transformer Architecture",
    Xiong et al., https://arxiv.org/pdf/2002.04745v1.pdf
    """

    Pre = "pre"
    Post = "post"


# CREDITS: the following is inspired by FastAI's Transformer implementation
class Residual(nn.Module):
    """Object-oriented handling of the residual path"""

    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        inputs = _to_tensor_list(inputs)

        return inputs[0] + self.layer(*inputs, *args, **kwargs)


class PreNorm(nn.Module):
    """Adds LayerNorm before computing attention

    ..Note: If a list of inputs is passed, all of them get normalized"""

    def __init__(self, d_model: int, sublayer: nn.Module, use_triton: bool = True):
        super().__init__()
        if _is_triton_available and use_triton:
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.sublayer = sublayer

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        inputs = _to_tensor_list(inputs)

        x_norm = [self.norm(x_) for x_ in inputs]
        return self.sublayer(*x_norm, *args, **kwargs)


class PostNorm(nn.Module):
    """Adds LayerNorm after computing attention"""

    def __init__(self, d_model: int, sublayer: nn.Module, use_triton: bool = True):
        super().__init__()
        if _is_triton_available and use_triton:
            self.norm: Union[nn.LayerNorm, FusedLayerNorm] = FusedLayerNorm(d_model)
        else:
            self.norm = nn.LayerNorm(d_model)

        self.sublayer = sublayer

    def forward(self, inputs: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs):
        inputs = _to_tensor_list(inputs)

        x = self.sublayer(*inputs, *args, **kwargs)
        return self.norm(x)


def _get_ln_factory(
    d_model: int,
    layer_norm_style: Optional[LayerNormStyle],
    residual: bool = True,
    use_triton: bool = True,
):
    def get_layer_wrapper(
        d_model: int,
        sublayer: nn.Module,
        layer_norm_style: Optional[LayerNormStyle],
        residual: bool = True,
    ):
        if residual:
            return (
                Residual(PreNorm(d_model, sublayer, use_triton))
                if layer_norm_style == LayerNormStyle.Pre
                else PostNorm(d_model, Residual(sublayer), use_triton)
            )
        return (
            PreNorm(d_model, sublayer, use_triton)
            if layer_norm_style == LayerNormStyle.Pre
            else PostNorm(d_model, sublayer, use_triton)
        )

    def ln_factory(sublayer: nn.Module):
        return get_layer_wrapper(d_model, sublayer, layer_norm_style, residual)

    return ln_factory


@dataclass(init=False)  # handle constructors explicitly to force type changes
class xFormerBlockConfig:
    dim_model: int
    feedforward_config: FeedforwardConfig
    position_encoding_config: Optional[PositionEmbeddingConfig]
    block_type: BlockType
    layer_norm_style: LayerNormStyle
    layer_position: LayerPosition
    use_triton: bool

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]],
        block_type: BlockType,
        layer_norm_style: LayerNormStyle = LayerNormStyle("post"),
        use_triton: bool = True,
    ):
        self.dim_model = dim_model
        self.block_type = block_type
        self.layer_norm_style = layer_norm_style

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
        self.layer_position = LayerPosition()


@dataclass(init=False)
class xFormerEncoderConfig(xFormerBlockConfig):
    multi_head_config: Dict[str, Any]

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        multi_head_config: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]] = None,
        layer_norm_style: str = "post",
        **kwargs,
    ):
        super().__init__(
            dim_model=dim_model,
            feedforward_config=feedforward_config,
            position_encoding_config=position_encoding_config,
            layer_norm_style=LayerNormStyle(layer_norm_style),
            block_type=BlockType("encoder"),
        )

        self.multi_head_config = multi_head_config


@dataclass(init=False)
class xFormerDecoderConfig(xFormerBlockConfig):
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
        **kwargs,
    ):
        super().__init__(
            dim_model=dim_model,
            feedforward_config=feedforward_config,
            position_encoding_config=position_encoding_config,
            layer_norm_style=LayerNormStyle(layer_norm_style),
            block_type=BlockType("encoder"),
        )

        self.multi_head_config_masked = multi_head_config_masked
        self.multi_head_config_cross = multi_head_config_cross


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

        # mini helper, builds a LayerNorm with the right Pre/Post config, residuals, and the right dimensions
        ln_factory = _get_ln_factory(config.dim_model, config.layer_norm_style)

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
            config.dim_model, config.layer_norm_style, residual=False
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
        x = self.wrap_att(q, k, v, att_mask=att_mask)
        x = self.wrap_ff(x)

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

        # mini helper, builds a LayerNorm with the right Pre/Post config and the right dimensions
        ln_factory = _get_ln_factory(config.dim_model, config.layer_norm_style)

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

        x = self.wrap_att([target_q, target_k, target_v], att_mask=decoder_att_mask)
        x = self.wrap_cross([x, memory, memory], att_mask=encoder_att_mask)
        x = self.wrap_ff(x)

        return x
