from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

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


@dataclass(init=False)
class _xFormerBlockConfig:
    dim_model: int
    feedforward_config: FeedforwardConfig
    position_encoding_config: Optional[PositionEmbeddingConfig]

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        position_encoding_config: Optional[Dict[str, Any]],
    ):
        self.dim_model = dim_model

        self.feedforward_config = generate_matching_config(
            feedforward_config, FEEDFORWARD_REGISTRY[feedforward_config["name"]].config
        )

        self.position_encoding_config = (
            generate_matching_config(
                position_encoding_config,
                POSITION_EMBEDDING_REGISTRY[position_encoding_config["name"]].config,
            )
            if position_encoding_config is not None
            else None
        )


@dataclass(init=False)
class xFormerEncoderConfig(_xFormerBlockConfig):
    multi_head_config: Dict[str, Any]
    block_type: BlockType
    num_layers: int
    layer_norm_style: LayerNormStyle

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        position_encoding_config: Dict[str, Any],
        multi_head_config: Dict[str, Any],
        block_type=BlockType("encoder"),
        num_layers=1,
        layer_norm_style=LayerNormStyle.Post,
    ):
        super().__init__(dim_model, feedforward_config, position_encoding_config)
        self.num_layers = num_layers
        self.block_type = block_type
        self.layer_norm_style = layer_norm_style
        self.multi_head_config = multi_head_config


@dataclass(init=False)
class xFormerDecoderConfig(_xFormerBlockConfig):
    multi_head_config_pre_encoder: Dict[str, Any]
    multi_head_config_post_encoder: Dict[str, Any]

    block_type: BlockType = field(default_factory=lambda: BlockType("decoder"))
    num_layers: int = 1
    layer_norm_style: LayerNormStyle = LayerNormStyle.Post

    def __init__(
        self,
        dim_model: int,
        feedforward_config: Dict[str, Any],
        position_encoding_config: Dict[str, Any],
        multi_head_config_pre_encoder: Dict[str, Any],
        multi_head_config_post_encoder: Dict[str, Any],
        block_type=BlockType("encoder"),
        num_layers=1,
        layer_norm_style=LayerNormStyle.Post,
    ):
        super().__init__(dim_model, feedforward_config, position_encoding_config)
        self.num_layers = num_layers
        self.block_type = block_type
        self.layer_norm_style = layer_norm_style
        self.multi_head_config_pre_encoder = multi_head_config_pre_encoder
        self.multi_head_config_post_encoder = multi_head_config_post_encoder


class xFormerEncoderBlock(nn.Module):
    r""" A vanilla Transformer Encoder block """

    def __init__(self, config: xFormerEncoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim_model)
        self.ln2 = nn.LayerNorm(config.dim_model)

        self.pose_encoding = (
            build_positional_embedding(asdict(config.position_encoding_config))
            if config.position_encoding_config
            else None
        )

        self.attn = build_multi_head_attention(config.multi_head_config)
        self.ff = build_feedforward(asdict(config.feedforward_config))
        self.layer_norm_style = config.layer_norm_style

    @classmethod
    def from_config(cls, config: xFormerEncoderConfig):
        return cls(config)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):
        if self.pose_encoding:
            x = self.pose_encoding(x)

        if input_mask is not None:
            # The mask acts as an input bias. In particular, nulling the influence of some elements
            # can be done by setting the corresponding mask to '-float("inf")'
            q = x
            k = x + input_mask.unsqueeze(-1)
            v = k
        else:
            q, k, v = x, x, x

        if self.layer_norm_style == LayerNormStyle.Post:
            x = self.ln1(x + self.attn(query=q, key=k, value=v, att_mask=att_mask))
            x = self.ln2(x + self.ff(x))

        else:
            x_norm = self.ln1(x)
            x = x + self.attn(x_norm, x_norm, x_norm, att_mask)
            x = x + self.ff(self.ln2(x))
        return x


class xFormerDecoderBlock(nn.Module):
    r""" A vanilla Transformer Decoder block """

    def __init__(self, config: xFormerDecoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.dim_model, config.feedforward_config.dim_model)
        self.linear2 = nn.Linear(config.dim_model, config.feedforward_config.dim_model)

        self.ln1 = nn.LayerNorm(config.dim_model)
        self.ln2 = nn.LayerNorm(config.dim_model)
        self.ln3 = nn.LayerNorm(config.dim_model)

        self.pose_encoding = (
            build_positional_embedding(config.position_encoding_config)
            if config.position_encoding_config
            else None
        )

        self.attn1 = build_multi_head_attention(config.multi_head_config_pre_encoder)
        self.attn2 = build_multi_head_attention(config.multi_head_config_post_encoder)

        self.ff = build_feedforward(config.feedforward_config)
        self.layer_norm_style = config.layer_norm_style

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

        if input_mask is not None:
            # The mask acts as an input bias. In particular, nulling the influence of some elements
            # can be done by setting the corresponding mask to '-float("inf")'
            target += input_mask.unsqueeze(-1)

        if input_mask is not None:
            # The mask acts as an input bias. In particular, nulling the influence of some elements
            # can be done by setting the corresponding mask to '-float("inf")'
            target_q = target
            target_k = target + input_mask.unsqueeze(-1)
            target_v = target_k
        else:
            target_q, target_k, target_v = target, target, target

        if self.layer_norm_style == LayerNormStyle.Post:
            # Masked multi head attention
            x = self.ln1(
                target
                + self.attn1(target_q, target_k, target_v, att_mask=decoder_att_mask)
            )

            # Include the memory/Encoder results
            x = self.ln2(
                x
                + self.attn2(
                    key=memory, value=memory, query=x, att_mask=encoder_att_mask
                )
            )

            # FF
            x = self.ln3(x + self.ff(x))
        else:
            # Masked multi head attention
            target_norm = self.ln1(target)
            x = target + self.attn1(
                target_norm, target_norm, target_norm, att_mask=decoder_att_mask
            )

            # Include the memory/Encoder results
            x = x + self.attn2(
                key=memory, value=memory, query=self.ln2(x), att_mask=encoder_att_mask
            )

            # FF
            x = x + self.ff(self.ln3(x))

        return x
