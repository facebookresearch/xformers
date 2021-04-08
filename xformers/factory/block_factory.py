from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn

from xformers.components import MultiHeadDispatchConfig, build_multi_head_attention
from xformers.components.attention import AttentionConfig  # noqa
from xformers.components.feedforward import FeedforwardConfig, build_feedforward
from xformers.components.positional_encoding import (
    PositionEncodingConfig,
    build_positional_encoding,
)


class BlockType(str, Enum):
    Encoder = "encoder"
    Decoder = "decoder"


@dataclass
class _xFormerBlockConfig:
    dim_model: int
    feedforward_config: FeedforwardConfig
    position_encoding_config: Optional[PositionEncodingConfig]

    def __post_init__(self):
        self.feedforward_config = FeedforwardConfig(**self.feedforward_config)
        if self.position_encoding_config:
            self.position_encoding_config = PositionEncodingConfig(
                **self.position_encoding_config
            )


@dataclass
class xFormerEncoderConfig(_xFormerBlockConfig):
    attention_config: AttentionConfig
    multi_head_config: MultiHeadDispatchConfig
    block_type: BlockType = field(default_factory=lambda: BlockType("encoder"))

    def __post_init__(self):
        try:
            super().__post_init__()
            self.attention_config = AttentionConfig(**self.attention_config)
            self.multi_head_config = MultiHeadDispatchConfig(**self.multi_head_config)
            self.block_type = BlockType(self.block_type)
        except TypeError:
            pass


@dataclass
class xFormerDecoderConfig(_xFormerBlockConfig):
    attention_configs: Tuple[AttentionConfig, AttentionConfig]
    multi_head_configs: Tuple[MultiHeadDispatchConfig, MultiHeadDispatchConfig]
    block_type: BlockType = field(default_factory=lambda: BlockType("decoder"))

    def __post_init__(self):
        try:
            super().__post_init__()
            self.attention_configs = tuple(
                AttentionConfig(**c) for c in self.attention_configs
            )
            self.multi_head_configs = tuple(
                MultiHeadDispatchConfig(**c) for c in self.multi_head_configs
            )
            self.block_type = BlockType(self.block_type)
        except TypeError:
            pass


class xFormerEncoderBlock(nn.Module):
    r""" A vanilla Transformer Encoder block """

    def __init__(self, config: xFormerEncoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim_model)
        self.ln2 = nn.LayerNorm(config.dim_model)

        self.pose_encoding = (
            build_positional_encoding(config.position_encoding_config)
            if config.position_encoding_config
            else None
        )

        self.attn = build_multi_head_attention(
            config.attention_config,
            config.multi_head_config,
        )
        self.ff = build_feedforward(config.feedforward_config)

    @classmethod
    def from_config(cls, config: xFormerEncoderConfig):
        return cls(config)

    def forward(self, x):
        if self.pose_encoding:
            x = self.pose_encoding(x)

        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.ff(x))
        return x


class xFormerDecoderBlock(nn.Module):
    r""" A vanilla Transformer Decoder block """

    def __init__(self, config: xFormerDecoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.dim_model, config.feedforward_config.dim_latent)
        self.linear2 = nn.Linear(config.dim_model, config.feedforward_config.dim_latent)

        self.ln1 = nn.LayerNorm(config.dim_model)
        self.ln2 = nn.LayerNorm(config.dim_model)
        self.ln3 = nn.LayerNorm(config.dim_model)

        self.pose_encoding = (
            build_positional_encoding(config.position_encoding_config)
            if config.position_encoding_config
            else None
        )

        self.attn1 = build_multi_head_attention(
            config.attention_configs[0], config.multi_head_configs[0]
        )
        self.attn2 = build_multi_head_attention(
            config.attention_configs[1], config.multi_head_configs[1]
        )

        self.ff = build_feedforward(config.feedforward_config)

    @classmethod
    def from_config(cls, config: xFormerDecoderConfig):
        return cls(config)

    def forward(self, target: torch.Tensor, memory: torch.Tensor):
        if self.pose_encoding:
            target = self.pose_encoding(target)

        # Masked multi head attention
        x = self.ln1(target + self.attn1(target, target, target))

        # Include the memory/Encoder results
        x = self.ln2(x + self.attn2(key=memory, value=memory, query=x))

        # FF
        x = self.ln3(x + self.ff(x))
        return x
