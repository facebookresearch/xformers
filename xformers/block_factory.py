from dataclasses import dataclass

import torch.nn as nn

from xformers.components.attention import (  # noqa
    AttentionConfig,
    MultiHeadDispatchConfig,
    build_multi_head_attention,
)
from xformers.components.feedforward import FeedforwardConfig, build_feedforward
from xformers.components.positional_encoding import PositionEncodingConfig


@dataclass
class xFormerConfig:
    dim_model: int
    attention_config: AttentionConfig
    multi_head_config: MultiHeadDispatchConfig
    feedforward_config: FeedforwardConfig
    position_encoding_config: PositionEncodingConfig


class xFormerBlock(nn.Module):
    """ a vanilla Transformer block """

    def __init__(self, config: xFormerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim_model)
        self.ln2 = nn.LayerNorm(config.dim_model)

        self.attn = build_multi_head_attention(
            config.attention_config, config.multi_head_config
        )
        self.ff = build_feedforward(config.feedforward_config)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
