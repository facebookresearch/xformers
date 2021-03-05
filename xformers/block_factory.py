from dataclasses import dataclass

import torch.nn as nn

from xformers.components.attention import AttentionConfig, build_attention  # noqa
from xformers.components.feedforward import FeedforwardConfig, build_feedforward
from xformers.components.positional_encoding import PositionEncodingConfig


@dataclass
class xFormerConfig:
    dim_model: int
    attention_config: AttentionConfig
    feedforward_config: FeedforwardConfig
    position_encoding_config: PositionEncodingConfig


class xFormerBlock(nn.Module):
    """ a vanilla Transformer block """

    def __init__(self, config: xFormerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim_model)
        self.ln2 = nn.LayerNorm(config.dim_model)

        self.attn = build_attention(config.attention_config)
        self.ff = build_feedforward(config.feedforward_config)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        return cls(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
