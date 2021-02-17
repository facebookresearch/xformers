from dataclasses import dataclass

import torch.nn as nn

from xformers.components.attention import AttentionConfig, MultiHeadAttention
from xformers.components.feedforward import MLP, FeedforwardConfig
from xformers.components.positional_encoding import PositionEncodingConfig

# TODO: @lefaudeux - import automatically in the base classes


@dataclass
class xFormerConfig:
    dim_embd: int
    attention_config: AttentionConfig
    feedforward_config: FeedforwardConfig
    position_encoding_config: PositionEncodingConfig


class xFormerBlock(nn.Module):
    """ a vanilla Transformer block """

    def __init__(self, config: xFormerConfig):
        super().__init__()

        # FIXME.. build the proper blocks on the fly
        self.ln1 = nn.LayerNorm(config.dim_embd)
        self.ln2 = nn.LayerNorm(config.dim_embd)

        self.attn = MultiHeadAttention.from_config(config.attention_config)
        self.mlp = MLP.from_config(config.feedforward_config)

    @classmethod
    def from_config(cls, config: xFormerConfig):
        # FIXME.. a bit useless
        return cls(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
