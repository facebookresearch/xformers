import math

import torch

from xformers.components.positional_encoding import (
    PositionEncoding,
    PositionEncodingConfig,
)

# Credits: A Joulin
# https://github.com/fairinternal/ajoulin/blob/master/transformer/model.py

# FIXME: placeholder


class SinePositionEncoding(PositionEncoding):
    def __init__(self, dim_embd: int, seq_len: int):
        super().__init__()

        pos = torch.arange(0.0, seq_len).unsqueeze(1).repeat(1, dim_embd)
        dim = torch.arange(0.0, dim_embd).unsqueeze(0).repeat(seq_len, 1)
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / seq_len))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

    @classmethod
    def from_config(self, config: PositionEncodingConfig) -> "SinePositionEncoding":
        # TODO: @lefaudeux
        pass
