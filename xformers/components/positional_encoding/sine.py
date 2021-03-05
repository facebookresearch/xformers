import math

import torch

from xformers.components.positional_encoding import (
    PositionEncoding,
    PositionEncodingConfig,
    register_positional_encoding,
)

# Credits: A Joulin
# https://github.com/fairinternal/ajoulin/blob/master/transformer/model.py

# FIXME: placeholder


@register_positional_encoding("sine")
class SinePositionEncoding(PositionEncoding):
    def __init__(self, dim_model: int, seq_len: int):
        super().__init__()

        pos = torch.arange(0.0, seq_len).unsqueeze(1).repeat(1, dim_model)
        dim = torch.arange(0.0, dim_model).unsqueeze(0).repeat(seq_len, 1)
        div = torch.exp(-math.log(10000) * (2 * (dim // 2) / seq_len))
        pos *= div
        pos[:, 0::2] = torch.sin(pos[:, 0::2])
        pos[:, 1::2] = torch.cos(pos[:, 1::2])
        self.register_buffer("pe", pos.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]

    @classmethod
    def from_config(cls, config: PositionEncodingConfig) -> "SinePositionEncoding":
        return cls(config.dim_model, config.seq_len)
