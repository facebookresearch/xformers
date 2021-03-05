# credits https://github.com/lucidrains/local-attention
import torch
import torch.nn as nn

from xformers.components.positional_encoding import (
    PositionEncoding,
    PositionEncodingConfig,
    register_positional_encoding,
)
from xformers.utils import to


def _shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    k = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % k, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, k)
    return shifted[..., :i, i - 1 :]


class RelativePositionalEncodingConfig(PositionEncodingConfig):
    n_heads: int


@register_positional_encoding("relative")
class RelativePositionalEncoding(PositionEncoding):
    def __init__(self, dim_model: int, seq_length: int, n_heads: int):
        super().__init__()
        self.scale = dim_model ** -0.5
        self.weights = nn.Parameter(torch.zeros(seq_length, n_heads, dim_model))

    def forward(self, q):
        emb = (
            torch.einsum("bhnid,jhd->bhnij", q, self.weights.type(q.dtype)) * self.scale
        )
        return _shift(emb)

    @classmethod
    def from_config(
        cls, config: RelativePositionalEncodingConfig
    ) -> "RelativePositionalEncoding":
        return cls(config.dim_model, config.seq_len, config.n_heads)
