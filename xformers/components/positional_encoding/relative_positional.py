# credits https://github.com/lucidrains/local-attention
import torch
import torch.nn as nn

from xformers.components.positional_encoding import (  # register_positional_encoding,
    PositionEncoding,
    PositionEncodingConfig,
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


# FIXME: Broken unit test for now
# @register_positional_encoding("relative")
class RelativePositionalEncoding(PositionEncoding):
    def __init__(self, dim_model: int, seq_len: int, n_heads: int, *args, **kwargs):
        super().__init__()
        self.scale = dim_model ** -0.5
        self.weights = nn.Parameter(torch.zeros(seq_len, n_heads, dim_model))

    def forward(self, q):
        emb = (
            torch.einsum("bhnid,jhd->bhnij", q, self.weights.type(q.dtype)) * self.scale
        )
        return _shift(emb)
