from dataclasses import dataclass
from typing import Optional

from torch import Tensor
import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import scaled_dot_product_attention
import torch.nn.functional as F


@dataclass
class LinformerSelfAttentionConfig(AttentionConfig):
    max_seq_len: int  # dimension of the input sequence
    num_heads: int
    compress: int

@register_attention("linformer", LinformerSelfAttentionConfig)
class LinformerAttention(Attention):
    def __init__(
        self, 
        dropout: float, 
        num_heads: int,
        max_seq_len: int = 4096, 
        compress: int = 4, 
        *args, **kwargs
    ):
        """
        Linformer attention mechanism, from
        "
        Linformer: Self-Attention with Linear Complexity. ArXiv, 2048(2019)
        Wang, S., Li, B. Z., Khabsa, M., Fang, H., & Ma, H. (2020).
        "

        The paper's notation are kept wherever possible
        """
        super().__init__()

        k = max_seq_len // compress

        self.num_heads = num_heads
        self.E = nn.Linear(max_seq_len, k, bias=False)
        self.F = nn.Linear(max_seq_len, k, bias=False)
        self.attn_drop = nn.Dropout(dropout, inplace=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ):
        bsz = q.shape[0] // self.num_heads
        mask = ~key_padding_mask.eq(1)
        mask = mask.to(q)

        tgt_len = k.shape[1]
        k_projected = F.linear(k.transpose(-2, -1), self.E.weight[:, 0:tgt_len]).transpose(-2, -1)
        v_projected = F.linear(v.transpose(-2, -1), self.F.weight[:, 0:tgt_len]).transpose(-2, -1)

        y = scaled_dot_product_attention(
            q=q, k=k_projected, v=v_projected, att_mask=None, dropout=self.attn_drop
        )

        y = y.view(bsz, self.num_heads, y.shape[1], y.shape[2])
        y = (y * mask[:,None,:,None]).view(bsz*self.num_heads, tgt_len, -1)
        return y
