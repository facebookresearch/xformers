from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import Attention, AttentionConfig, register_attention

from performer_pytorch import FastAttention


"""
Using third-party implementation for quick experiments. #TODO test if favor match the performance here
"""

@dataclass
class PerformerConfig(AttentionConfig):
    block_size: int
    num_heads: int
    dim_model: int
    rp_dim: int


@register_attention("performer", PerformerConfig)
class PerformerAttention(Attention):
    def __init__(
        self, 
        dropout: float, 
        num_heads: int,
        dim_model: int,
        rp_dim: int = 256,
        *args, **kwargs
    ):

        super().__init__()

        self.num_head = num_heads
        self.head_dim = dim_model // num_heads
        self.rp_dim = rp_dim
        self.attn_fn = FastAttention(dim_heads = self.head_dim, nb_features = self.rp_dim, causal = False, kernel_fn = nn.ReLU())


    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        att_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ): 
        batch_size = q.shape[0] // self.num_head
        q = q.view(batch_size, self.num_head, -1, self.head_dim)
        v = v.view(batch_size, self.num_head, -1, self.head_dim)
        k = k.view(batch_size, self.num_head, -1, self.head_dim)
        
        mask = ~key_padding_mask.eq(1)
        mask = mask.to(q)

        out = self.attn_fn(
            q / math.sqrt(math.sqrt(self.head_dim)),
            k / math.sqrt(math.sqrt(self.head_dim)),
            v, mask)

        return out.view(batch_size*self.num_head, -1, self.head_dim)