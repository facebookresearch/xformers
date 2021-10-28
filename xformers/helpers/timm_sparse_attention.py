# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers.components.attention.core import scaled_dot_product_attention


class TimmSparseAttention(torch.nn.Module):
    """
    Almost drop-in replacement for timm attention
    but using the sparsity-aware scaled_dot_product_attention from xformers
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_mask=None,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.attn_mask = attn_mask

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv = qkv.flatten(1, 2)

        q, k, v = qkv.unbind()

        x = scaled_dot_product_attention(
            q, k, v, self.attn_mask, dropout=self.attn_drop
        )
        x = x.reshape(B, self.num_heads, N, C // self.num_heads)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
