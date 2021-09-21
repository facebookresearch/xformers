from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import Attention, AttentionConfig, register_attention

"""
Using official implementation for quick experiments.
"""

@dataclass
class NystromConfig(AttentionConfig):
    block_size: int
    num_heads: int
    dim_model: int
    num_landmarks: int
    conv_kernel_size: int


@register_attention("nystrom_orig", NystromConfig)
class NystromOrigAttention(Attention):
    def __init__(
        self, 
        num_heads: int,
        dim_model: int,
        num_landmarks: int = 64,
        conv_kernel_size: int = 35,
        *args, **kwargs
    ):
        super().__init__()

        self.num_head = num_heads
        self.head_dim = dim_model // num_heads
        self.num_landmarks = num_landmarks
        self.init_option = "original"
        self.conv_kernel_size = conv_kernel_size

        self.conv = nn.Conv2d(
            in_channels = self.num_head, out_channels = self.num_head,
            kernel_size = (self.conv_kernel_size, 1), padding = (self.conv_kernel_size // 2, 0),
            bias = False,
            groups = self.num_head)

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
        orig_seq_len = q.shape[1]
        q = q.view(batch_size, self.num_head, -1, self.head_dim)
        v = v.view(batch_size, self.num_head, -1, self.head_dim)
        k = k.view(batch_size, self.num_head, -1, self.head_dim)
        
        # pad to factors of self.num_landmarks
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        q, _ = _pad_to_window_size(q, self.num_landmarks)
        k, _ = _pad_to_window_size(k, self.num_landmarks)
        v, _ = _pad_to_window_size(v, self.num_landmarks)

        if key_padding_mask.shape[1] % self.num_landmarks != 0:
            pad_len = (self.num_landmarks - key_padding_mask.shape[1] % self.num_landmarks) % self.num_landmarks
            # key padding mask: 1 means padding tokens
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1) 

        mask = ~key_padding_mask.eq(1)
        mask = mask.to(q)

        q = q * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))
        k = k * mask[:, None, :, None] / math.sqrt(math.sqrt(self.head_dim))

        seq_len = q.shape[1]

        if self.num_landmarks >= seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-1, -2)) - 1e4 * (1 - mask[:, None, None, :]), dim = -1)
            x = torch.matmul(attn, v)
        else:
            Q_landmarks = q.reshape(-1, self.num_head, self.num_landmarks, seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = k.reshape(-1, self.num_head, self.num_landmarks, seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, k.transpose(-1, -2)) - 1e4 * (1 - mask[:, None, None, :]), dim = -1)
            x = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        x += self.conv(v * mask[:, None, :, None])

        return x[:,:,:orig_seq_len].view(batch_size*self.num_head, orig_seq_len, self.head_dim)


    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        if self.init_option == "original":
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V
