from dataclasses import dataclass
from typing import Optional, Sequence
from functools import partial, reduce
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import Attention, AttentionConfig, register_attention

@dataclass
class BlockwiseConfig(AttentionConfig):
    block_size: int
    num_heads: int
    dim_model: int
    window_size: int


@register_attention("blockwise", BlockwiseConfig)
class BlockWiseAttention(Attention):
    def __init__(
        self, 
        dropout: float, 
        num_heads: int,
        dim_model: int,
        window_size: int = 256,
        *args, **kwargs
    ):
        super().__init__()
        self.drop_attn = nn.Dropout(dropout)
        self.num_head = num_heads
        self.head_dim = dim_model // num_heads
        self.dim = dim_model
        self.window_size = window_size        

    def get_tiles(self, x, transpose=False):
        # x: bsz x n_heads x seqlen x d_head
        bsz, n_heads, seqlen, d_h = x.shape
        out_shape = (bsz, n_heads, seqlen//self.window_size-1, 2 * self.window_size, d_h)
        in_strides = x.stride()
        out_strides = (in_strides[0], in_strides[1], in_strides[2]*self.window_size, in_strides[2], 1)

        x_main = x.as_strided(size=out_shape, stride=out_strides)
        x_last = x[:, :, None, -2*self.window_size:, :]
        x = torch.cat([x_main, x_last], dim=2)
        if transpose:
            return x.transpose(-1, -2)
        else:
            #  bsz x n_heads x seqlen//wlen x 2*wlen x d_h
            return x

    def get_tiled_mask(self, mask):
        bsz, seqlen = mask.shape
        out_shape = (bsz, seqlen//self.window_size-1, 2*self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])
        mask_main = mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, :]
        mask_last = mask[:, None, None, -2*self.window_size:]

        return torch.cat([mask_main, mask_last], dim=2)[:, :, :, None, :]

    def sliding_chunks_matmul_qk(self, Q, K, padding_mask):
        # Q, K: bsz x num_heads x seqlen x d_head
        # padding_mask: bsz x seqlen
        bsz, num_heads, seqlen, d_h = Q.shape
        mask_tiles = self.get_tiled_mask(padding_mask)
        K_tiles = self.get_tiles(K, transpose=True)
        Q_tiles = Q.view(bsz, num_heads, seqlen//self.window_size, self.window_size, d_h)
        # bsz x num_heads x seqlen//winsize x winsize x 2winsize
        qk_scores = Q_tiles.matmul(K_tiles)
        qk_scores.masked_fill_(mask_tiles, -10000)
        return qk_scores.view(bsz, num_heads, seqlen, 2*self.window_size)

    def get_tiles_v2(self, x, transpose=False):
        if self.window_size <= 0:
            return x

        bsz, n_heads, seqlen, d_h = x.shape
        n_groups = seqlen // self.window_size
        ext_len = max(self.window_size//2, 1)
        x = F.pad(x, (0, 0, ext_len, ext_len), value=0)
        strides = x.stride()
        if transpose:
            out_shape = (bsz, n_heads, n_groups, d_h, 2 * ext_len + self.window_size)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[3], strides[2])
        else:
            out_shape = (bsz, n_heads, n_groups, 2 * ext_len + self.window_size, d_h)
            out_stride = (strides[0], strides[1], self.window_size * strides[2], strides[2], strides[3])
        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def get_tiled_mask_v2(self, mask):
        # only mask along the key dimension
        bsz, seqlen = mask.shape
        ext_len = max(self.window_size//2, 1)
        mask = F.pad(mask, (ext_len, ext_len), value=True) # (bsz, seq_len + 2*ext_len)
        out_shape = (bsz, seqlen//self.window_size, 2*ext_len + self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1]*self.window_size, in_stride[1])

        return mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, None, :]

    def sliding_chunks_matmul_qk_v2(self, Q, K, padding_mask):
        bsz, num_heads, seqlen, d_h = Q.shape
        if self.window_size > 0:
            # Q, K: bsz x num_heads x seqlen x d_head
            # padding_mask: bsz x seqlen

            mask_tiles = self.get_tiled_mask_v2(padding_mask)
            K_tiles = self.get_tiles_v2(K, transpose=True)
            Q_tiles = Q.view(bsz, num_heads, seqlen//self.window_size, self.window_size, d_h)
            # bsz x num_heads x seqlen//winsize x winsize x 2winsize
            qk_scores = Q_tiles.matmul(K_tiles)
            qk_scores = qk_scores.masked_fill(mask_tiles, -10000)
            return qk_scores.view(bsz, num_heads, seqlen, -1)
        else:
            qk_scores = torch.sum(Q*K, dim=-1, keepdim=True)
            return qk_scores

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        att_mask: Optional[torch.Tensor] = None, 
        key_padding_mask: Optional[Tensor] = None,
        *args, **kwargs
    ):
        """
        This version enables specify some global tokens, according values in key_padding_mask: {-1: global, 0: local, 1: masked tokens}
        """

        # global_tokens = self.global_tokens

        batch_size = q.shape[0] // self.num_head
        sequence_length = q.shape[1]

        # xformer format, att_mask: B x seq_len x seq_len #TODO: better solution?
        # if key_padding_mask is not None:
        #     mask = key_padding_mask == 0
        # if att_mask is not None:
        #     assert len(att_mask.shape) == 3
        #     att_mask = att_mask.reshape(batch_size, self.num_head, -1, sequence_length)[:,0,:,:]
        #     mask = att_mask.sum(1) > 0
        # else:
        #     mask = None

        # Always set the first token as global tokens
        key_padding_mask = key_padding_mask.to(q)
        key_padding_mask[:,0] = -1

        Q = q.view(batch_size, self.num_head, -1, self.head_dim).mul(1./math.sqrt(self.head_dim))
        K = k.view(batch_size, self.num_head, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.dim)
        V = v.view(batch_size, self.num_head, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.dim)

        # needs centain sequence length to make the block wise local attention work
        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0,0,0,pad_len), value=0), pad_len
        Q, _ = _pad_to_window_size(Q, self.window_size)
        K, _ = _pad_to_window_size(K, self.window_size)
        V, _ = _pad_to_window_size(V, self.window_size)
        if key_padding_mask.shape[1] % self.window_size != 0:
            pad_len = (self.window_size - key_padding_mask.shape[1] % self.window_size) % self.window_size
            # key padding mask: 1 means padding tokens
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1) 

        K = self.split_heads(K) # (B, H, seq_len, head_dim)
        V = self.split_heads(V)

        # 1. check size of x (bsz, seq-1, dim_model);  # TODO, 2. the padding mask value

        # global attention tokens
        extra_attention_mask = key_padding_mask < 0
        num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
        max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()

        if max_num_extra_indices_per_batch <= 0:
            extra_attention_mask = None
        else:
            extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
            zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch, device=extra_attention_mask.device)
            # mask indicating which values are actually going to be padding
            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
            # 2) location of the non-padding values in the selected global attention
            selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
            # 3) location of the padding values in the selected global attention
            selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)

        # keys of global tokens
        if extra_attention_mask is not None:
            K_transpose = K.transpose(1,2)
            Q_transpose = Q.transpose(1,2)
            selected_k = K_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = K_transpose[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (Q_transpose, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            attn_weights_over_g_tokens = selected_attn_weights.transpose(1,2)

            V_transpose = V.transpose(1,2)
            selected_v = V_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = V_transpose[extra_attention_mask_nonzeros]

        # linear attention mask
        padding_mask = key_padding_mask != 0 # True means masked position

        win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask) # bsz x num_heads x seqlen x 2winsize
        # else:
        #     win_attn_weights = None

        if extra_attention_mask is not None:
            all_attn_ = torch.cat([win_attn_weights, attn_weights_over_g_tokens], dim=-1)
        else:
            all_attn_ = win_attn_weights
        all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)

        hard_mask = key_padding_mask == 1
        all_attn = all_attn.masked_fill(hard_mask[:,None,:,None], 0)

        # if not self.fp32:
        all_attn = all_attn.to(q)
        all_attn = self.drop_attn(all_attn)

        C = 0

        if win_attn_weights is not None:
            win_attn_probs = all_attn[:,:,:,:win_attn_weights.shape[-1]]
            seq_len = win_attn_probs.shape[2]
            # if self.window_size > 0:
            win_attn_probs = win_attn_probs.view(batch_size, self.num_head, seq_len // self.window_size, self.window_size,-1)
            V_tiles = self.get_tiles_v2(V, transpose=False)
            C += win_attn_probs.matmul(V_tiles).view(batch_size, self.num_head, seq_len, self.head_dim)
            # else:
                # C += win_attn_probs * V

        if extra_attention_mask is not None:
            global_attn_probs = all_attn[:,:,:,-attn_weights_over_g_tokens.shape[-1]:]
            # selected_v shape: (batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            C += global_attn_probs.matmul(selected_v.transpose(1,2))

            selected_q = Q_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            selected_q[selection_padding_mask_nonzeros] = Q_transpose[extra_attention_mask_nonzeros]
            g2all_attn_weights = selected_q.transpose(1,2).matmul(K.transpose(-1, -2)) # (batch_size, self.num_head, max_num_extra_indices_per_batch, seq_len)
            
            g2all_attn_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if hard_mask is not None:
                g2all_attn_weights = g2all_attn_weights.masked_fill(
                    hard_mask.unsqueeze(1).unsqueeze(2),
                    -10000.0,
                )

            g2all_attn_probs_float = F.softmax(g2all_attn_weights, dim=-1, dtype=torch.float32)
            g2all_attn_probs = self.drop_attn(g2all_attn_probs_float.type_as(g2all_attn_weights))
            g2all_attn = g2all_attn_probs.matmul(V) # (batch_size, self.num_head, max_num_extra_indices_per_batch, head_dim)

            # replace results in C
            nonzero_global_attn = g2all_attn[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]

            C[extra_attention_mask_nonzeros[0],:,extra_attention_mask_nonzeros[1]] = nonzero_global_attn
        
        # get rid of the padding positions
        C = C[:,:,:sequence_length].view(-1, sequence_length, self.head_dim)

        return C