from dataclasses import dataclass
from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from xformers.components.attention import Attention, AttentionConfig, register_attention

@dataclass
class LongShortConfig(AttentionConfig):
    block_size: int
    num_heads: int
    dim_model: int
    num_landmarks: int
    window_size: int
    global_tokens: int

@register_attention("longshort", LongShortConfig)
class LongShortAttention(Attention):
    def __init__(
        self, 
        dropout: float, 
        num_heads: int,
        dim_model: int,
        num_landmarks: int = 64,
        window_size: int = 256,
        global_tokens: int = 1,
        *args, **kwargs
    ):
        """
        longshort transformer
        https://arxiv.org/abs/2107.02192

        https://github.com/NVIDIA/transformer-ls

        #TODO only support encoding only settings
        """
        super().__init__()
        self.drop_attn = nn.Dropout(dropout)
        self.num_head = num_heads
        self.head_dim = dim_model // num_heads
        self.num_landmarks = num_landmarks
        self.dim = dim_model
        self.window_size = window_size        
        self.requires_orig_inputs = True

        self.global_tokens = global_tokens # default num of global tokens

        # Sec 3.4 to stablize the attention combination
        self.dual_ln_s = nn.LayerNorm(self.num_head * self.head_dim)
        self.dual_ln_l = nn.LayerNorm(self.num_head * self.head_dim)

        self.dconv_fc = nn.Linear(self.dim, self.num_head * self.num_landmarks)
        # input-dependent compression
    
    #### v2: try to change how the additional layernorm is applied, make it more competible with roberta checkpoint
    #### when compress the k/v, k/v are NOT normalized
    def forward(
        self, 
        x: torch.Tensor,
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
        x, pad_len = _pad_to_window_size(x, self.window_size)
        Q, _ = _pad_to_window_size(Q, self.window_size)
        K, _ = _pad_to_window_size(K, self.window_size)
        V, _ = _pad_to_window_size(V, self.window_size)
        if key_padding_mask.shape[1] % self.window_size != 0:
            pad_len = (self.window_size - key_padding_mask.shape[1] % self.window_size) % self.window_size
            # key padding mask: 1 means padding tokens
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1) 

        # normalization is to have the same scale as compressed key/values
        K_normalized = self.split_heads(self.dual_ln_l(K)) # (B, H, seq_len, head_dim)
        V_normalized = self.split_heads(self.dual_ln_l(V))

        K = self.split_heads(K)
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
            K_transpose = K_normalized.transpose(1,2)
            Q_transpose = Q.transpose(1,2)
            selected_k = K_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = K_transpose[extra_attention_mask_nonzeros]
            # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (Q_transpose, selected_k))
            selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
            attn_weights_over_g_tokens = selected_attn_weights.transpose(1,2)

            V_transpose = V_normalized.transpose(1,2)
            selected_v = V_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = V_transpose[extra_attention_mask_nonzeros]

        # linear attention mask
        padding_mask = key_padding_mask != 0 # True means masked position

        K_compress = V_compress = None
        if self.num_landmarks > 0:
            # self.dconv_fc(x): each lanmark of each head attend to each location
            head_scores = self.dconv_fc(x).masked_fill(padding_mask[:, :, None], -10000) # only try to compress the k/v of those local tokens
            head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32) #.to(X)
            # if not self.fp32:
            head_scores = head_scores.to(x)

            # bsz x num_head x num_lms x length
            head_scores = head_scores.view(batch_size, -1, self.num_head, self.num_landmarks).permute(0, 2, 3, 1)

            # TODO: should we use normalized K/V here? different from paper descriptions
            K_compress = head_scores.matmul(K) # bsz x num_head x num_lms x head_dim
            V_compress = head_scores.matmul(V)

        assert self.num_landmarks > 0

        if self.dual_ln_s is not None and K_compress is not None:
            K_compress = self.dual_ln_s(K_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim))
            K_compress = self.split_heads(K_compress)
            V_compress = self.dual_ln_s(V_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim))
            V_compress = self.split_heads(V_compress)

        # if self.num_landmarks > 0 or (cls_embed_q is not None):
            # bsz x num_head x length x num_lms
        attn_compress = Q.matmul(K_compress.transpose(-1, -2))

        win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K_normalized, padding_mask) # bsz x num_heads x seqlen x 2winsize
        # else:
        #     win_attn_weights = None

        if extra_attention_mask is not None:
            all_attn_ = torch.cat([attn_compress, win_attn_weights, attn_weights_over_g_tokens], dim=-1)
        else:
            all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)
        all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)

        hard_mask = key_padding_mask == 1
        all_attn = all_attn.masked_fill(hard_mask[:,None,:,None], 0)

        # if not self.fp32:
        all_attn = all_attn.to(x)
        all_attn = self.drop_attn(all_attn)

        C = 0
        if attn_compress is not None:
            C += all_attn[:,:,:,:K_compress.shape[2]].matmul(V_compress)

        if win_attn_weights is not None:
            win_attn_probs = all_attn[:,:,:,K_compress.shape[2]:K_compress.shape[2] + win_attn_weights.shape[-1]]
            seq_len = win_attn_probs.shape[2]
            # if self.window_size > 0:
            win_attn_probs = win_attn_probs.view(batch_size, self.num_head, seq_len // self.window_size, self.window_size,-1)
            V_tiles = self.get_tiles_v2(V_normalized, transpose=False)
            C += win_attn_probs.matmul(V_tiles).view(batch_size, self.num_head, seq_len, self.head_dim)
            # else:
                # C += win_attn_probs * V

        if extra_attention_mask is not None:
            global_attn_probs = all_attn[:,:,:,-attn_weights_over_g_tokens.shape[-1]:]
            # selected_v shape: (batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            C += global_attn_probs.matmul(selected_v.transpose(1,2))

            # global tokens attending to all other tokens. 
            # TODO: Two options are possible here: whether to use normalized key/value
            selected_q = Q_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
            selected_q[selection_padding_mask_nonzeros] = Q_transpose[extra_attention_mask_nonzeros]
            g2all_attn_weights = selected_q.transpose(1,2).matmul(K_normalized.transpose(-1, -2)) # (batch_size, self.num_head, max_num_extra_indices_per_batch, seq_len)
            g2all_attn_probs_float = F.softmax(g2all_attn_weights, dim=-1, dtype=torch.float32)
            g2all_attn_probs = self.drop_attn(g2all_attn_probs_float.type_as(g2all_attn_weights))
            g2all_attn = g2all_attn_probs.matmul(V_normalized) # (batch_size, self.num_head, max_num_extra_indices_per_batch, head_dim)

            # replace results in C
            nonzero_global_attn = g2all_attn[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            C[extra_attention_mask_nonzeros[0],:,extra_attention_mask_nonzeros[1]] = nonzero_global_attn

        C = C[:,:,:sequence_length].view(-1, sequence_length, self.head_dim)

        return C


    # #### v1 with added global tokens at the begining
    # def forward(
    #     self, 
    #     x: torch.Tensor,
    #     q: torch.Tensor, 
    #     k: torch.Tensor, 
    #     v: torch.Tensor, 
    #     att_mask: Optional[torch.Tensor] = None, 
    #     key_padding_mask: Optional[Tensor] = None,
    #     *args, **kwargs
    # ):
    #     """
    #     This version enables specify some global tokens, according values in key_padding_mask: {-1: global, 0: local, 1: masked tokens}
    #     """

    #     # global_tokens = self.global_tokens

    #     batch_size = q.shape[0] // self.num_head
    #     sequence_length = q.shape[1]

    #     # xformer format, att_mask: B x seq_len x seq_len #TODO: better solution?
    #     # if key_padding_mask is not None:
    #     #     mask = key_padding_mask == 0
    #     # if att_mask is not None:
    #     #     assert len(att_mask.shape) == 3
    #     #     att_mask = att_mask.reshape(batch_size, self.num_head, -1, sequence_length)[:,0,:,:]
    #     #     mask = att_mask.sum(1) > 0
    #     # else:
    #     #     mask = None

    #     # Always set the first token as global tokens
    #     key_padding_mask = key_padding_mask.to(q)
    #     key_padding_mask[:,0] = -1

    #     Q = q.view(batch_size, self.num_head, -1, self.head_dim).mul(1./math.sqrt(self.head_dim))
    #     K = k.view(batch_size, self.num_head, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.dim)
    #     V = v.view(batch_size, self.num_head, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.dim)

    #     # needs centain sequence length to make the block wise local attention work
    #     def _pad_to_window_size(x, window_size):
    #         seq_len = x.size(-2)
    #         pad_len = (window_size - seq_len % window_size) % window_size
    #         return F.pad(x, (0,0,0,pad_len), value=0), pad_len
    #     x, pad_len = _pad_to_window_size(x, self.window_size)
    #     Q, _ = _pad_to_window_size(Q, self.window_size)
    #     K, _ = _pad_to_window_size(K, self.window_size)
    #     V, _ = _pad_to_window_size(V, self.window_size)
    #     if key_padding_mask.shape[1] % self.window_size != 0:
    #         pad_len = (self.window_size - key_padding_mask.shape[1] % self.window_size) % self.window_size
    #         # key padding mask: 1 means padding tokens
    #         key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1) 

    #     # normalization is to have the same scale as compressed key/values
    #     K_normalized = self.split_heads(self.dual_ln_l(K)) # (B, H, seq_len, head_dim)
    #     V_normalized = self.split_heads(self.dual_ln_l(V))

    #     # 1. check size of x (bsz, seq-1, dim_model);  # TODO, 2. the padding mask value

    #     # global attention tokens
    #     extra_attention_mask = key_padding_mask < 0
    #     num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
    #     max_num_extra_indices_per_batch = num_extra_indices_per_batch.max()

    #     if max_num_extra_indices_per_batch <= 0:
    #         extra_attention_mask = None
    #     else:
    #         extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
    #         zero_to_max_range = torch.arange(0, max_num_extra_indices_per_batch, device=extra_attention_mask.device)
    #         # mask indicating which values are actually going to be padding
    #         num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
    #         selection_padding_mask = zero_to_max_range < num_extra_indices_per_batch.unsqueeze(dim=-1)
    #         # 2) location of the non-padding values in the selected global attention
    #         selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
    #         # 3) location of the padding values in the selected global attention
    #         selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)

    #     # keys of global tokens
    #     if extra_attention_mask is not None:
    #         K_transpose = K_normalized.transpose(1,2)
    #         Q_transpose = Q.transpose(1,2)
    #         selected_k = K_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
    #         selected_k[selection_padding_mask_nonzeros] = K_transpose[extra_attention_mask_nonzeros]
    #         # (bsz, seq_len, num_heads, max_num_extra_indices_per_batch)
    #         selected_attn_weights = torch.einsum('blhd,bshd->blhs', (Q_transpose, selected_k))
    #         selected_attn_weights[selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]] = -10000
    #         attn_weights_over_g_tokens = selected_attn_weights.transpose(1,2)

    #         V_transpose = V_normalized.transpose(1,2)
    #         selected_v = V_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
    #         selected_v[selection_padding_mask_nonzeros] = V_transpose[extra_attention_mask_nonzeros]

    #     # linear attention mask
    #     padding_mask = key_padding_mask != 0 # True means masked position

    #     K_compress = V_compress = None
    #     if self.num_landmarks > 0:
    #         # self.dconv_fc(x): each lanmark of each head attend to each location
    #         head_scores = self.dconv_fc(x).masked_fill(padding_mask[:, :, None], -10000) # only try to compress the k/v of those local tokens
    #         head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32) #.to(X)
    #         # if not self.fp32:
    #         head_scores = head_scores.to(x)

    #         # bsz x num_head x num_lms x length
    #         head_scores = head_scores.view(batch_size, -1, self.num_head, self.num_landmarks).permute(0, 2, 3, 1)

    #         # TODO: should we use normalized K/V here? different from paper descriptions
    #         K_compress = head_scores.matmul(K_normalized) # bsz x num_head x num_lms x head_dim
    #         V_compress = head_scores.matmul(V_normalized)


    #     assert self.num_landmarks > 0

    #     if self.dual_ln_s is not None and K_compress is not None:
    #         K_compress = self.dual_ln_s(K_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim))
    #         K_compress = self.split_heads(K_compress)
    #         V_compress = self.dual_ln_s(V_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim))
    #         V_compress = self.split_heads(V_compress)

    #     # if self.num_landmarks > 0 or (cls_embed_q is not None):
    #         # bsz x num_head x length x num_lms
    #     attn_compress = Q.matmul(K_compress.transpose(-1, -2))

    #     win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K_normalized, padding_mask) # bsz x num_heads x seqlen x 2winsize
    #     # else:
    #     #     win_attn_weights = None

    #     if extra_attention_mask is not None:
    #         all_attn_ = torch.cat([attn_compress, win_attn_weights, attn_weights_over_g_tokens], dim=-1)
    #     else:
    #         all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)
    #     all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)

    #     hard_mask = key_padding_mask == 1
    #     all_attn = all_attn.masked_fill(hard_mask[:,None,:,None], 0)

    #     # if not self.fp32:
    #     all_attn = all_attn.to(x)
    #     all_attn = self.drop_attn(all_attn)

    #     C = 0
    #     if attn_compress is not None:
    #         C += all_attn[:,:,:,:K_compress.shape[2]].matmul(V_compress)

    #     if win_attn_weights is not None:
    #         win_attn_probs = all_attn[:,:,:,K_compress.shape[2]:K_compress.shape[2] + win_attn_weights.shape[-1]]
    #         seq_len = win_attn_probs.shape[2]
    #         # if self.window_size > 0:
    #         win_attn_probs = win_attn_probs.view(batch_size, self.num_head, seq_len // self.window_size, self.window_size,-1)
    #         V_tiles = self.get_tiles_v2(V_normalized, transpose=False)
    #         C += win_attn_probs.matmul(V_tiles).view(batch_size, self.num_head, seq_len, self.head_dim)
    #         # else:
    #             # C += win_attn_probs * V

    #     if extra_attention_mask is not None:
    #         global_attn_probs = all_attn[:,:,:,-attn_weights_over_g_tokens.shape[-1]:]
    #         # selected_v shape: (batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
    #         C += global_attn_probs.matmul(selected_v.transpose(1,2))

    #         # global tokens attending to all other tokens. 
    #         # TODO: Two options are possible here: whether to use normalized key/value
    #         selected_q = Q_transpose.new_zeros(batch_size, max_num_extra_indices_per_batch, self.num_head, self.head_dim)
    #         selected_q[selection_padding_mask_nonzeros] = Q_transpose[extra_attention_mask_nonzeros]
    #         g2all_attn_weights = selected_q.transpose(1,2).matmul(K_normalized.transpose(-1, -2)) # (batch_size, self.num_head, max_num_extra_indices_per_batch, seq_len)
    #         g2all_attn_probs_float = F.softmax(g2all_attn_weights, dim=-1, dtype=torch.float32)
    #         g2all_attn_probs = self.drop_attn(g2all_attn_probs_float.type_as(g2all_attn_weights))
    #         g2all_attn = g2all_attn_probs.matmul(V_normalized) # (batch_size, self.num_head, max_num_extra_indices_per_batch, head_dim)

    #         # replace results in C
    #         nonzero_global_attn = g2all_attn[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]

    #         C[extra_attention_mask_nonzeros[0],:,extra_attention_mask_nonzeros[1]] = nonzero_global_attn
        
    #     # get rid of the padding positions
    #     C = C[:,:,:sequence_length].view(-1, sequence_length, self.head_dim)

    #     return C

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

    # #### v0 the original implementaion
    # def forward(
    #     self, 
    #     x: torch.Tensor,
    #     q: torch.Tensor, 
    #     k: torch.Tensor, 
    #     v: torch.Tensor, 
    #     att_mask: Optional[torch.Tensor] = None, 
    #     key_padding_mask: Optional[Tensor] = None,
    #     *args, **kwargs
    # ):
    #     global_tokens = self.global_tokens

    #     batch_size = q.shape[0] // self.num_head
    #     sequence_length = q.shape[1]

    #     # xformer format, att_mask: B x seq_len x seq_len #TODO: better solution?
    #     # if key_padding_mask is not None:
    #     #     mask = key_padding_mask == 0
    #     if att_mask is not None:
    #         assert len(att_mask.shape) == 3
    #         att_mask = att_mask.reshape(batch_size, self.num_head, -1, sequence_length)[:,0,:,:]
    #         mask = att_mask.sum(1) > 0
    #     else:
    #         mask = None

    #     Q = q.view(batch_size, self.num_head, -1, self.head_dim).mul(1./math.sqrt(self.head_dim))
    #     K = k.view(batch_size, self.num_head, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.dim)
    #     V = v.view(batch_size, self.num_head, -1, self.head_dim).transpose(1,2).reshape(batch_size, -1, self.dim)

    #     if global_tokens > 0:
    #         # cls_embed = x[:,:1].contiguous()
    #         cls_embed_q = Q[:,:,:global_tokens].contiguous()
    #         cls_embed_k = K[:,:global_tokens].contiguous()
    #         cls_embed_v = V[:,:global_tokens].contiguous()

    #         x = x[:,global_tokens:].contiguous() # B x (seq - 1) x dim_model
    #         Q = Q[:,:,global_tokens:].contiguous() # B x heads x (seq - 1) x head_dim
    #         K = K[:,global_tokens:].contiguous() # B x (seq - 1) x dim_model
    #         V = V[:,global_tokens:].contiguous()
    #         mask = mask[:,global_tokens:].contiguous() if mask is not None else None # B x (seq - 1)

    #     def _pad_to_window_size(x, window_size):
    #         seq_len = x.size(-2)
    #         pad_len = (window_size - seq_len % window_size) % window_size
    #         return F.pad(x, (0,0,0,pad_len), value=0), pad_len
    
    #     x, pad_len = _pad_to_window_size(x, self.window_size)
    #     Q, _ = _pad_to_window_size(Q, self.window_size)
    #     K, _ = _pad_to_window_size(K, self.window_size)
    #     V, _ = _pad_to_window_size(V, self.window_size)
    #     if mask.shape[1] % self.window_size != 0:
    #         pad_len = (self.window_size - mask.shape[1] % self.window_size) % self.window_size
    #         mask = torch.cat([mask, mask.new_zeros(mask.size(0), pad_len).to(mask)], dim=1)

    #     K = self.split_heads(self.dual_ln_l(K))
    #     V = self.split_heads(self.dual_ln_l(V))

    #     # 1. check size of x (bsz, seq-1, dim_model);  # TODO, 2. the padding mask value

    #     padding_mask = ~mask.bool()

    #     K_compress = V_compress = None
    #     if self.num_landmarks > 0:
            
    #         head_scores = self.dconv_fc(x).masked_fill(padding_mask[:, :, None], float('-inf'))
    #         head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32) #.to(X)
    #         # if not self.fp32:
    #         head_scores = head_scores.to(x)

    #         # bsz x num_head x num_lms x length
    #         head_scores = head_scores.view(batch_size, -1, self.num_head, self.num_landmarks).permute(0, 2, 3, 1)
    #         K_compress = head_scores.matmul(K) # bsz x num_head x num_lms x head_dim
    #         V_compress = head_scores.matmul(V)

    #     if global_tokens > 0:
    #         Q_cls = cls_embed_q # B x heads x 1 x head_dim
    #         K_cls = self.split_heads(cls_embed_k) # B x heads x 1 x head_dim
    #         V_cls = self.split_heads(cls_embed_v)
    #         if self.num_landmarks > 0:
    #             K_compress = torch.cat([K_cls, K_compress], dim=2) # B x heads x (1 + lms) x head_dim
    #             V_compress = torch.cat([V_cls, V_compress], dim=2)
    #         else:
    #             K_compress = K_cls
    #             V_compress = V_cls

    #     if self.dual_ln_s is not None and K_compress is not None:
    #         K_compress = self.dual_ln_s(K_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim))
    #         K_compress = self.split_heads(K_compress)
    #         V_compress = self.dual_ln_s(V_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim))
    #         V_compress = self.split_heads(V_compress)

    #     if self.num_landmarks > 0 or (cls_embed_q is not None):
    #         # bsz x num_head x length x num_lms
    #         attn_compress = Q.matmul(K_compress.transpose(-1, -2))
    #     else:
    #         attn_compress = None

    #     if self.window_size > 0 or self.num_landmarks == 0:
    #         # First, compute the compressed part, or the attentions on the landmarks
    #         # First use window attention to attend to the diagonals
    #         # V: bsize, self.seq_len, self.num_head, self.head_dim
    #         # win_attn_weights = self.sliding_chunks_matmul_qk(Q, K, padding_mask)
    #         win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask)
    #     else:
    #         win_attn_weights = None

    #     if attn_compress is None:
    #         all_attn_ = win_attn_weights
    #     elif win_attn_weights is None:
    #         all_attn_ = attn_compress
    #     else:
    #         all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)

    #     all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)
    #     all_attn = all_attn.masked_fill(padding_mask[:,None,:,None], 0)

    #     # if not self.fp32:
    #     all_attn = all_attn.to(x)
    #     all_attn = self.drop_attn(all_attn)

    #     C = 0
    #     if attn_compress is not None:
    #         C += all_attn[:,:,:,:K_compress.shape[2]].matmul(V_compress)

    #     if win_attn_weights is not None:
    #         win_attn_probs = all_attn[:,:,:,-win_attn_weights.shape[-1]:]
    #         seq_len = win_attn_probs.shape[2]
    #         if self.window_size > 0:
    #             win_attn_probs = win_attn_probs.view(batch_size, self.num_head, seq_len // self.window_size, self.window_size,-1)
    #             V_tiles = self.get_tiles_v2(V, transpose=False)
    #             C += win_attn_probs.matmul(V_tiles).view(batch_size, self.num_head, seq_len, self.head_dim)
    #         else:
    #             C += win_attn_probs * V

    #     if cls_embed_q is not None:
    #         # if self.fp32:
    #         #     Q_cls, K_cls, V_cls = Q_cls.float(), K_cls.float(), V_cls.float()
    #         # bsz x n_heads x 1 x (1+seqlen)
    #         cls_scores = torch.cat([Q_cls.matmul(K_cls.transpose(-1, -2)),
    #                                 Q_cls.matmul(C.transpose(-1, -2)).masked_fill(padding_mask[:,None,None,:], float('-inf'))],
    #                                dim=-1)
    #         cls_probs = torch.softmax(cls_scores, dim=-1, dtype=torch.float32)#.to(X)
    #         # if not self.fp32:
    #         cls_probs = cls_probs.to(x)
    #         out_cls_embed = cls_probs[:,:,:,:global_tokens].matmul(V_cls) + cls_probs[:,:,:,global_tokens:].matmul(C)

    #     if cls_embed_q is not None:
    #         C = torch.cat([out_cls_embed, C], dim=2)
        
    #     # get rid of the padding positions
    #     C = C[:,:,:sequence_length].view(-1, sequence_length, self.head_dim)

    #     return C
