import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import Attention, AttentionConfig, register_attention


@dataclass
class LongShortConfig(AttentionConfig):
    block_size: int
    num_heads: int
    seq_len: int
    dim_model: int
    num_landmarks: int
    window_size: int


@register_attention("longshort", LongShortConfig)
class LongShortAttention(Attention):
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_landmarks: int,
        dim_model: int,
        seq_len: int,
        window_size: int,
        *args,
        **kwargs,
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
        self.seq_len = seq_len
        self.dim = dim_model
        self.window_size = window_size
        self.requires_orig_inputs = True

        self.cls_from_seq = True

        # Sec 3.4 to stablize the attention combination
        self.dual_ln_s = nn.LayerNorm(self.num_head * self.head_dim)
        self.dual_ln_l = nn.LayerNorm(self.num_head * self.head_dim)

        self.dconv_fc = nn.Linear(self.dim, self.num_head * self.num_landmarks)
        # input-dependent compression

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = q.shape[0] // self.num_head
        sequence_length = q.shape[1]

        # xformer format, att_mask: B x seq_len x seq_len #TODO: better solution?
        if key_padding_mask is not None:
            mask: Optional[torch.Tensor] = key_padding_mask == 0
        elif att_mask is not None:
            assert len(att_mask.shape) == 3
            att_mask = att_mask.reshape(batch_size, self.num_head, -1, sequence_length)[
                :, 0, :, :
            ]
            mask = att_mask.sum(1) > 0
        else:
            mask = None

        Q = q.view(batch_size, self.num_head, -1, self.head_dim).mul(
            1.0 / math.sqrt(self.head_dim)
        )
        K = (
            k.view(batch_size, self.num_head, -1, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size, -1, self.dim)
        )
        V = (
            v.view(batch_size, self.num_head, -1, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size, -1, self.dim)
        )

        # FIXME: the different cases are not properly handled
        assert mask is not None

        if self.cls_from_seq:
            # cls_embed = x[:,:1].contiguous()
            cls_embed_q = Q[:, :, :1].contiguous()
            cls_embed_k = K[:, :1].contiguous()
            cls_embed_v = V[:, :1].contiguous()

            x = x[:, 1:].contiguous()  # B x (seq - 1) x dim_model
            Q = Q[:, :, 1:].contiguous()  # B x heads x (seq - 1) x head_dim
            K = K[:, 1:].contiguous()  # B x (seq - 1) x dim_model
            V = V[:, 1:].contiguous()

            mask = (
                mask[:, 1:].contiguous() if mask is not None else None
            )  # B x (seq - 1)

        def _pad_to_window_size(x, window_size):
            seq_len = x.size(-2)
            pad_len = (window_size - seq_len % window_size) % window_size
            return F.pad(x, (0, 0, 0, pad_len), value=0), pad_len

        x, pad_len = _pad_to_window_size(x, self.window_size)
        Q, _ = _pad_to_window_size(Q, self.window_size)
        K, _ = _pad_to_window_size(K, self.window_size)
        V, _ = _pad_to_window_size(V, self.window_size)
        if mask.shape[1] % self.window_size != 0:
            pad_len = (
                self.window_size - mask.shape[1] % self.window_size
            ) % self.window_size
            mask = torch.cat(
                [mask, mask.new_zeros(mask.size(0), pad_len).to(mask)], dim=1
            )

        K = self.split_heads(self.dual_ln_l(K))
        V = self.split_heads(self.dual_ln_l(V))

        # 1. check size of x (bsz, seq-1, dim_model);  # TODO, 2. the padding mask value

        padding_mask = ~mask.bool()

        K_compress = V_compress = None
        if self.num_landmarks > 0:
            head_scores = self.dconv_fc(x).masked_fill(
                padding_mask[:, :, None], float("-inf")
            )
            head_scores = F.softmax(head_scores, dim=1, dtype=torch.float32)  # noqa
            # if not self.fp32:
            head_scores = head_scores.to(x)

            # bsz x num_head x num_lms x length
            head_scores = head_scores.view(
                batch_size, -1, self.num_head, self.num_landmarks
            ).permute(0, 2, 3, 1)
            K_compress = head_scores.matmul(K)  # bsz x num_head x num_lms x head_dim
            V_compress = head_scores.matmul(V)

        assert K_compress is not None
        assert V_compress is not None

        if cls_embed_q is not None:

            Q_cls = cls_embed_q  # B x heads x 1 x head_dim
            K_cls = self.split_heads(cls_embed_k)  # B x heads x 1 x head_dim
            V_cls = self.split_heads(cls_embed_v)

            if self.num_landmarks > 0:
                K_compress = torch.cat(
                    [K_cls, K_compress], dim=2
                )  # B x heads x (1 + lms) x head_dim
                V_compress = torch.cat([V_cls, V_compress], dim=2)
            else:
                K_compress = K_cls
                V_compress = V_cls

        if self.dual_ln_s is not None and K_compress is not None:
            K_compress = self.dual_ln_s(
                K_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            )
            K_compress = self.split_heads(K_compress)
            V_compress = self.dual_ln_s(
                V_compress.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            )
            V_compress = self.split_heads(V_compress)

        if self.num_landmarks > 0 or (cls_embed_q is not None):
            # bsz x num_head x length x num_lms
            attn_compress = Q.matmul(K_compress.transpose(-1, -2))
        else:
            attn_compress = None

        if self.window_size > 0 or self.num_landmarks == 0:
            # First, compute the compressed part, or the attentions on the landmarks
            # First use window attention to attend to the diagonals
            # V: bsize, self.seq_len, self.num_head, self.head_dim
            # win_attn_weights = self.sliding_chunks_matmul_qk(Q, K, padding_mask)
            win_attn_weights = self.sliding_chunks_matmul_qk_v2(Q, K, padding_mask)
        else:
            win_attn_weights = None

        if attn_compress is None:
            all_attn_ = win_attn_weights
        elif win_attn_weights is None:
            all_attn_ = attn_compress
        else:
            all_attn_ = torch.cat([attn_compress, win_attn_weights], dim=-1)

        all_attn = all_attn_.float().softmax(dim=-1).to(win_attn_weights)
        all_attn = all_attn.masked_fill(padding_mask[:, None, :, None], 0)

        # if not self.fp32:
        all_attn = all_attn.to(x)
        all_attn = self.drop_attn(all_attn)

        # FIXME
        C = torch.zeros([])
        if attn_compress is not None:
            C += all_attn[:, :, :, : K_compress.shape[2]].matmul(V_compress)

        if win_attn_weights is not None:
            win_attn_probs = all_attn[:, :, :, -win_attn_weights.shape[-1] :]
            if self.window_size > 0:
                win_attn_probs = win_attn_probs.view(
                    batch_size,
                    self.num_head,
                    sequence_length // self.window_size,
                    self.window_size,
                    -1,
                )
                V_tiles = self.get_tiles_v2(V, transpose=False)
                C += win_attn_probs.matmul(V_tiles).view(
                    batch_size, self.num_head, sequence_length, self.head_dim
                )
            else:
                C += win_attn_probs * V

        if cls_embed_q is not None:
            # if self.fp32:
            #     Q_cls, K_cls, V_cls = Q_cls.float(), K_cls.float(), V_cls.float()
            # bsz x n_heads x 1 x (1+seqlen)
            cls_scores = torch.cat(
                [
                    Q_cls.matmul(K_cls.transpose(-1, -2)),
                    Q_cls.matmul(C.transpose(-1, -2)).masked_fill(
                        padding_mask[:, None, None, :], float("-inf")
                    ),
                ],
                dim=-1,
            )
            cls_probs = torch.softmax(cls_scores, dim=-1, dtype=torch.float32)  # .to(X)
            # if not self.fp32:
            cls_probs = cls_probs.to(x)
            out_cls_embed = V_cls * cls_probs[:, :, :, :1] + cls_probs[
                :, :, :, 1:
            ].matmul(C)

        if cls_embed_q is not None:
            C = torch.cat([out_cls_embed, C], dim=2)

        # if self.fp32:
        #     # Finally convert it back, same as Nystromformer
        #     C = C.to(x)

        # get rid of the padding positions
        C = C[:, :, :-pad_len].view(-1, sequence_length, self.head_dim)

        return C

    def get_tiles(self, x, transpose=False):
        # x: bsz x n_heads x seqlen x d_head
        bsz, n_heads, seqlen, d_h = x.shape
        out_shape = (
            bsz,
            n_heads,
            seqlen // self.window_size - 1,
            2 * self.window_size,
            d_h,
        )
        in_strides = x.stride()
        out_strides = (
            in_strides[0],
            in_strides[1],
            in_strides[2] * self.window_size,
            in_strides[2],
            1,
        )

        x_main = x.as_strided(size=out_shape, stride=out_strides)
        x_last = x[:, :, None, -2 * self.window_size :, :]
        x = torch.cat([x_main, x_last], dim=2)
        if transpose:
            return x.transpose(-1, -2)
        else:
            #  bsz x n_heads x seqlen//wlen x 2*wlen x d_h
            return x

    def get_tiled_mask(self, mask):
        bsz, seqlen = mask.shape
        out_shape = (bsz, seqlen // self.window_size - 1, 2 * self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1] * self.window_size, in_stride[1])
        mask_main = mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, :]
        mask_last = mask[:, None, None, -2 * self.window_size :]

        return torch.cat([mask_main, mask_last], dim=2)[:, :, :, None, :]

    def sliding_chunks_matmul_qk(self, Q, K, padding_mask):
        # Q, K: bsz x num_heads x seqlen x d_head
        # padding_mask: bsz x seqlen
        bsz, num_heads, seqlen, d_h = Q.shape
        mask_tiles = self.get_tiled_mask(padding_mask)
        K_tiles = self.get_tiles(K, transpose=True)
        Q_tiles = Q.view(
            bsz, num_heads, seqlen // self.window_size, self.window_size, d_h
        )
        # bsz x num_heads x seqlen//winsize x winsize x 2winsize
        qk_scores = Q_tiles.matmul(K_tiles)
        qk_scores.masked_fill_(mask_tiles, float("-inf"))
        return qk_scores.view(bsz, num_heads, seqlen, 2 * self.window_size)

    def get_tiles_v2(self, x, transpose=False):
        if self.window_size <= 0:
            return x

        bsz, n_heads, seqlen, d_h = x.shape
        n_groups = seqlen // self.window_size
        ext_len = max(self.window_size // 2, 1)
        x = F.pad(x, (0, 0, ext_len, ext_len), value=0)
        strides = x.stride()
        if transpose:
            out_shape = (bsz, n_heads, n_groups, d_h, 2 * ext_len + self.window_size)
            out_stride = (
                strides[0],
                strides[1],
                self.window_size * strides[2],
                strides[3],
                strides[2],
            )
        else:
            out_shape = (bsz, n_heads, n_groups, 2 * ext_len + self.window_size, d_h)
            out_stride = (
                strides[0],
                strides[1],
                self.window_size * strides[2],
                strides[2],
                strides[3],
            )
        return torch.as_strided(x, size=out_shape, stride=out_stride)

    def get_tiled_mask_v2(self, mask):
        # only mask along the key dimension
        bsz, seqlen = mask.shape
        ext_len = max(self.window_size // 2, 1)
        mask = F.pad(mask, (ext_len, ext_len), value=True)
        out_shape = (bsz, seqlen // self.window_size, 2 * ext_len + self.window_size)
        in_stride = mask.stride()
        out_stride = (in_stride[0], in_stride[1] * self.window_size, in_stride[1])
        return mask.as_strided(size=out_shape, stride=out_stride)[:, None, :, None, :]

    def sliding_chunks_matmul_qk_v2(self, Q, K, padding_mask):
        bsz, num_heads, seqlen, d_h = Q.shape
        if self.window_size > 0:
            # Q, K: bsz x num_heads x seqlen x d_head
            # padding_mask: bsz x seqlen

            mask_tiles = self.get_tiled_mask_v2(padding_mask)
            K_tiles = self.get_tiles_v2(K, transpose=True)
            Q_tiles = Q.view(
                bsz, num_heads, seqlen // self.window_size, self.window_size, d_h
            )
            # bsz x num_heads x seqlen//winsize x winsize x 2winsize
            qk_scores = Q_tiles.matmul(K_tiles)
            qk_scores = qk_scores.masked_fill(mask_tiles, float("-inf"))
            return qk_scores.view(bsz, num_heads, seqlen, -1)
        else:
            qk_scores = torch.sum(Q * K, dim=-1, keepdim=True)
            return qk_scores

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X
