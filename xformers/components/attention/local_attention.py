# Credits : https://github.com/lucidrains/local-attention
# see https://arxiv.org/pdf/2003.05997.pdf
# and https://arxiv.org/pdf/2004.05150.pdf
# FIXME: proper credits

import math
from functools import reduce
from operator import mul
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from xformers.components.positional_encoding.relative_positional import (
    RelativePositionalEncoding,
)

from . import Attention, AttentionConfig, register_attention

TOKEN_SELF_ATTN_VALUE = -5e4  # carefully set for half precision to work


def default(value, d):
    return d if value is None else value


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def expand_dim(t, dim, k, unsqueeze=True):
    if unsqueeze:
        t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def pad_to_multiple(tensor, multiple, dim=-1, value=0):
    seqlen = tensor.shape[dim]
    m = seqlen / multiple
    if m.is_integer():
        return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value=value)


def look_around(x, backward=1, forward=0, pad_value=-1, dim=2):
    t = x.shape[1]
    dims = (len(x.shape) - dim) * (0, 0)
    padded_x = F.pad(x, (*dims, backward, forward), value=pad_value)
    tensors = [
        padded_x[:, ind : (ind + t), ...] for ind in range(forward + backward + 1)
    ]
    return torch.cat(tensors, dim=dim)


class LocalAttentionConfig(AttentionConfig):
    window_size: int
    autopad: bool
    shared_qk: bool
    exact_window_size: bool
    look_backward: int
    look_forward: int


@register_attention("local_attention")
class LocalAttention(Attention):
    def __init__(
        self,
        window_size,
        causal=False,
        look_backward=1,
        look_forward=None,
        attention_dropout=0.0,
        shared_qk=False,
        rel_pos_emb_config=None,
        autopad=False,
        exact_window_size=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        look_forward = default(look_forward, 0 if causal else 1)
        assert not (causal and look_forward > 0), "you cannot look forward if causal"

        self.window_size = window_size
        self.causal = causal
        self.look_backward = look_backward
        self.look_forward = look_forward
        self.exact_window_size = exact_window_size
        self.autopad = autopad

        self.dropout = nn.Dropout(attention_dropout)

        self.shared_qk = shared_qk

        self.rel_pos = None
        if rel_pos_emb_config is not None:
            dim_head, heads = rel_pos_emb_config
            rel_pos_length = window_size * (1 + look_forward + look_backward)
            self.heads = heads
            self.rel_pos = RelativePositionalEncoding(dim_head, rel_pos_length, heads)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
    ):

        # FIXME: This is wrong, just for unit testing
        if not k:
            k = q
        if not v:
            v = q

        shape = q.shape

        q, k, v = map(lambda t: t.reshape(-1, *t.shape[-2:]), (q, k, v))

        if self.autopad:
            orig_t = q.shape[1]
            q, k, v = map(
                lambda t: pad_to_multiple(t, self.window_size, dim=-2), (q, k, v)
            )

        window_size, causal, look_backward, look_forward, shared_qk = (
            self.window_size,
            self.causal,
            self.look_backward,
            self.look_forward,
            self.shared_qk,
        )

        b, s, e = q.shape  # batch x sequence x embedding
        device, dtype = q.device, q.dtype
        assert (
            s % window_size
        ) == 0, f"sequence length {s} must be divisible by window size {window_size} for local attention"

        windows = s // window_size

        if shared_qk:
            k = F.normalize(k, 2, dim=-1).type_as(q)

        ticker = torch.arange(s, device=device, dtype=dtype)[None, :]
        b_t = ticker.reshape(1, windows, window_size)

        bq, bk, bv = map(lambda t: t.reshape(b, windows, window_size, -1), (q, k, v))  # type: ignore

        look_around_kwargs = {"backward": look_backward, "forward": look_forward}
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)

        bq_t = b_t
        bq_k = look_around(b_t, **look_around_kwargs)

        dots = torch.einsum("bhie,bhje->bhij", bq, bk) * (e ** -0.5)

        if self.rel_pos is not None:
            rel_attn = self.rel_pos(bq.view(-1, self.heads, *bq.shape[1:])).reshape_as(
                dots
            )
            dots = dots + rel_attn

        mask_value = max_neg_value(dots)

        if shared_qk:
            mask = bq_t[:, :, :, None] == bq_k[:, :, None, :]
            dots.masked_fill_(mask, TOKEN_SELF_ATTN_VALUE)
            del mask

        if causal:
            mask = bq_t[:, :, :, None] < bq_k[:, :, None, :]

            if self.exact_window_size:
                max_causal_window_size = self.window_size * self.look_backward
                mask = mask | (
                    bq_t[:, :, :, None] > (bq_k[:, :, None, :] + max_causal_window_size)
                )

            dots.masked_fill_(mask, mask_value)
            del mask

        mask = bq_k[:, :, None, :] == -1
        dots.masked_fill_(mask, mask_value)
        del mask

        if input_mask is not None:
            h = b // input_mask.shape[0]
            if self.autopad:
                input_mask = pad_to_multiple(
                    input_mask, window_size, dim=-1, value=False
                )
            input_mask = input_mask.reshape(-1, windows, window_size)  # type: ignore
            mq = mk = input_mask
            mk = look_around(mk, pad_value=False, **look_around_kwargs)
            mask = mq[:, :, :, None] * mk[:, :, None, :]
            mask = merge_dims(0, 1, expand_dim(mask, 1, h))
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhje->bhie", attn, bv)
        out = out.reshape(-1, s, e)  # type: ignore

        if self.autopad:
            out = out[:, :orig_t, :]

        return out.reshape(*shape)
