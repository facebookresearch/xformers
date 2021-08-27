from dataclasses import dataclass
from functools import partial, reduce
from inspect import isfunction
from operator import mul
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from xformers.components.attention import Attention, AttentionConfig, register_attention


@dataclass
class SinkhornSelfAttentionConfig(AttentionConfig):
    block_size: int
    temperature: float
    sinkhorm_iter: int
    num_heads: int
    require_key_mask: bool


@register_attention("sinkhorn", SinkhornSelfAttentionConfig)
class SinkhornAttention(Attention):
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        block_size: int = 256,
        temperature: float = 0.7,
        sinkhorm_iter: int = 7,
        *args,
        **kwargs,
    ):
        """
        Sparse Sinkhorn Attention
        https://arxiv.org/abs/2002.11296

        Code largely based on https://github.com/lucidrains/sinkhorn-transformer

        The paper's notation are kept wherever possible

        #TODO only support encoding only settings
        """
        super().__init__()
        self.bucket_size = block_size
        self.sinkhorn_iter = sinkhorm_iter
        self.temperature = temperature
        self.attn_drop = nn.Dropout(dropout)
        self.num_heads = num_heads
        self.sort_net = AttentionSortNet(block_size, temperature, sinkhorm_iter)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ):
        # q, k, v: (B * nh, S, hs)
        bh = q.size(0)
        bsz = bh // self.num_heads
        tgt_len = k.size(1)
        head_dim = q.size(-1)
        buckets = q.shape[1] // self.bucket_size

        b_q = bucket(buckets, q)
        b_k, b_v = map(partial(bucket, buckets), (k, v))  # BH * bct * n_b * D

        R = self.sort_net(q, k)
        R = R.type_as(q).to(q)

        b_k_r = reorder_buckets(b_k, R).reshape(
            bh, buckets, -1, head_dim
        )  # BH * bct * 2n_b * D
        b_v_r = reorder_buckets(b_v, R).reshape(
            bh, buckets, -1, head_dim
        )  # BH * bct * 2n_b * D

        b_k = torch.cat((b_k_r, b_k), dim=2)
        b_v = torch.cat((b_v_r, b_v), dim=2)

        dots = torch.einsum("buie,buje->buij", b_q, b_k) * (head_dim ** -0.5)

        mask_value = -10000

        if not key_padding_mask and att_mask is not None:
            # try to recover from attn_mask
            att_mask = att_mask.reshape(bsz, self.num_heads, -1, tgt_len)[:, 0, :, :]
            assert att_mask.size(1) == 1
            key_padding_mask = att_mask[:, 0, :].eq(0)

        # # mask
        # if key_padding_mask is not None:
        assert key_padding_mask is not None
        q_mask = default(
            key_padding_mask.eq(0),
            lambda: torch.ones((bsz, tgt_len), device=q.device).bool(),
        )
        kv_mask = q_mask
        mq, mk = bucket(buckets, q_mask), bucket(buckets, kv_mask)  # B * bkt * n_b

        def expand_head_and_merge_into_batch(x):
            return merge_dims(0, 1, expand_dim(x.unsqueeze(1), 1, self.num_heads))

        mq, mk = map(expand_head_and_merge_into_batch, (mq, mk))  # BH * bkt * n_b
        mk_r = batched_index_select(mk, R.abs().argmax(dim=-1))
        mk_r = mk_r.reshape(bh, buckets, -1)
        mk = torch.cat((mk_r, mk), dim=2)
        mask = mq[:, :, :, None] * mk[:, :, None, :]

        dots.masked_fill_(~mask, mask_value)
        del mask

        dots = dots.softmax(dim=-1)
        dots = self.attn_drop(dots)

        attn = torch.einsum("buij,buje->buie", dots, b_v)
        attn = unbucket(attn)

        return attn


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))


def expand_dim(t, dim, k):
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def merge_dims(ind_from, ind_to, tensor):
    shape = list(tensor.shape)
    arr_slice = slice(ind_from, ind_to + 1)
    shape[arr_slice] = [reduce(mul, shape[arr_slice])]
    return tensor.reshape(*shape)


def default(x, d):
    if x is None:
        return d if not isfunction(d) else d()
    return x


def bucket(buckets, t, dim=1):
    shape = list(t.shape)
    shape[dim : dim + 1] = [buckets, -1]
    return t.reshape(*shape)


def reorder_buckets(t, r):
    return torch.einsum("buv,bvtd->butd", r, t)


def unbucket(t, dim=1):
    shape = list(t.shape)
    shape[dim : dim + 2] = [-1]
    return t.reshape(*shape)


class AttentionSortNet(nn.Module):
    def __init__(self, bucket_size, temperature, sinkhorn_iter):
        super().__init__()
        self.bucket_size = bucket_size
        self.temperature = temperature
        self.sinkhorn_iter = sinkhorn_iter

    def forward(self, q, k, topk=1):
        dim = q.size(-1)

        buckets = q.shape[1] // self.bucket_size
        kv_buckets = k.shape[1] // self.bucket_size

        b_q = bucket(buckets, q)
        b_k = bucket(kv_buckets, k)

        sq = b_q.mean(dim=2)  # TODO original paper uses sum
        sk = b_k.mean(dim=2)

        R = torch.einsum("bie,bje->bij", sq, sk).to(q) * (dim ** -0.5)

        return gumbel_sinkhorn(F.relu(R), self.sinkhorn_iter, self.temperature)


def log(t, eps=1e-6):
    return torch.log(t + eps)


def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def sinkhorn_sorting_operator(r, n_iters=8):
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)
