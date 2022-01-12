# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# Credits: this is heavily inspired by the official implementation, present in
# https://github.com/sarthmit/Compositional-Attention
# Original author: Sarthak Mittal

# This is a simplified version, for the sake of clarity, and because some features could be exposed later
# via the library directly.
# In particular, code paths for TPUs, quantization and gumbel softmax have been removed
# We're also following the same dimension ordering as in the rest of the xformers library
# which is to say [Batch, Sequence, Embedding] wherever possible

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xformers import _is_triton_available
from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    register_attention,
)

if _is_triton_available:
    from xformers.triton.softmax import softmax


@dataclass
class CompositionalAttentionConfig(AttentionConfig):
    num_heads: int
    dim_head: int
    num_rules: Optional[int]
    dropout: float
    qk_rule: bool = False
    dim_selection: Optional[int] = None
    nonlinear: bool = False
    q_compose: bool = False
    dim_attn: Optional[int] = None
    kdim: Optional[int] = None
    vdim: Optional[int] = None
    bias: bool = True
    causal: Optional[bool] = False


@register_attention("compositional", CompositionalAttentionConfig)
class CompositionalAttention(Attention):
    """Compositional Attention, as proposed in
    "Compositional Attention: Disentangling search and retrieval"_, S. Mittal et al.

    A key insight from this proposal is that the attention mechanism can be conceived as two steps:
    a search and a retrieval operation. When queried, the model can search for the most relevant information
    (Softmax(QKt)), then retrieve information given the Value.

    Contrary to the original attention proposal, which does not consider interactions in between heads,
    the comppositional attention will consider all possible interactions and softmax over that dimension,
    so that the information retrieved covers the most relevant dimensions. The number of heads and rules to
    use is thus typically smaller than for a comparable traditional Transformer, and asking for the same number of heads
    may not fit in memory.

    Args:
        num_heads: The number of heads *for the search operation*
        dim_head: Latent space for a given head
        dim_selection: dimension of the scoring/selection space for the retrievals
        numn_rules: The number of rules to consider *for the retrieval operation*
        dropout: attention dropout probability
        qk_rule: QK product will drive the retrieval process
        nonlinear: use a non linear method to score the retrievals
        dim_attn: dimension (embedding) of the attention
        kdim, vdim: dimensions of K and V, if different from Q
        bias: use bias in the initial projection step
        causal: causal computations (attend to the past only)

    _"Compositional Attention: Disentangling search and retrieval": https://arxiv.org/pdf/2110.09419v1.pdf
    """

    def __init__(
        self,
        num_heads,
        dim_head,
        num_rules=None,
        dropout=0.0,
        qk_rule=False,
        dim_selection=None,
        nonlinear=False,
        q_compose=False,
        dim_attn=None,
        kdim=None,
        vdim=None,
        bias=True,
        causal=False,
        *_,
        **__,
    ):
        super().__init__()

        # Define the inherited flags
        self.requires_input_projection = (
            False  # This attention handles its own projection
        )

        self.requires_skip_multi_head = (
            True  # This attention owns the multi-head mechanism
        )

        # Handle defaults / undefined values
        num_rules = num_heads if num_rules is None else num_rules
        dim_embed = int(num_heads * dim_head)
        dim_attn = dim_embed if dim_attn is None else dim_attn
        dim_selection = (
            dim_embed // num_heads if dim_selection is None else dim_selection
        )

        # All the initial definition plumbing
        self.dim_embed = dim_embed
        self.dim_attn = dim_attn
        self.kdim = kdim if kdim is not None else dim_embed
        self.vdim = vdim if vdim is not None else dim_embed
        self.qkv_same_dim = self.kdim == dim_embed and self.vdim == dim_embed

        self.num_heads = num_heads
        self.num_rules = num_rules
        self.qk_rule = qk_rule
        self.dim_selection = dim_selection
        self.nonlinear = nonlinear
        self.q_compose = q_compose

        self.dropout_module = nn.Dropout(dropout)
        self.dim_head = dim_embed // num_heads
        self.value_dim = dim_attn // num_rules

        assert (
            self.dim_head * num_heads == self.dim_embed
        ), "dim_embed must be divisible by num_heads"

        assert (
            self.value_dim * num_rules == self.dim_attn
        ), "value_dim must be divisible by num_rules"

        self.scaling = self.dim_head ** -0.5
        self.scaling_values = self.dim_selection ** -0.5

        self.k_proj = nn.Linear(self.kdim, dim_embed, bias=bias)
        self.v_proj = nn.Linear(self.vdim, dim_attn, bias=bias)
        self.q_proj = nn.Linear(dim_embed, dim_embed, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.value_dim, dim_embed, bias=bias)

        if self.qk_rule:
            self.value_k = nn.Linear(self.value_dim, self.dim_selection, bias=bias)
            if self.q_compose:
                self.value_q = nn.Linear(self.dim_head, self.dim_selection, bias=bias)
            else:
                self.value_q = nn.Linear(
                    dim_embed, self.dim_selection * self.num_heads, bias=bias
                )
        else:
            if self.q_compose:
                self.value_q = nn.Linear(self.dim_head, self.dim_selection, bias=bias)
            else:
                self.value_q = nn.Linear(
                    dim_embed, self.dim_selection * self.num_heads, bias=bias
                )
            if self.nonlinear:
                self.score_network = nn.Sequential(
                    nn.Linear(
                        self.dim_selection + self.value_dim,
                        self.dim_selection,
                        bias=bias,
                    ),
                    nn.ReLU(),
                    nn.Linear(self.dim_selection, 1, bias=bias),
                )
            else:
                self.score_network = nn.Linear(
                    self.dim_selection + self.value_dim, 1, bias=bias
                )

        self.causal = causal

        self.reset_parameters()

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        if self.qk_rule:
            nn.init.xavier_uniform_(self.value_k.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.value_q.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.value_q.weight)
            if self.nonlinear:
                nn.init.xavier_uniform_(self.score_network[0].weight)
                nn.init.xavier_uniform_(self.score_network[2].weight)
            else:
                nn.init.xavier_uniform_(self.score_network.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        att_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Input shape: Time x Batch x Channel

        Args:
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """

        B, Sq, E = q.shape
        _, Sk, _ = k.shape

        assert E == self.dim_embed

        # First define projected query/key/values
        # We keep the projected and original tensors in flight,
        # depending on the options the original values could be reused
        q_unprojected = q
        q = self.q_proj(q) * self.scaling
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Init causal mask if needed, now that we know the context length
        if self.causal and (
            self._causal_mask is None or self._causal_mask.shape[0] != Sk
        ):
            self._causal_mask = AttentionMask.make_causal(Sq, Sq, device=q.device)

        # Convenience, create an attention mask if a tensor was passed
        # This sanitizes different mask types being passed, from now on it's additive
        if isinstance(att_mask, torch.Tensor):
            # By default we don't know of the causality, and a check would be expensive
            att_mask_additive: Optional[AttentionMask] = (
                AttentionMask.from_bool(att_mask)
                if att_mask.dtype == torch.bool
                else AttentionMask(att_mask, is_causal=False)
            )
        else:
            att_mask_additive = None

        # Handle the attention and key padding masks
        if self._causal_mask is not None:
            # Optionally add the causal mask
            if att_mask_additive is not None:
                att_mask_additive += self._causal_mask
            else:
                att_mask_additive = self._causal_mask

        # Flatten the heads or the rules
        q = (
            q.view(B, Sq, self.num_heads, self.dim_head)
            .movedim(2, 1)
            .flatten(0, 1)  # [B * num_heads, Sq, dim_head]
        )
        k = (
            k.view(B, Sk, self.num_heads, self.dim_head).movedim(2, 1).flatten(0, 1)
        )  # [B * num_heads, Sk, dim_head]
        v = v.view(B, -1, self.num_rules, self.value_dim).movedim(2, 1).flatten(0, 1)

        # Compute the search: Softmax(QKt)
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [B * self.num_heads, Sq, Sk]

        if att_mask_additive is not None:
            attn_weights += att_mask_additive.values

        if _is_triton_available:
            attn_weights = softmax(attn_weights, causal=self.causal)
        else:
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)

        attn_weights = attn_weights.view(B, self.num_heads, Sq, Sk)
        attn_probs = self.dropout_module(attn_weights)

        # Now compute the information retrieval
        # keep all the heads in flight, we'll score the different possibilities
        # - compute all the possible retrievals
        v = v.view(B, 1, self.num_rules, Sk, self.value_dim)
        attn_probs = attn_probs.unsqueeze(2)
        attn = torch.matmul(attn_probs, v).view(
            B, self.num_heads, self.num_rules, Sq, self.value_dim
        )

        attn = attn.movedim(3, 1)  # [B, Sq, H, Rules, Values]

        # - search the most appropriate retrieval among all the values
        if self.q_compose:
            v_q = self.value_q(q.transpose(0, 1)).view(
                B, Sq, self.num_heads, 1, self.dim_selection
            )
        else:
            v_q = self.value_q(q_unprojected).view(
                B, Sq, self.num_heads, 1, self.dim_selection
            )

        if self.qk_rule:
            v_q *= self.scaling_values
            v_k = (
                self.value_k(attn)
                .view(B, Sq, self.num_heads, self.num_rules, self.dim_selection)
                .transpose(4, 3)
                .contiguous()
            )
            v_score = torch.matmul(v_q, v_k).view(
                B, Sq, self.num_heads, self.num_rules, 1
            )
        else:
            v_q = v_q.expand(-1, -1, -1, self.num_rules, -1)
            v_in = torch.cat([attn, v_q], dim=-1)
            v_score = self.score_network(v_in).view(
                B, Sq, self.num_heads, self.num_rules, 1
            )

        v_score = F.softmax(v_score, dim=3)

        attn = (attn * v_score).sum(dim=3).view(B, Sq, self.num_heads * self.value_dim)
        attn = self.out_proj(attn)

        return attn
