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

from xformers.components.attention import (
    Attention,
    AttentionConfig,
    AttentionMask,
    register_attention,
)
from xformers.components.attention.core import _softmax
from xformers.components.input_projection import InputProjection, InputProjectionConfig


def _either_or(a: Optional[int], b: int) -> int:
    return a if a is not None else b


@dataclass
class CompositionalAttentionConfig(AttentionConfig):
    dim_model: int
    num_heads: int
    dim_attn: Optional[int] = None
    num_rules: Optional[int] = None
    dim_key: Optional[int] = None
    dim_value: Optional[int] = None
    dim_selection: Optional[int] = None
    dropout: float
    qk_rule: bool = False
    nonlinear: bool = False
    q_compose: bool = False
    bias: bool = True
    causal: Optional[bool] = False
    in_proj_container: Optional[InputProjection] = None
    use_separate_proj_weight: Optional[bool] = False


@register_attention("compositional", CompositionalAttentionConfig)
class CompositionalAttention(Attention):
    """Compositional Attention, as proposed in
    "Compositional Attention: Disentangling search and retrieval"_, S. Mittal et al.

    A key insight from this proposal is that the attention mechanism can be conceived as two steps:
    a search and a retrieval operation. When queried, the model can search for the most relevant information
    (Softmax(QKt)), then retrieve information given the Value.

    Contrary to the original attention proposal, which does not consider interactions in between heads,
    the compositional attention will consider all possible interactions and softmax over that dimension,
    so that the information retrieved covers the most relevant dimensions. The number of heads and rules to
    use is thus typically smaller than for a comparable traditional Transformer, and asking for the same number of heads
    may not fit in memory.

    Args:
        dim_model: dimension of the incoming latent space
        num_heads: number of heads *for the search operation*
        dim_attn: dimension (embedding) of the attention
        num_rules: number of rules to consider *for the retrieval operation*
        dim_selection: dimension of the scoring/selection space for the retrievals
        dim_key, dim_value: dimensions of K and V, if different from Q
        dropout: attention dropout probability
        qk_rule: QK product will drive the retrieval process
        nonlinear: use a non linear method to score the retrievals
        bias: use bias in the initial projection step
        causal: causal computations (attend to the past only)

    _"Compositional Attention: Disentangling search and retrieval": https://arxiv.org/pdf/2110.09419v1.pdf
    """

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        dim_attn: Optional[int] = None,
        num_rules: Optional[int] = None,
        dim_selection: Optional[int] = None,
        dim_key: Optional[int] = None,
        dim_value: Optional[int] = None,
        dropout=0.0,
        qk_rule=False,
        nonlinear=False,
        q_compose=False,
        in_proj_container: Optional[InputProjection] = None,
        use_separate_proj_weight: Optional[bool] = False,
        bias=True,
        causal=False,
        *_,
        **__,
    ):
        super().__init__()

        # Define the inherited flags
        self.requires_skip_multi_head = (
            True  # This attention owns the multi-head mechanism
        )

        # Handle defaults / undefined values
        self.dim_model = dim_model
        num_rules = _either_or(num_rules, num_heads)
        dim_selection = _either_or(dim_selection, dim_model // num_heads)

        # All the initial definition plumbing
        dim_attn = _either_or(dim_attn, dim_model)
        dim_key = _either_or(dim_key, dim_model)
        dim_value = _either_or(dim_value, dim_model)

        self.in_proj_container = (
            in_proj_container
            if in_proj_container is not None
            else InputProjection(
                query_proj_params=InputProjectionConfig(dim_model, dim_key, bias=bias),
                key_proj_params=InputProjectionConfig(dim_model, dim_key, bias=bias)
                if use_separate_proj_weight
                else None,
                value_proj_params=InputProjectionConfig(dim_model, dim_value, bias=bias)
                if use_separate_proj_weight
                else None,
            )
        )

        self.num_heads = num_heads
        self.num_rules = num_rules
        self.qk_rule = qk_rule
        self.dim_selection = dim_selection
        self.nonlinear = nonlinear
        self.q_compose = q_compose

        self.dropout_module = nn.Dropout(dropout)
        self.dim_head = dim_model // num_heads
        self.value_dim = dim_attn // num_rules

        assert (
            self.value_dim * num_rules == dim_attn
        ), "value_dim must be divisible by num_rules"

        self.scaling = self.dim_head**-0.5
        self.scaling_values = self.dim_selection**-0.5

        self.out_proj = nn.Linear(self.num_heads * self.value_dim, dim_model, bias=bias)

        if self.qk_rule:
            self.value_k = nn.Linear(self.value_dim, self.dim_selection, bias=bias)
            if self.q_compose:
                self.value_q = nn.Linear(self.dim_head, self.dim_selection, bias=bias)
            else:
                self.value_q = nn.Linear(
                    dim_model, self.dim_selection * self.num_heads, bias=bias
                )
        else:
            if self.q_compose:
                self.value_q = nn.Linear(self.dim_head, self.dim_selection, bias=bias)
            else:
                self.value_q = nn.Linear(
                    dim_model, self.dim_selection * self.num_heads, bias=bias
                )
            if self.nonlinear:
                self.score_network: nn.Module = nn.Sequential(
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

        # Properties specific to this attention mechanism
        self.supports_attention_mask = True
        self.supports_key_padding_mask = False

        self._reset_parameters()

    def _reset_parameters(self):
        # NOTE: in_proj_container is already initialized

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
            att_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
        """

        B, Sq, E = q.shape
        _, Sk, _ = k.shape

        assert E == self.dim_model

        # First define projected query/key/values
        # We keep the projected and original tensors in flight,
        # depending on the options the original values could be reused
        q_unprojected = q
        q, k, v = self.in_proj_container(query=q, key=k, value=v)
        q *= self.scaling

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

        attn_weights = _softmax(attn_weights, causal=self.causal)

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

        # - extracted values are the original attention (inc. all the values) weighted by value score
        attn = (attn * v_score).sum(dim=3).view(B, Sq, self.num_heads * self.value_dim)

        # Final attention projection, same as other mechanisms
        attn = self.out_proj(attn)

        return attn
