# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
    scaled_dot_product_attention,
    scaled_query_key_softmax,
)
from xformers.components.attention.utils import (
    bool_mask_to_additive,
    iterative_pinv,
    reshape_key_padding_mask,
)

logger = logging.getLogger("xformers")


@dataclass
class NystromSelfAttentionConfig(AttentionConfig):
    """
    num_heads               Number of heads.
    num_landmarks           Number of landmarks to use for softmax approximation. 64 often sufficient for a good
                            approximation according to https://arxiv.org/pdf/2102.03902.pdf.
    causal                  Apply a causal mask, in that the attention cannot be applied to the future.
    use_razavi_pinverse     If true, use iterative method from (Razavi et al. 2014) to approximate the Moore-Penrose
                            inverse, otherwise use standard torch inverse.
    pinverse_original_init  True if using original initialization when calculating Moore-Penrose pseudo inverse using
                            method from (Razavi et al. 2014).
                            False if using exact coefficient computation (leads to faster convergence).
    inv_iterations          Number of iterations for calculating the Moore-Penrose pseudo inverse.
    v_skip_connection       A module that will take V as input and will be added as a skip connection to the
                            softmax approximation. A skip connection is added in the paper to help with training.
    conv_kernel_size        Kernel size for convolution optionally added to help in training.
                            If v_skip_connection is not specified, this will be used to define the default
                            depth wise convolution used as a skip connection.
                            If both conv_kernel_size and v_skip_connection are None, no skip connection will
                            be added.
    landmark_pooling        Which module to use when computing landmarks. Default is AdaptiveAvgPool2d.
    """

    num_heads: int
    num_landmarks: Optional[int]
    landmark_pooling: Optional[nn.Module]
    causal: Optional[bool]
    pinverse_original_init: Optional[bool]
    inv_iterations: Optional[int]
    v_skip_connection: Optional[nn.Module]
    conv_kernel_size: Optional[int]
    use_razavi_pinverse: Optional[bool]


class AvgPool(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x: torch.Tensor):
        # Average independently for every segment in the sequence dimension
        seq_len = x.shape[1]
        head_dim = x.shape[2]
        segments = seq_len // self.n
        assert segments > 0, "num_landmarks should be smaller than the sequence length"

        # Dimensions are a match
        if seq_len % self.n == 0:
            return x.reshape(
                -1,
                self.n,
                segments,
                head_dim,
            ).mean(dim=-2)

        # Handle the last segment boundary being off
        n_round = self.n - seq_len % self.n

        x_avg_round = (
            x[:, : n_round * segments, :]
            .reshape(-1, n_round, segments, head_dim)
            .mean(dim=-2)
        )
        x_avg_off = (
            x[:, n_round * segments :, :]
            .reshape(-1, self.n - n_round, segments + 1, head_dim)
            .mean(dim=-2)
        )
        return torch.cat((x_avg_round, x_avg_off), dim=-2)


@register_attention("nystrom", NystromSelfAttentionConfig)
class NystromAttention(Attention):
    # TODO: update defaults for use_razavi_pinverse and inv_iterations
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_landmarks: int = 64,
        landmark_pooling: Optional[nn.Module] = None,
        causal: bool = False,
        use_razavi_pinverse: bool = True,
        pinverse_original_init: bool = False,
        inv_iterations: int = 6,  # recommended default in paper was 6.
        v_skip_connection: Optional[nn.Module] = None,
        conv_kernel_size: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Nystrom attention mechanism, from Nystromformer_.
        ::

            "A Nystrom-based Algorithm for Approximating Self-Attention."
            Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., Singh, V. (2021)

            Reference codebase: https://github.com/mlpen/Nystromformer

        .. _Nystromformer: https://arxiv.org/pdf/2102.03902.pdf

        """
        super().__init__()
        # merged key padding mask and attention mask is not accepted
        self.requires_separate_masks = True
        self.num_landmarks = num_landmarks
        # TODO: should be able to not have to pass in num_heads
        self.num_heads = num_heads
        self.use_razavi_pinverse = use_razavi_pinverse
        self.pinverse_original_init = pinverse_original_init
        self.inv_iterations = inv_iterations
        self.attn_drop = nn.Dropout(dropout)
        self.skip_connection = v_skip_connection
        self.causal = causal

        if self.skip_connection is None and conv_kernel_size is not None:
            self.skip_connection = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

        if landmark_pooling is not None:
            self.landmark_pooling = landmark_pooling
        else:
            self.landmark_pooling = AvgPool(n=self.num_landmarks)

        # Optional lower triangular masks for causal attention
        self.causal_mask_1: Optional[torch.Tensor] = None
        self.causal_mask_2: Optional[torch.Tensor] = None
        self.causal_mask_3: Optional[torch.Tensor] = None

        # This attention does not support attention masks
        self.supports_attention_mask = False
        self.supports_key_padding_mask = True

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        r"""
        key_padding_mask    Only a key padding mask is accepted here. The size must be (batch size, sequence length) or
                            (batch size * num_heads, 1, sequence length). If dimensions are not correct, the mask will
                            be ignored. An additive mask is expected, meaning float values using "-inf" to mask values
        """

        batched_dim = k.size(0)
        seq_len = k.size(-2)
        tt = {"dtype": q.dtype, "device": q.device}

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                logger.warning(
                    "Bool mask found, but an additive mask is expected. Converting but this is slow"
                )

                key_padding_mask = bool_mask_to_additive(key_padding_mask)

            if key_padding_mask.ndim == 2:
                key_padding_mask = reshape_key_padding_mask(
                    key_padding_mask, batched_dim
                )

            zeros = torch.zeros_like(key_padding_mask)
            ones = torch.ones_like(key_padding_mask)
            is_masked = torch.isinf(-key_padding_mask)

            # _mask takes 1 if the token is not padded, otherwise 0.
            _mask = torch.where(is_masked, zeros, ones)
            _mask = _mask.transpose(2, 1)
            assert _mask.shape == (batched_dim, q.shape[1], 1)

            # Mask q and k before pooling
            # https://github.com/mlpen/Nystromformer/blob/main/code/attention_nystrom.py#L31
            q = q * _mask
            k = k * _mask

            assert key_padding_mask.size() == (batched_dim, 1, seq_len), (
                f"key_padding_mask has invalid dimensions {key_padding_mask.size()}."
                f" Must have dimensions {batched_dim, 1, seq_len} or (batch_size, {seq_len})."
            )

        if self.num_landmarks >= seq_len:
            mask: Optional[torch.Tensor] = None

            if self.causal:
                mask = self._triu_mask(batched_dim, seq_len, seq_len, **tt)

            if key_padding_mask is not None:
                mask = key_padding_mask if mask is None else mask + key_padding_mask

            x = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=mask)

        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)

            if self.causal and (
                self.causal_mask_1 is None
                or (batched_dim, seq_len, self.num_landmarks)
                != self.causal_mask_1.size()
            ):
                self.causal_mask_1 = self._triu_mask(
                    batched_dim, seq_len, self.num_landmarks, **tt
                )
                self.causal_mask_2 = self._triu_mask(
                    batched_dim, self.num_landmarks, self.num_landmarks, **tt
                )
                self.causal_mask_3 = self._triu_mask(
                    batched_dim, self.num_landmarks, seq_len, **tt
                )

            mask_3: Optional[torch.Tensor] = self.causal_mask_3
            if key_padding_mask is not None:
                mask_3 = (
                    key_padding_mask if mask_3 is None else mask_3 + key_padding_mask
                )

            kernel_1 = scaled_query_key_softmax(q=q, k=k_landmarks, att_mask=None)
            kernel_2 = scaled_query_key_softmax(
                q=q_landmarks, k=k_landmarks, att_mask=None
            )
            kernel_3 = scaled_dot_product_attention(
                q=q_landmarks, k=k, v=v, att_mask=mask_3
            )

            kernel_2_inv = (
                iterative_pinv(
                    kernel_2, self.inv_iterations, self.pinverse_original_init
                )
                if self.use_razavi_pinverse
                else torch.linalg.pinv(kernel_2)
            )

            x = torch.matmul(
                torch.matmul(
                    kernel_1,
                    kernel_2_inv,
                ),
                kernel_3,
            )

        if self.skip_connection:
            # Assumption here is that v is 3D.
            v_conv = self.skip_connection(
                v.reshape(-1, self.num_heads, v.size(-2), v.size(-1))
            )
            x += v_conv.reshape(-1, v_conv.size(-2), v_conv.size(-1))
        x = self.attn_drop(x)
        return x

    def _triu_mask(self, dim_1: int, dim_2: int, dim_3: int, **kwargs) -> torch.Tensor:
        device = kwargs["device"]
        dtype = kwargs["dtype"]

        return torch.triu(
            torch.ones(dim_2, dim_3, dtype=dtype, device=device) * float("-inf"),
            diagonal=1,
        ).expand(
            dim_1, -1, -1
        )  # micro optim, save memory on the batch dimension
