from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
    scaled_dot_product_attention,
    scaled_query_key_softmax,
)
from xformers.components.attention.utils import iterative_pinv


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
        Nystrom attention mechanism, from
        "
        Nystromformer: A Nystrom-based Algorithm for Approximating Self-Attention.
        Xiong, Y., Zeng, Z., Chakraborty, R., Tan, M., Fung, G., Li, Y., Singh, V. (2021)
        "
        ArXiv: https://arxiv.org/pdf/2102.03902.pdf
        Code: https://github.com/mlpen/Nystromformer
        """
        super().__init__()

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
            self.landmark_pooling = torch.nn.AdaptiveAvgPool2d(
                (self.num_landmarks, None)
            )

        # Optional lower triangular masks for causal attention
        self.causal_mask_1: Optional[torch.Tensor] = None
        self.causal_mask_2: Optional[torch.Tensor] = None
        self.causal_mask_3: Optional[torch.Tensor] = None

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs
    ):
        batched_dim = k.size(0)
        seq_len = k.size(-2)

        if self.num_landmarks == seq_len:
            mask = None
            if self.causal:
                mask = self._tril_mask(batched_dim, seq_len, seq_len)
            x = scaled_dot_product_attention(q=q, k=k, v=v, att_mask=mask)

        else:
            q_landmarks = self.landmark_pooling(q)
            k_landmarks = self.landmark_pooling(k)

            if self.causal and self.causal_mask_1 is None:
                self.causal_mask_1 = self._tril_mask(
                    batched_dim, seq_len, self.num_landmarks
                ).to(q.device)
                self.causal_mask_2 = self._tril_mask(
                    batched_dim, self.num_landmarks, self.num_landmarks
                ).to(q.device)
                self.causal_mask_3 = self._tril_mask(
                    batched_dim, self.num_landmarks, seq_len
                ).to(q.device)

            kernel_1 = scaled_query_key_softmax(q, k_landmarks, self.causal_mask_1)
            kernel_2 = scaled_query_key_softmax(
                q_landmarks, k_landmarks, self.causal_mask_2
            )
            kernel_3 = scaled_dot_product_attention(
                q=q_landmarks, k=k, v=v, att_mask=self.causal_mask_3
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

    def _tril_mask(self, dim_1: int, dim_2: int, dim_3: int):
        return torch.tril(torch.ones(dim_1, dim_2, dim_3, dtype=torch.bool), diagonal=0)
