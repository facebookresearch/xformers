from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from xformers.components.attention import Attention, AttentionConfig, register_attention
from xformers.components.attention.core import (
    scaled_dot_product_attention,
    scaled_query_key_softmax,
)


@dataclass(init=False)
class NystromSelfAttentionConfig(AttentionConfig):
    """
    num_heads               Number of heads.
    num_landmarks           Number of landmarks to use for softmax approximation. 64 often sufficient for a good
                            approximation according to https://arxiv.org/pdf/2102.03902.pdf.
    pinverse_original_init  True if using original initialization when calculating Moore-Penrose pseudo inverse.
                            False if using exact coefficient computation (leads to faster convergence).
    inv_iterations          Number of iterations for calculating the Moore-Penrose pseudo inverse.
    conv_kernel_size        Kernel size for convolution optionally added to help in training.
    """

    num_heads: int
    num_landmarks: Optional[int]
    pinverse_original_init: Optional[bool]
    inv_iterations: Optional[int]
    conv_kernel_size: Optional[int]


@register_attention("nystrom")
class NystromAttention(Attention):
    def __init__(
        self,
        dropout: float,
        num_heads: int,
        num_landmarks: int = 64,
        pinverse_original_init: bool = False,
        inv_iterations: int = 6,  # recommended default in paper was 6.
        conv_kernel_size: Optional[int] = None,
        *args,
        **kwargs
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
        self.pinverse_original_init = pinverse_original_init
        self.use_conv = conv_kernel_size is not None
        self.attn_drop = nn.Dropout(dropout)

        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(conv_kernel_size, 1),
                padding=(conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):

        head_dim = k.size(-1)
        seq_len = k.size(-2)
        # TODO: apply attention mask to q and k. Mask dimensions SxS or BxS?

        if self.num_landmarks == seq_len:
            x = scaled_dot_product_attention(q, k, v, att_mask)

        else:
            q_landmarks = q.reshape(
                -1,
                self.num_landmarks,
                seq_len // self.num_landmarks,
                head_dim,
            ).mean(dim=-2)
            k_landmarks = k.reshape(
                -1,
                self.num_landmarks,
                seq_len // self.num_landmarks,
                head_dim,
            ).mean(dim=-2)

            kernel_1 = scaled_query_key_softmax(q, k_landmarks, None)
            kernel_2 = scaled_query_key_softmax(q_landmarks, k_landmarks, None)
            # TODO: apply attention mask
            kernel_3 = scaled_dot_product_attention(q_landmarks, k, v, None)

            x = torch.matmul(
                torch.matmul(kernel_1, self._iterative_inv(kernel_2)), kernel_3
            )

        if self.use_conv:
            # TODO: apply attention mask to v.
            # Assumption here is that v is 3D.
            v_conv = self.conv(v.reshape(-1, self.num_heads, v.size(-2), v.size(-1)))
            x += v_conv.reshape(-1, v_conv.size(-2), v_conv.size(-1))
        x = self.attn_drop(x)
        return x

    def _iterative_inv(self, mat: torch.Tensor, n_iter=6):
        """
        Computing the Moore-Penrose inverse.
        Use an iterative method from (Razavi et al. 2014) to approximate the Moore-Penrose inverse via efficient
        matrix-matrix multiplications.
        """

        i = torch.eye(mat.size(-1), device=mat.device)
        k = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.pinverse_original_init:
            # This original implementation is more conservative to compute coefficient of Z_0.
            v = 1 / torch.max(torch.sum(k, dim=-2)) * k.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster
            # convergence.
            v = (
                1
                / torch.max(torch.sum(k, dim=-2), dim=-1).values[:, None, None]
                * k.transpose(-1, -2)
            )

        for _ in range(n_iter):
            kv = torch.matmul(k, v)
            v = torch.matmul(
                0.25 * v,
                13 * i - torch.matmul(kv, 15 * i - torch.matmul(kv, 7 * i - kv)),
            )
        return v

    @classmethod
    def from_config(cls, config: AttentionConfig) -> "Attention":
        return cls(**NystromSelfAttentionConfig.as_patchy_dict(config))
