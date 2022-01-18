# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from xformers.components.attention import Attention, AttentionConfig, register_attention

_mask_type_warning = True

_use_triton = torch.cuda.is_available()
if _use_triton:
    try:
        from triton.ops.blocksparse import matmul as blocksparse_matmul
        from triton.ops.blocksparse import softmax as blocksparse_softmax

        from xformers.triton.softmax import MaskType

    except ImportError as e:
        logging.warning(
            f"Triton is not available: {e}.\nBlockSparse attention will not be available"
        )
        _use_triton = False


if _use_triton:

    @dataclass
    class BlockSparseAttentionConfig(AttentionConfig):
        layout: torch.Tensor  # The dimensions of the random features
        block_size: int
        dropout: float
        num_heads: int

    @register_attention("blocksparse", BlockSparseAttentionConfig)
    class BlockSparseAttention(Attention):
        r"""
        Thin wrap over the Triton blocksparse computations. The sparsity pattern is determined through the layout.

        .. warning: the layout is assumed to have the dimensions [heads, seq, seq].
            If some dimensions are missing, we assume that the same layout is to be used across heads.

        .. note: it is possible to pass a specific per batch mask in the forward call,
            but this will not lead to any speed up.
            Any constant sparsity pattern is better passed through the layout parameter.
        """

        def __init__(
            self,
            layout: torch.Tensor,
            block_size: int = 16,
            dropout: float = 0.0,
            num_heads: int = 1,  # optional, used to adapt the layout if in need
            *args,
            **kwargs,
        ):
            if layout.dim() == 2:
                logging.warning(
                    "The layout passed is lacking a head dimension and a batch dimension"
                )
                logging.warning(
                    "Now assuming that the same layout is to be used across all heads"
                )
                layout = layout.unsqueeze(0).expand(num_heads, -1, -1)
                logging.warning(f"New layout dimensions: {layout.shape}")

            assert block_size >= 16, "Minimum block size is 16, for now at least"

            super().__init__()
            self.attn_drop = nn.Dropout(dropout, inplace=False)

            # Pure blocksparse data
            self.layout = layout
            self.block_size = block_size

            # blocksparse operators
            self.sparse_dot_sdd = blocksparse_matmul(
                self.layout,
                self.block_size,
                "sdd",
                trans_a=False,
                trans_b=True,
            )
            self.sparse_dot_dsd = blocksparse_matmul(
                self.layout,
                self.block_size,
                "dsd",
                trans_a=False,
                trans_b=False,
            )
            self.sparse_softmax = blocksparse_softmax(self.layout, self.block_size)

            # make sure that the head dimension is not folded down with the batch
            self.requires_head_dimension = True

            # key padding mask and attention mask must be passed in separately
            self.requires_separate_masks = True

        def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            att_mask: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            scale: float = 1.0,
            *args,
            **kwargs,
        ) -> torch.Tensor:

            # NOTE:
            # The attention mask will be taken into account when computing the softmax
            # meaning that non-masked values which are present in the initial blocksparse layout will be computed.
            # If blocks are to be constantly masked, better perf would thus be reached by signalling them out in the
            # initial attention setup

            if att_mask is not None and att_mask.dtype != q.dtype:
                global _mask_type_warning
                if _mask_type_warning:
                    logging.warning(
                        "Attention mask has to be multiplicative. Fixing that but this slows things down"
                    )
                    _mask_type_warning = False  # Only warn once
                att_mask = att_mask.to(q.dtype)

            assert (
                att_mask is None or att_mask.dim() == 2
            ), "The attention mask is constant across heads, expected dimensions are [seq x seq]"

            # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
            # When the computations are block sparse, the matrix types change along the way:
            # - (sparse) attention matrix = (dense) Kt * (dense) Q
            assert (
                q.shape[-2] == k.shape[-2]
            ), "Blocksparse requires the same dimensions for K and Q for now"
            sparse_att_mat = self.sparse_dot_sdd(q, k)

            # - softmax on the sparse attention matrix
            sparse_att_mat = self.sparse_softmax(
                sparse_att_mat,
                scale=scale,
                key_padding_mask=key_padding_mask,
                attn_mask=att_mask,
                key_padding_mask_mode=MaskType.MUL,
                attn_mask_mode=MaskType.MUL,
            )

            # - then (dense) attention is (sparse) attention matrix * dense (value)
            a = self.sparse_dot_dsd(sparse_att_mat, v)

            return a
