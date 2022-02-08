# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
from dataclasses import dataclass
from re import L
from typing import Optional
from functools import partial

import torch

from xformers import _is_triton_available
from xformers.components.attention import Attention, AttentionConfig, register_attention, attention_patterns
from xformers.components.attention.utils import bool_mask_to_additive
import torch.nn.functional as F

_mask_type_warning = True

if _is_triton_available:
    from triton.ops.blocksparse import matmul as blocksparse_matmul
    from triton.ops.blocksparse import softmax as blocksparse_softmax

    from xformers.triton.softmax import MaskType
    from xformers.triton.utils import gpu_capabilities_older_than_70

    # Blocksparse requires Tensor cores
    if gpu_capabilities_older_than_70():
        logging.warning(
            "Blocksparse is not available: the current GPU does not expose Tensor cores"
        )
        _is_triton_available = False


def _split_heads(num_heads: int, x: torch.Tensor):
    shape = list(x.shape)
    shape[:1] = [-1, num_heads]
    return x.reshape(*shape)

if _is_triton_available:

    @dataclass
    class BlockSparseLocalAttentionConfig(AttentionConfig):
        # layout: torch.Tensor  # The dimensions of the random features
        seq_len: int
        block_size: int
        dropout: float
        num_heads: int

    @register_attention("blocksparse_local", BlockSparseLocalAttentionConfig)
    class BlockSparseLocalAttention(Attention):
        r"""
        Thin wrap over the Triton blocksparse computations. The sparsity pattern is determined through the layout.

        .. warning: the layout is assumed to have the dimensions [heads, seq, seq].
            If some dimensions are missing, we assume that the same layout is to be used across heads.

        .. warning: for now, the sequence (context) length has to be a power of two. This constraint could
            be relaxed in the future.

        .. note: it is possible to pass a specific per batch mask in the forward call,
            but this will not lead to any speed up.
            Any constant sparsity pattern is better passed through the layout parameter.
        """

        def __init__(
            self,
            seq_len: int,
            block_size: int = 512,
            dropout: float = 0.0,
            block_unit: int = 16,
            num_heads: int = 1,  # optional, used to adapt the layout if in need
            *args,
            **kwargs,
        ):

            def _generate_mask(heads, seq_len, block_size):
                local_mask = torch.zeros(heads, seq_len, seq_len)
                for block_start in range(0, seq_len, block_size):
                    local_mask[:, block_start:block_start+block_size, block_start:block_start+block_size] = 1
                return local_mask

            block_mask = _generate_mask(num_heads, seq_len, block_size)
            layout = attention_patterns.pattern_to_layout(block_mask, block_unit)

            if layout.dim() == 2:
                layout = layout.unsqueeze(0).expand(num_heads, -1, -1)

            assert block_unit >= 16, "Minimum block size is 16, for now at least"

            super().__init__()

            self.num_heads = num_heads
            self.attn_drop = torch.nn.Dropout(dropout, inplace=False)

            # Pure blocksparse data
            self.layout = layout
            self.block_size = block_size
            self.block_unit = block_unit

            # blocksparse operators
            self.sparse_dot_sdd = blocksparse_matmul(
                self.layout,
                self.block_unit,
                "sdd",
                trans_a=False,
                trans_b=True,
            )
            self.sparse_dot_dsd = blocksparse_matmul(
                self.layout,
                self.block_unit,
                "dsd",
                trans_a=False,
                trans_b=False,
            )
            self.sparse_softmax = blocksparse_softmax(self.layout, self.block_unit)

            # make sure that the head dimension is not folded down with the batch
            self.requires_head_dimension = False

            # key padding mask and attention mask must be passed in separately
            self.requires_separate_masks = True

            self.requires_same_k_q_dimensions = True

        def update_mask_type(self, mask: torch.Tensor):
            global _mask_type_warning
            if _mask_type_warning:
                logging.warning(
                    "Mask has to be additive. Fixing that but this slows things down"
                )
            mask = bool_mask_to_additive(mask)

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
            r"""
            att_mask            A 2D attention mask. The dtype must be the same as q. An additive mask is expected,
                                meaning float values using "-inf" to mask values.
            key_padding_mask    A mask with size (batch size x sequence length). The dtype must be the same as q.
                                An additive mask is expected, meaning float values using "-inf" to mask values
            """

            # NOTE:
            # The attention mask will be taken into account when computing the softmax
            # meaning that non-masked values which are present in the initial blocksparse layout will be computed.
            # If blocks are to be constantly masked, better perf would thus be reached by signalling them out in the
            # initial attention setup

            bh = q.size(0)
            orig_seq_len = q.size(1)
            bsz = bh // self.num_heads
            head_dim = q.size(-1)

            if key_padding_mask is None:
                key_padding_mask = torch.zeros(int(q.shape[0]/self.num_heads), q.size(-2))
            att_mask = None # HACK

            # pad the input length to factors of bucket size
            def _pad_to_window_size(x, window_size):
                seq_len = x.size(-2)
                pad_len = (window_size - seq_len % window_size) % window_size
                return F.pad(x, (0,0,0,pad_len), value=0), pad_len
            q, _ = _pad_to_window_size(q, self.block_size)
            k, _ = _pad_to_window_size(k, self.block_size)
            v, _ = _pad_to_window_size(v, self.block_size)

            if key_padding_mask.shape[1] % self.block_size != 0:
                pad_len = (self.block_size - key_padding_mask.shape[1] % self.block_size) % self.block_size
                key_padding_mask = torch.cat([key_padding_mask, key_padding_mask.new_ones(key_padding_mask.size(0), pad_len).to(key_padding_mask)], dim=1)
            key_padding_mask[key_padding_mask == 1] = float('-inf')
            key_padding_mask = key_padding_mask.to(q)

            q, k, v = map(partial(_split_heads, self.num_heads), (q, k, v))

            if att_mask is not None and att_mask.dtype == torch.bool:
                self.update_mask_type(att_mask)
            if key_padding_mask is not None and key_padding_mask.dtype == torch.bool:
                self.update_mask_type(key_padding_mask)

            assert (
                att_mask is None or att_mask.dim() == 2
            ), "The attention mask is constant across heads, expected dimensions are [seq x seq]"

            assert (
                q.shape[-2] == k.shape[-2]
            ), "Blocksparse requires the same dimensions for K and Q for now"

            assert (
                q.shape[-2] == self.layout.shape[-2] * self.block_unit
            ), "Actual sequence size and layout are inconsistent"
            assert (
                k.shape[-2] == self.layout.shape[-2] * self.block_unit
            ), "Actual sequence size and layout are inconsistent"

            assert math.log(
                q.shape[-2], 2
            ).is_integer(), (
                "For now blocksparse only works on power-of-two sequence lengths"
            )

            # Blocksparse only works on fp16
            q_dtype = q.dtype

            if q_dtype != torch.float16:
                q, k, v = q.half(), k.half(), v.half()

                if att_mask is not None:
                    att_mask = att_mask.half()

                if key_padding_mask is not None:
                    key_padding_mask = key_padding_mask.half()

            # Self-attend: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
            # When the computations are block sparse, the matrix types change along the way:
            # - (sparse) attention matrix = (dense) Kt * (dense) Q
            q = q / math.sqrt(q.size(-1))
            sparse_att_mat = self.sparse_dot_sdd(q, k)

            # - softmax on the sparse attention matrix
            sparse_att_mat = self.sparse_softmax(
                sparse_att_mat,
                scale=scale,
                key_padding_mask=key_padding_mask,
                attn_mask=att_mask,
                key_padding_mask_mode=MaskType.ADD,
                attn_mask_mode=MaskType.ADD,
            )

            sparse_att_mat = self.attn_drop(sparse_att_mat)

            # - then (dense) attention is (sparse) attention matrix * dense (value)
            a = self.sparse_dot_dsd(sparse_att_mat, v)

            out = a.view(bh, -1, head_dim)[:,:orig_seq_len]

            if q_dtype != torch.float16:
                return out.to(q_dtype)

            return out
