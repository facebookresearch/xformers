# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
"""
This file contains biases that can be used as the `attn_bias` argument in
:attr:`xformers.ops.memory_efficient_attention`.
Essentially, a bias is a Tensor which will be added to the ``Q @ K.t`` before
computing the ``softmax``.


The goal of having custom made classes (instead of dense tensors) is that
we want to avoid having to load the biases from memory in the kernel, for
performance reasons. We also want to be able to know before-hand which
parts of the attention matrix we will need to compute (eg causal masks).


Some very common biases are LowerTriangularMask and BlockDiagonalMask.
"""

from mslk.attention.fmha.attn_bias import (  # noqa: E402, F401
    _GappySeqInfo,
    _PaddedSeqLenInfo,
    _SeqLenInfo,
    AttentionBias,
    BlockDiagonalCausalFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionFromBottomRightMask,
    BlockDiagonalCausalLocalAttentionMask,
    BlockDiagonalCausalLocalAttentionPaddedKeysMask,
    BlockDiagonalCausalMask,
    BlockDiagonalCausalWithOffsetGappyKeysMask,
    BlockDiagonalCausalWithOffsetPaddedKeysMask,
    BlockDiagonalGappyKeysMask,
    BlockDiagonalLocalAttentionFromBottomRightGappyKeysMask,
    BlockDiagonalLocalAttentionPaddedKeysMask,
    BlockDiagonalMask,
    BlockDiagonalPaddedKeysMask,
    LocalAttentionFromBottomRightMask,
    LowerTriangularFromBottomRightLocalAttentionMask,
    LowerTriangularFromBottomRightMask,
    LowerTriangularMask,
    LowerTriangularMaskWithTensorBias,
    PagedBlockDiagonalCausalLocalPaddedKeysMask,
    PagedBlockDiagonalCausalWithOffsetGappyKeysMask,
    PagedBlockDiagonalCausalWithOffsetPaddedKeysMask,
    PagedBlockDiagonalGappyKeysMask,
    PagedBlockDiagonalPaddedKeysMask,
    VARLEN_BIASES,
)
