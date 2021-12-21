# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.cuda.amp import custom_bwd, custom_fwd

from xformers.triton.k_mem_efficient_attention import mem_efficient_fw


class mem_efficient_attention(torch.autograd.Function):
    """
    Implementing memory efficient attention, from
    "Self-attention Does Not Need O(n2) Memory", Rabe et al.

    https://arxiv.org/abs/2112.05682v2
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, q, k, v, bias):
        res = mem_efficient_fw(q, k, v, bias)

        return res

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        return None, None, None, None
