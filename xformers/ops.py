# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers import _is_triton_available

if _is_triton_available:
    from xformers.triton.softmax import softmax as triton_softmax


def masked_matmul(a, b, mask=None):
    if torch.overrides.has_torch_function((a, b, mask)):
        return torch.overrides.handle_torch_function(
            masked_matmul, (a, b, mask), a, b, mask
        )

    att = a @ b

    if mask is None:
        return att

    if mask.dtype == torch.bool:
        # TODO: replace this with
        # torch.where(
        #   mask,
        #   torch.tensor(0, dtype=a.dtype, device=a.device),
        #   torch.tensor(float('-inf'), dtype=a.dtype, device=a.device))
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        att += mask
    return att


def softmax(a: torch.Tensor) -> torch.Tensor:
    if _is_triton_available and type(a) is torch.Tensor:
        # causal case is handled by CausalTensor
        return triton_softmax(a, mask=None, causal=False)
    else:
        return torch.softmax(a, dim=-1)
