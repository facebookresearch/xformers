# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.
from typing import Tuple

import torch

try:
    from torchembed._triton import fused_rope_forward as _torchembed_rope_forward

    _torchembed_available = True
except ImportError:
    _torchembed_available = False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        query: Query tensor of shape ``(*, seq_len, dim)``.
        key: Key tensor of shape ``(*, seq_len, dim)``.
        cos: Cosine values of shape ``(seq_len, rotary_dim)`` where
            ``rotary_dim <= dim``.
        sin: Sine values of shape ``(seq_len, rotary_dim)``.

    Returns:
        Tuple ``(rotated_query, rotated_key)`` with the same shapes as inputs.

    When the optional ``torchembed`` package is installed and the tensors
    are on CUDA, this dispatches to its fused triton kernel.  Otherwise a
    PyTorch reference path is used.
    """
    if _torchembed_available and query.is_cuda and query.device.type == "cuda":
        return _torchembed_rope_forward(query, key, cos, sin)

    rot_dim = cos.shape[-1]
    q_rot = query[..., :rot_dim]
    q_pass = query[..., rot_dim:]
    k_rot = key[..., :rot_dim]
    k_pass = key[..., rot_dim:]

    # Broadcast cos/sin to match query/key shape.
    # cos/sin are (seq_len, rot_dim); query is (..., seq_len, dim).
    # Add singleton dimensions at the front so cos/sin broadcast over leading dims.
    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    q_out = q_rot * cos + _rotate_half(q_rot) * sin
    k_out = k_rot * cos + _rotate_half(k_rot) * sin

    if q_pass.shape[-1] > 0:
        q_out = torch.cat([q_out, q_pass], dim=-1)
    if k_pass.shape[-1] > 0:
        k_out = torch.cat([k_out, k_pass], dim=-1)
    return q_out, k_out
