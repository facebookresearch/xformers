# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components.positional_embedding import RotaryEmbedding
from xformers.components.positional_embedding.rotary import (
    apply_rotary_pos_emb,
    rotate_half,
)

DEVICES = (
    [torch.device("cpu")]
    if not torch.cuda.is_available()
    else [
        torch.device("cuda")
    ]  # save a bit on CI for now, we have seperate cpu and gpu jobs
)
BATCH = 2
SEQ = 32
HEADS = 2
EMB = 32


def test_helper_methods():
    # rotate_half
    tens = torch.tensor([[0, 1, 2, 3], [3, 1, 2, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    tens_rotated = rotate_half(tens)
    assert torch.equal(
        tens_rotated,
        torch.tensor([[-2, -3, 0, 1], [-2, 0, 3, 1], [0, -1, 0, 1], [-1, 0, 1, 0]]),
    )

    # apply_rotary_pos_emb
    cos_test = torch.ones((1, 1, 4, 4))
    sin_test = cos_test.clone()
    q_test = 3 * torch.ones((2, 2, 3, 4))
    q_applied = apply_rotary_pos_emb(q_test, cos_test, sin_test)
    assert torch.equal(
        q_applied,
        torch.concat(
            (
                torch.zeros((2, 2, 3, 2), dtype=torch.float),
                6 * torch.ones((2, 2, 3, 2), dtype=torch.float),
            ),
            dim=-1,
        ),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_rotary_embeddings(device, dtype):
    rotary = RotaryEmbedding(EMB).to(device)

    # Generate dummy inputs
    q = torch.ones(
        (BATCH, HEADS, SEQ, EMB), device=device, dtype=dtype
    )  # uniform on purpose
    k = q.clone()

    q_rot, k_rot = rotary(q, k)

    assert q_rot.dtype == q.dtype
    assert k_rot.dtype == k.dtype

    # Check that the sequences now encode relative position information
    q, k = q.float(), k.float()
    q_rot, k_rot = q_rot.float(), k_rot.float()

    att = torch.einsum("bhne,bhme->bhnm", q, k)
    att_rot = torch.einsum("bhne,bhme->bhnm", q_rot, k_rot)

    # - the attention for the same positions is not meaningfully changed
    assert torch.allclose(
        torch.diag(att[0, 0, :, :]), torch.diag(att_rot[0, 0, :, :]), rtol=0.1
    )

    # - the post-rotary attention is more focused on the diagonal
    diag_max = torch.max(torch.diag(att_rot[0, 0, :, :]))
    att_rot -= diag_max
    att_rot = (
        att_rot <= 1e-4
    )  # all non diagonal elements had lower attention than diagonal (+ float tolerance)
    assert torch.all(att_rot)

    # Test that different sequence lengths is ok
    _, _ = rotary(q[:, :, :-16, :], k)
