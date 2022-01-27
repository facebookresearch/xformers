# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components.positional_embedding import RotaryEmbedding

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


@pytest.mark.parametrize("device", DEVICES)
def test_rotary_embeddings(device):
    rotary = RotaryEmbedding(EMB).to(device)

    # Generate dummy inputs
    q = torch.ones((BATCH, HEADS, SEQ, EMB), device=device)  # uniform on purpose
    k = q.clone()

    k_rot, q_rot = rotary(q, k)

    # Check that the sequences now encode relative position information
    att = torch.einsum("bhne,bhme->bhnm", q, k)
    att_rot = torch.einsum("bhne,bhme->bhnm", q_rot, k_rot)

    # - the attention for the same positions is not changed
    assert torch.allclose(torch.diag(att[0, 0, :, :]), torch.diag(att_rot[0, 0, :, :]))

    # - the post-rotary attention is more focused on the diagonal
    att_rot -= att_rot[
        0, 0, 0, 0
    ].clone()  # all diagonal elements will have the same value
    att_rot = (
        att_rot <= 1e-4
    )  # all non diagonal elements had lower attention than diagonal (+ float tolerance)
    assert torch.all(att_rot)

    # Test that different sequence lengths is ok
    _, _ = rotary(q[:, :, :-16, :], k)
