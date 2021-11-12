# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch

from xformers.components.attention.utils import (
    maybe_merge_masks,
    reshape_key_padding_mask,
)


def test_reshape_key_padding_mask():
    batch_size = 2
    num_heads = 2
    seq_len = 4

    batched_dim = batch_size * num_heads

    key_padding_mask = torch.randint(0, 2, (batch_size, seq_len)).to(dtype=torch.bool)
    reshaped_mask = reshape_key_padding_mask(
        key_padding_mask=key_padding_mask, batched_dim=batched_dim
    )
    assert reshaped_mask.size() == (batched_dim, 1, seq_len)

    merged_mask = maybe_merge_masks(
        att_mask=None,
        key_padding_mask=key_padding_mask,
        batch_size=batch_size,
        src_len=seq_len,
        num_heads=num_heads,
    )
    assert torch.equal(merged_mask, reshaped_mask.expand(-1, seq_len, -1))

    key_padding_mask = torch.randint(0, 2, (batched_dim, seq_len)).to(dtype=torch.bool)
    reshaped_mask = reshape_key_padding_mask(
        key_padding_mask=key_padding_mask, batched_dim=batched_dim
    )
    assert reshaped_mask.size() == (batched_dim, 1, seq_len)
