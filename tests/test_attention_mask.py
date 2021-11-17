# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from xformers.components.attention import AttentionMask


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires a CUDA device"
)
def test_mask_creation():
    # Check that we can create from boolean
    bool_mask = torch.rand((256, 256)) > 0.5
    additive_mask = AttentionMask.from_bool(bool_mask)
    assert (bool_mask == additive_mask.to_bool()).all()

    bool_mask = torch.rand((2, 256, 256)) > 0.5
    additive_mask = AttentionMask.from_bool(bool_mask)
    assert (bool_mask == additive_mask.to_bool()).all()
    assert additive_mask.ndim == bool_mask.ndim

    # Check that we can create from multiplicative
    ref_mask = torch.randint(0, 2, (256, 256))
    mul_mask = ref_mask.float()
    additive_mask = AttentionMask.from_multiplicative(mul_mask)
    assert (ref_mask.bool() == additive_mask.to_bool()).all()

    # Check the causal mask
    causal_mask = AttentionMask.make_causal(256, 256)
    assert (torch.tril(torch.ones(256, 256)).bool() == causal_mask.to_bool()).all()
    assert causal_mask.is_causal

    causal_mask = AttentionMask.make_causal(256)
    assert (torch.tril(torch.ones(256, 256)).bool() == causal_mask.to_bool()).all()

    causal_mask = AttentionMask.make_causal(256, 128)
    assert (torch.tril(torch.ones(256, 128)).bool() == causal_mask.to_bool()).all()

    # Check that we can add masks
    bool_mask_1 = torch.rand((256, 256)) > 0.5
    add_mask_1 = AttentionMask.from_bool(bool_mask_1)

    bool_mask_2 = torch.rand((256, 256)) > 0.5
    add_mask_2 = AttentionMask.from_bool(bool_mask_2)

    assert ((add_mask_1 + add_mask_2).to_bool() == (bool_mask_1 & bool_mask_2)).all()

    # Check type handling
    additive_mask = AttentionMask.from_bool(torch.rand((256, 256)) > 0.5)
    additive_mask = additive_mask.to(device=torch.device("cuda"))
    assert "cuda" in str(additive_mask.values.device)

    # Check that the causal flag is maintained
    additive_mask = AttentionMask.make_causal(256, 256)
    additive_mask = additive_mask.to(device=torch.device("cuda"))
    assert additive_mask.is_causal
