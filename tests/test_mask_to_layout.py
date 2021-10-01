# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.helpers import mask_to_layout


def test_mask_to_layout():
    BLOCK = 16
    SIZE = 128
    LAYOUT_SIZE = SIZE // BLOCK

    # All ones
    mask1 = torch.ones((SIZE, SIZE), dtype=torch.bool)
    layout1 = mask_to_layout(mask1, BLOCK)
    ref1 = torch.ones((LAYOUT_SIZE, LAYOUT_SIZE), dtype=torch.int)
    assert torch.allclose(layout1, ref1)

    # Diagonal -> expect block diagonal
    mask2 = torch.eye(SIZE, dtype=torch.bool)
    layout2 = mask_to_layout(mask2, BLOCK)
    ref2 = torch.eye(LAYOUT_SIZE, dtype=torch.int)
    assert torch.allclose(layout2, ref2)

    # Lower triangular, without the diagonal
    # note that the layout will need to have the diagonal, else the coefficients close enough would not be computed
    mask3 = torch.tril(torch.ones((SIZE, SIZE)), diagonal=-1).to(torch.bool)
    layout3 = mask_to_layout(mask3, BLOCK)
    ref3 = torch.tril(torch.ones((LAYOUT_SIZE, LAYOUT_SIZE)), diagonal=0).to(torch.int)
    assert torch.allclose(layout3, ref3)

    # Handle heads properly
    mask = torch.cat((mask1, mask2, mask3))
    layout = mask_to_layout(mask, BLOCK)
    assert torch.allclose(layout, torch.cat((ref1, ref2, ref3)))

    # Catch problematic dimensions
    mask_off = torch.ones((SIZE + 3, SIZE), dtype=torch.bool)
    with pytest.raises(AssertionError):
        mask_to_layout(mask_off, BLOCK)
