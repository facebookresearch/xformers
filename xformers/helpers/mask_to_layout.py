# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import torch


def mask_to_layout(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Given a mask and blocksize, return the corresponding layout
    which makes sure that all the positives in the mask are covered
    """
    assert mask.ndim >= 2, "We're expecting [Heads, Seq, Seq] or [Seq, Seq]"
    _should_squeeze = False

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
        _should_squeeze = True

    assert (
        mask.shape[1] % block_size == 0 and mask.shape[2] % block_size == 0
    ), "We're only handling masks divisible by block_size"

    # Now mark the mask
    layout = torch.nn.functional.max_pool2d(
        mask.to(torch.float), kernel_size=block_size, stride=block_size
    )
    layout = layout.to(torch.int)

    if _should_squeeze:
        layout.squeeze_(0)

    return layout
