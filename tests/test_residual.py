# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import pytest
import torch

from xformers.components import NormalizationType, PreNorm


class Passthrough(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        return args


@pytest.mark.parametrize("normalization", [n.value for n in NormalizationType])
def test_pre_norm(normalization):
    # Check that passing the same tensor a bunch of times skips the extra normalizations
    x = torch.rand((3, 3), requires_grad=True)

    wrap = PreNorm(
        d_norm=3, sublayer=Passthrough(), normalization=normalization, use_triton=False
    )
    outputs = wrap(inputs=[x, x, x])

    assert id(outputs[0]) == id(outputs[1])

    # Check the BW pass
    torch.sum(outputs[0]).backward()
