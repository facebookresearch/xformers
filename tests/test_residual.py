# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers.components import PreNorm


class Passthrough(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args):
        return args


def test_pre_norm():
    # Check that passing the same tensor a bunch of times skips the extra normalizations
    x = torch.rand((3, 3))

    wrap = PreNorm(d_model=3, sublayer=Passthrough(), use_triton=False)
    outputs = wrap(inputs=[x, x, x])

    assert id(outputs[0]) == id(outputs[1])
