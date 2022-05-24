# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: Initially suggested by Jason Ramapuram, see
# https://github.com/facebookresearch/xformers/issues/203

import pickle

from torch import nn

from xformers.factory import xFormer, xFormerConfig

test_config = [
    {
        "reversible": False,  # Turn on to test the effect of using reversible layers
        "block_type": "encoder",
        "num_layers": 2,
        "dim_model": 768,
        "layer_norm_style": "pre",
        "multi_head_config": {
            "num_heads": 12,
            "residual_dropout": 0.1,
            "use_rotary_embeddings": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": 0.1,
                "causal": False,
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": 0.1,
            "activation": "gelu",
            "hidden_layer_multiplier": 4,
        },
    }
]


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.xformer = xFormer.from_config(xFormerConfig(test_config))


def test_pickling():
    test = ViT()
    pickle.dumps(test)
