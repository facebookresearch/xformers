# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: Initially suggested by Jason Ramapuram, see
# https://github.com/facebookresearch/xformers/issues/203

import pickle
from copy import deepcopy

import pytest
from torch import nn

from xformers import _is_triton_available
from xformers.factory import xFormer, xFormerConfig

test_config = [
    {
        "reversible": False,
        "block_type": "encoder",
        "num_layers": 2,
        "dim_model": 768,
        "residual_norm_style": "pre",
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
            "name": "FusedMLP",
            "dropout": 0.1,
            "activation": "gelu",
            "hidden_layer_multiplier": 4,
        },
    }
]


class ViT(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        test_config[0]["feedforward_config"]["name"] = mlp
        xformer_config = xFormerConfig(test_config)
        self.xformer = xFormer.from_config(xformer_config)


MLPs = ["MLP"]
if _is_triton_available():
    MLPs.append("FusedMLP")


@pytest.mark.parametrize("mlp", MLPs)
def test_pickling(mlp):
    test = ViT(mlp)
    _ = pickle.dumps(test)
    _ = deepcopy(test)
