# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import torch

from xformers.factory import xFormer, xFormerConfig
from xformers.helpers.hierarchical_configs import (
    BasicLayerConfig,
    get_hierarchical_configuration,
)

BATCH = 20
SEQ = 512
MODEL = 384


def test_hierarchical_transformer():
    image_size = 32

    base_hierarchical_configs = [
        BasicLayerConfig(
            embedding=64,
            attention_mechanism="pooling",
            patch_size=7,
            stride=4,
            padding=2,
            seq_len=image_size * image_size // 16,
            feedforward="MLP",
        ),
        BasicLayerConfig(
            embedding=128,
            attention_mechanism="pooling",
            patch_size=3,
            stride=2,
            padding=1,
            seq_len=image_size * image_size // 64,
            feedforward="MLP",
            repeat_layer=2,
        ),
        BasicLayerConfig(
            embedding=320,
            attention_mechanism="scaled_dot_product",
            patch_size=3,
            stride=2,
            padding=1,
            seq_len=image_size * image_size // 256,
            feedforward="MLP",
        ),
    ]

    # Fill in the gaps in the config
    xformer_config = get_hierarchical_configuration(
        base_hierarchical_configs,
        residual_norm_style="pre",
        use_rotary_embeddings=False,
        mlp_multiplier=4,
        dim_head=32,
    )
    config = xFormerConfig(xformer_config)
    hierarchical_xformer = xFormer.from_config(config)

    # Forward some dummy data
    dummy = torch.rand((2, 3, image_size, image_size))
    _ = hierarchical_xformer(dummy)
