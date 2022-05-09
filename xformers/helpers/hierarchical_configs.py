# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import copy
from dataclasses import dataclass
from typing import Any, Dict, List

from xformers.components.residual import LayerNormStyle


@dataclass
class BasicLayerConfig:
    embedding: int
    attention_mechanism: str
    patch_size: int
    stride: int
    padding: int
    seq_len: int


def get_hierarchical_configuration(
    layer_basic_configs: List[BasicLayerConfig],
    layernorm_style: LayerNormStyle = LayerNormStyle.Pre,
    use_rotary_embeddings: bool = True,
    mlp_multiplier: int = 4,
    dim_head=32,
):
    """
    A small helper to generate hierarchical xformers configurations,
    which correspond for instance to poolformer or swin architectures.

    Contrary to more "classical" Transformer architectures, which conserve the sequence/context
    length across layers, hierarchical Transformers trade the sequence length for the embedding dimension
    """

    base_config: Dict[str, Any] = {
        "block_type": "encoder",
        "dim_model": 0,
        "use_triton": False,
        "layer_norm_style": str(layernorm_style),
        "multi_head_config": {
            "num_heads": 0,
            "use_rotary_embeddings": use_rotary_embeddings,
            "attention": {
                "name": "TBD",
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "activation": "gelu",
            "hidden_layer_multiplier": mlp_multiplier,
            "dropout": 0.0,
        },
        "position_encoding_config": {
            "name": "learnable",
            "seq_len": 0,
            "add_class_token": False,
        },
        "patch_embedding_config": {
            "in_channels": 3,
            "kernel_size": 0,
            "stride": 0,
            "padding": 0,
        },
    }

    xformers_config = []
    in_channels = 3

    for layer_basic_config in layer_basic_configs:
        lc = copy.deepcopy(base_config)

        # Fill in the changing model dimensions
        lc["dim_model"] = layer_basic_config.embedding

        # Update the patches
        lc["patch_embedding_config"] = {
            "in_channels": in_channels,
            "kernel_size": layer_basic_config.patch_size,
            "stride": layer_basic_config.stride,
            "padding": layer_basic_config.padding,
        }

        # Update the number of channels for the next layer
        in_channels = lc["dim_model"] * 1

        lc["position_encoding_config"]["seq_len"] = layer_basic_config.seq_len

        # Fill in the number of heads
        lc["multi_head_config"]["num_heads"] = layer_basic_config.embedding // dim_head
        assert layer_basic_config.embedding % dim_head == 0

        # Fill in the attention mechanism
        lc["multi_head_config"]["attention"][
            "name"
        ] = layer_basic_config.attention_mechanism

        print(lc)
        xformers_config.append(lc)

    return xformers_config
