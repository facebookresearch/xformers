# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from xformers.components.residual import ResidualNormStyle


@dataclass
class BasicLayerConfig:
    embedding: int
    attention_mechanism: str
    patch_size: int
    stride: int
    padding: int
    seq_len: int
    feedforward: str
    normalization: str = "layernorm"
    repeat_layer: int = 1


def get_hierarchical_configuration(
    layer_base_configs: List[BasicLayerConfig],
    residual_norm_style: ResidualNormStyle = ResidualNormStyle.Pre,
    use_rotary_embeddings: bool = True,
    mlp_multiplier: int = 4,
    in_channels: int = 3,
    dim_head: Optional[int] = None,
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
        "residual_norm_style": str(residual_norm_style),
        "multi_head_config": {
            "num_heads": 1,
            "use_rotary_embeddings": use_rotary_embeddings,
            "attention": {
                "name": "TBD",
            },
        },
        "feedforward_config": {
            "name": "TBD",
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
            "in_channels": in_channels,
            "kernel_size": 0,
            "stride": 0,
            "padding": 0,
        },
    }

    xformers_config = []
    in_channels = in_channels

    for layer_base_config in layer_base_configs:
        lc = copy.deepcopy(base_config)

        lc["normalization"] = layer_base_config.normalization

        # Fill in the changing model dimensions
        lc["dim_model"] = layer_base_config.embedding

        # Update the patches
        lc["patch_embedding_config"] = {
            "in_channels": in_channels,
            "kernel_size": layer_base_config.patch_size,
            "stride": layer_base_config.stride,
            "padding": layer_base_config.padding,
        }

        # Update the number of channels for the next layer
        in_channels = lc["dim_model"] * 1

        lc["position_encoding_config"]["seq_len"] = layer_base_config.seq_len

        # Fill in the number of heads (defaults to 1)
        if dim_head is not None:
            lc["multi_head_config"]["num_heads"] = (
                layer_base_config.embedding // dim_head
            )
            assert layer_base_config.embedding % dim_head == 0

        # Fill in the attention mechanism
        lc["multi_head_config"]["attention"][
            "name"
        ] = layer_base_config.attention_mechanism

        # FIll in the feedforward
        lc["feedforward_config"]["name"] = layer_base_config.feedforward

        print(lc)
        xformers_config.append(lc)

        # Handle repeated layers (without the patch embeddings)
        if layer_base_config.repeat_layer > 1:
            lc_repeat = copy.deepcopy(lc)
            lc_repeat.pop("patch_embedding_config")
            xformers_config += [lc_repeat] * (layer_base_config.repeat_layer - 1)

    return xformers_config
