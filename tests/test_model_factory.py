# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.factory.model_factory import xFormer, xFormerConfig

BATCH = 20
SEQ = 512
EMB = 384
DEVICES = (
    [torch.device("cpu")]
    if not torch.cuda.is_available()
    else [
        torch.device("cuda")
    ]  # save a bit on CI for now, we have seperate cpu and gpu jobs
)

encoder_configs = {
    "reversible": False,
    "block_type": "encoder",
    "dim_model": 384,
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": 64,
        "dim_model": EMB,
    },
    "num_layers": 3,
    "multi_head_config": {
        "num_heads": 4,
        "residual_dropout": 0,
        "attention": {
            "name": "linformer",
            "dropout": 0,
            "causal": True,
            "seq_len": 512,
        },
        "dim_model": EMB,
    },
    "feedforward_config": {
        "name": "MLP",
        "dropout": 0,
        "activation": "relu",
        "hidden_layer_multiplier": 4,
        "dim_model": EMB,
    },
}

decoder_configs = {
    "block_type": "decoder",
    "dim_model": 384,
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": 64,
        "dim_model": EMB,
    },
    "num_layers": 2,
    "multi_head_config_masked": {
        "num_heads": 4,
        "residual_dropout": 0,
        "dim_model": EMB,
        "attention": {
            "name": "linformer",
            "dropout": 0,
            "causal": True,
            "seq_len": 512,
        },
    },
    "multi_head_config_cross": {
        "num_heads": 4,
        "residual_dropout": 0,
        "dim_model": EMB,
        "attention": {
            "name": "linformer",
            "dropout": 0,
            "causal": True,
            "seq_len": 512,
        },
    },
    "feedforward_config": {
        "name": "MLP",
        "dropout": 0,
        "activation": "relu",
        "hidden_layer_multiplier": 4,
        "dim_model": EMB,
    },
}

test_configs_list = [encoder_configs, decoder_configs]
test_configs_dict = {"encoder": encoder_configs, "decoder": decoder_configs}

""" Test all the model configurations saved in model_presets. """


@pytest.mark.parametrize("config", [test_configs_list, test_configs_dict])
@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.parametrize("tie_embedding_weights", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_presets(config, reversible, tie_embedding_weights, device):
    # Build the model
    if isinstance(config, list):
        config[0]["reversible"] = reversible
    else:
        config["encoder"]["reversible"] = reversible

    modelConfig = xFormerConfig(config, tie_embedding_weights)
    if isinstance(modelConfig.stack_configs, dict):
        for k, blockConfig in modelConfig.stack_configs.items():
            assert blockConfig.layer_position
    else:
        for blockConfig in modelConfig.stack_configs:
            assert blockConfig.layer_position

    model = xFormer.from_config(modelConfig).to(device)

    # Dummy inputs, test a forward
    inputs = (torch.rand((BATCH, SEQ), device=device) * 10).abs().to(torch.int)

    input_mask = torch.randn(SEQ, dtype=torch.float, device=device)
    input_mask[input_mask < 0.0] = -float("inf")
    _ = model(inputs, encoder_input_mask=input_mask, decoder_input_mask=input_mask)
