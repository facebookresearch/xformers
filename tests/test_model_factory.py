# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import pytest
import torch

import xformers.factory.weight_init as xformers_weight_init
from xformers.factory import xFormer, xFormerConfig, xFormerWeightInit

BATCH = 2
SEQ = 16
EMB = 16
VOC = 16

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
    "dim_model": EMB,
    "residual_norm_style": "pre",
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": VOC,
        "dim_model": EMB,
    },
    "num_layers": 3,
    "multi_head_config": {
        "num_heads": 4,
        "residual_dropout": 0,
        "attention": {
            "name": "scaled_dot_product",
            "dropout": 0,
            "causal": True,
            "seq_len": SEQ,
        },
        "dim_model": EMB,
    },
    "feedforward_config": {
        "name": "MLP",
        "dropout": 0,
        "activation": "relu",
        "hidden_layer_multiplier": 4,
        "dim_model": EMB,
        "number_of_experts": 4,
        "gate_config": "top_2",
    },
}

decoder_configs = {
    "block_type": "decoder",
    "dim_model": EMB,
    "residual_norm_style": "pre",
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": VOC,
        "dim_model": EMB,
    },
    "num_layers": 2,
    "multi_head_config_masked": {
        "num_heads": 4,
        "residual_dropout": 0,
        "dim_model": EMB,
        "attention": {
            "name": "scaled_dot_product",
            "dropout": 0,
            "causal": True,
            "seq_len": SEQ,
        },
    },
    "multi_head_config_cross": {
        "num_heads": 4,
        "residual_dropout": 0,
        "dim_model": EMB,
        "attention": {
            "name": "scaled_dot_product",
            "dropout": 0,
            "causal": True,
            "seq_len": SEQ,
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
@pytest.mark.parametrize("residual_norm_style", ["pre", "post", "deepnorm"])
@pytest.mark.parametrize("device", DEVICES)
def test_presets(
    config, reversible, tie_embedding_weights, residual_norm_style, device
):
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    # Build the model
    if isinstance(config, list):
        # Only the encoder can be reversible
        config[0]["reversible"] = reversible

        config[0]["residual_norm_style"] = residual_norm_style
        config[1]["residual_norm_style"] = residual_norm_style
    else:
        config["encoder"]["reversible"] = reversible
        config["encoder"]["residual_norm_style"] = residual_norm_style
        config["decoder"]["residual_norm_style"] = residual_norm_style

    modelConfig = xFormerConfig(config, tie_embedding_weights)
    if isinstance(modelConfig.stack_configs, dict):
        for _, blockConfig in modelConfig.stack_configs.items():
            assert blockConfig.layer_position
    else:
        for blockConfig in modelConfig.stack_configs:
            assert blockConfig.layer_position

    context = (
        pytest.raises(AssertionError)
        if reversible and (tie_embedding_weights or residual_norm_style == "deepnorm")
        else nullcontext()
    )

    with context:
        model = xFormer.from_config(modelConfig).to(device)

        def check_against_default(p):
            # check that a different gain than 1 was used
            vanilla = p.clone()
            torch.nn.init.xavier_normal_(p, gain=1)
            change = torch.abs((torch.std(vanilla) - torch.std(p)) / torch.std(p))
            assert change > 0.1

        # Check deepnorm init, if applicable
        if residual_norm_style == "deepnorm":
            for n, p in model.encoders.named_parameters():
                # Check the MHA
                if "in_proj_weight" in n:
                    # self attention projection, check that the value projection has been changed
                    M, _ = p.shape
                    K = M // 3

                    value_rel_std = torch.abs(
                        torch.std(p[:K, :]) - torch.std(p[-K:, :])
                    )
                    qp_rel_std = torch.abs(torch.std(p[:K, :]) - torch.std(p[K:-K, :]))

                    # Check that the value proj init has been changed by more than the noise
                    assert (
                        value_rel_std / qp_rel_std > 2
                    ), f"{(value_rel_std/qp_rel_std)}"

                if "v_proj_weight" in n:
                    check_against_default(p)

                if "mha.proj" in n and "weight" in n:
                    check_against_default(p)

                # Check the feedforward
                if "feedforward" in n and "weight" in n:
                    check_against_default(p)

        # Dummy inputs, test a forward
        inputs = (torch.rand((BATCH, SEQ), device=device) * 10).abs().to(torch.int)

        input_mask = torch.randn(SEQ, dtype=torch.float, device=device)
        input_mask[input_mask < 0.0] = -float("inf")
        outputs = model(
            inputs, encoder_input_mask=input_mask, decoder_input_mask=input_mask
        )

        # Test a BW
        loss = torch.sum(torch.abs(outputs))
        loss.backward()

        # If we requested tied embedding weights, check that this is the case indeed
        if tie_embedding_weights and not reversible:
            assert model.encoders[0].pose_encoding == model.decoders[0].pose_encoding


@pytest.mark.parametrize("weight_init", [w.value for w in xFormerWeightInit])
@pytest.mark.parametrize("feedforward", ["MLP", "Conv2DFeedforward"])
@pytest.mark.parametrize("deepnorm", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_weight_init(weight_init, feedforward, deepnorm, device):
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    config = test_configs_dict

    if deepnorm:
        config["encoder"]["residual_norm_style"] = "deepnorm"
        config["encoder"]["feedforward_config"]["name"] = feedforward

        config["decoder"]["residual_norm_style"] = "deepnorm"

    # Make sure that all the init methods catch all the weights
    xformers_weight_init._assert_if_not_initialized = True

    # Build the model
    config_instance = xFormerConfig(  # noqa
        config, tie_embedding_weights=False, weight_init=weight_init
    )

    _ = xFormer.from_config(config_instance).to(device)
