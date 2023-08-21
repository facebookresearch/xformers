# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

from xformers.factory.model_factory import xFormer, xFormerConfig

BATCH = 2
SEQ = 64
EMB = 48
VOCAB = 16
DEVICES = (
    [torch.device("cpu")]
    if not torch.cuda.is_available()
    else [torch.device("cuda")]  # save a bit on CI, we have separate cpu and gpu jobs
)

_test_config_encoder = {
    "reversible": False,
    "block_type": "encoder",
    "dim_model": EMB,
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": VOCAB,
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
    },
}

_test_config_decoder = {
    "block_type": "decoder",
    "dim_model": EMB,
    "position_encoding_config": {
        "name": "vocab",
        "seq_len": SEQ,
        "vocab_size": VOCAB,
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
            "seq_len": SEQ,
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

# Test a pure encoder, a pure decoder, an encoder/decoder stack
_test_configs = [
    [_test_config_encoder, _test_config_decoder],
    [_test_config_encoder],
]


def _rev_config(config, flag: bool):
    for c in filter(
        lambda x: x["block_type"] == "encoder",
        config,
    ):
        c["reversible"] = flag

    return config


@pytest.mark.parametrize("config", _test_configs)
@pytest.mark.parametrize("device", DEVICES)
def test_reversible_runs(config, device):

    # Build both a reversible and non-reversible model
    model_non_reversible = xFormer.from_config(
        xFormerConfig(_rev_config(config, False))
    ).to(device)
    model_reversible = xFormer.from_config(xFormerConfig(_rev_config(config, True))).to(
        device
    )

    # Dummy inputs, test a forward
    inputs = (torch.rand((BATCH, SEQ), device=device) * 10).abs().to(torch.int)
    _ = model_non_reversible(inputs)
    _ = model_reversible(inputs)


@pytest.mark.parametrize("device", DEVICES)
def test_reversible_no_alternate(device):

    # Check that we cannot build a non-coherent stack
    with pytest.raises(AssertionError):
        rev = dict(_test_config_encoder)  # we need to make a copy
        rev["reversible"] = True
        non_rev = dict(_test_config_encoder)
        non_rev["reversible"] = False

        _ = xFormer.from_config(xFormerConfig([rev, non_rev])).to(device)


@pytest.mark.parametrize("config", _test_configs)
@pytest.mark.parametrize("device", DEVICES)
def test_reversible_train(config, device):
    torch.manual_seed(0)
    random.seed(0)

    # Dummy inputs, test some training to make sure that we both can approximate the same thing to some extent
    # This is not super scientific, more of a foolproof catch
    def data():
        input_a = torch.zeros((BATCH, SEQ), device=device).to(torch.int)
        input_b = (torch.rand((BATCH, SEQ), device=device) * VOCAB).abs().to(torch.int)

        target_a = torch.zeros((BATCH, SEQ), device=device)
        target_b = torch.ones((BATCH, SEQ), device=device)

        if random.random() > 0.5:
            return torch.cat([input_a, input_b], dim=0), torch.cat(
                [target_a, target_b], dim=0
            )

        return torch.cat([input_b, input_a], dim=0), torch.cat(
            [target_b, target_a], dim=0
        )

    def step(model: torch.nn.Module, optim: torch.optim.Optimizer):
        batch, target = data()
        model.train()
        optim.zero_grad()

        outputs = model(batch)
        loss = torch.norm(torch.mean(outputs, dim=-1) - target)
        loss.backward()

        # Clip grad and error out if we're producing NaNs, part of the unit test
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 10.0, norm_type=2.0, error_if_nonfinite=True
        )
        optim.step()

        return loss.item()

    def evaluate(model: torch.nn.Module):
        batch, target = data()
        model.eval()
        outputs = model(batch)
        return torch.norm(torch.mean(outputs, dim=-1) - target).item()

    # Build both a reversible and non-reversible model
    model_non_reversible = xFormer.from_config(
        xFormerConfig(_rev_config(config, False))
    ).to(device)
    model_reversible = xFormer.from_config(xFormerConfig(_rev_config(config, True))).to(
        device
    )

    optim_rev = torch.optim.SGD(model_reversible.parameters(), lr=1e-3, momentum=0.9)
    optim_non_rev = torch.optim.SGD(
        model_non_reversible.parameters(), lr=1e-3, momentum=0.9
    )

    # Check that both models can be trained to comparable results
    eval_start_rev = evaluate(model_reversible)
    eval_start_non_rev = evaluate(model_non_reversible)

    for i in range(100):
        print(i, " reversible: ", step(model_reversible, optim_rev))
        print(i, " non reversible: ", step(model_non_reversible, optim_non_rev))

    # Check that we can classify this dummy example
    # Arbitrary threshold
    eval_stop_rev = evaluate(model_reversible)
    eval_stop_non_rev = evaluate(model_non_reversible)
    if len(config) < 2:  # only check the encoder case
        train_ratio_rev = eval_start_rev / eval_stop_rev
        train_ratio_non_rev = eval_start_non_rev / eval_stop_non_rev

        # Assert that train ratio > 1 (we trained),
        # and reversible is not much worse than non-reversible (it's actually better on this dummy test)

        assert train_ratio_rev > 1
        assert train_ratio_non_rev > 1
        assert train_ratio_rev > train_ratio_non_rev
