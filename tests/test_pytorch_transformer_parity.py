import random
import time

import pytest
import torch

from xformers.factory.model_factory import xFormer, xFormerConfig

BATCH = 20
SEQ = 32
EMB = 8
VOCAB = 8
HEADS = 4
DROP = 0.1
LAYERS = 2
ACTIVATION = "relu"

_devices = (
    [torch.device("cpu")]
    if not torch.cuda.is_available()
    else [torch.device("cuda")]  # save a bit on CI, we have seperate cpu and gpu jobs
)

_test_config_encoder = {
    "block_config": {
        "block_type": "encoder",
        "dim_model": EMB,
        "num_layers": LAYERS,
        "layer_norm_style": "post",
        "multi_head_config": {
            "num_heads": HEADS,
            "residual_dropout": DROP,
            "use_separate_proj_weight": False,
            "bias": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": DROP,
                "causal": False,
                "seq_len": SEQ,
            },
            "dim_model": EMB,
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": DROP,
            "activation": ACTIVATION,
            "hidden_layer_multiplier": 4,
            "dim_model": EMB,
        },
    },
}


_test_config_decoder = {
    "block_config": {
        "block_type": "decoder",
        "dim_model": EMB,
        "num_layers": LAYERS,
        "layer_norm_style": "post",
        "multi_head_config_masked": {
            "num_heads": HEADS,
            "residual_dropout": DROP,
            "dim_model": EMB,
            "use_separate_proj_weight": False,
            "bias": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": DROP,
                "causal": False,
                "seq_len": SEQ,
            },
        },
        "multi_head_config_cross": {
            "num_heads": HEADS,
            "residual_dropout": DROP,
            "dim_model": EMB,
            "use_separate_proj_weight": False,
            "bias": True,
            "attention": {
                "name": "scaled_dot_product",
                "dropout": DROP,
                "causal": False,
                "seq_len": SEQ,
            },
        },
        "feedforward_config": {
            "name": "MLP",
            "dropout": DROP,
            "activation": ACTIVATION,
            "hidden_layer_multiplier": 4,
            "dim_model": EMB,
        },
    }
}

_test_config = [_test_config_encoder, _test_config_decoder]


def _data(device):
    # The dummy task is basically to classify sequences, either pure zeroes or some noise
    input_a = torch.zeros((BATCH, SEQ, EMB), device=device)
    input_b = (torch.rand((BATCH, SEQ, EMB), device=device) * VOCAB).abs()

    target_a = torch.zeros((BATCH, SEQ), device=device)
    target_b = torch.ones((BATCH, SEQ), device=device)

    if random.random() > 0.5:
        return torch.cat([input_a, input_b], dim=0), torch.cat(
            [target_a, target_b], dim=0
        )

    return torch.cat([input_b, input_a], dim=0), torch.cat([target_b, target_a], dim=0)


def reset_seeds():
    torch.manual_seed(0)
    random.seed(0)


def step(model: torch.nn.Module, optim: torch.optim.Optimizer, device):
    model.train()
    optim.zero_grad()
    batch, target = _data(device)

    try:
        outputs = model(batch)
    except TypeError:
        # Pytorch decoder exposes target explicitly
        outputs = model(batch, tgt=batch)

    loss = torch.norm(torch.mean(outputs, dim=-1) - target)
    loss.backward()

    # Clip grad and error out if we're producing NaNs, part of the unit test
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 10.0, norm_type=2.0, error_if_nonfinite=True
    )
    optim.step()

    return loss.item()


def evaluate(model: torch.nn.Module, device):
    batch, target = _data(device)
    model.eval()
    try:
        outputs = model(batch)
    except TypeError:
        # Pytorch decoder exposes target explicitly
        outputs = model(batch, tgt=batch)

    return torch.norm(torch.mean(outputs, dim=-1) - target).item()


def train(model, optimizer, name, steps, device):
    # Dummy training, just checking that both options give the same results
    # Same seed for everyone
    reset_seeds()
    start = time.time()
    for i in range(steps):
        loss = step(model, optimizer, device)
        print(i, name, loss)

    print("Trained {} in {:.3}s".format(name, time.time() - start))


@pytest.mark.parametrize("device", _devices)
def test_pytorch_encoder_parity(device):
    # Build both a xFormers and Pytorch model
    reset_seeds()
    model_xformers = xFormer.from_config(xFormerConfig([_test_config_encoder])).to(
        device
    )
    print(model_xformers)

    model_pytorch = torch.nn.TransformerEncoder(
        torch.nn.TransformerEncoderLayer(
            d_model=EMB,
            nhead=HEADS,
            dim_feedforward=4 * EMB,
            dropout=DROP,
            activation=ACTIVATION,
            layer_norm_eps=1e-05,
            batch_first=True,  # (batch, seq, feature)
            device=device,
        ),
        num_layers=LAYERS,
    )
    print(model_pytorch)

    optim_xformers = torch.optim.SGD(model_xformers.parameters(), lr=1e-3, momentum=0.9)
    optim_pytorch = torch.optim.SGD(model_pytorch.parameters(), lr=1e-3, momentum=0.9)

    # Check that both models can be trained to comparable results
    eval_start_xformer = evaluate(model_xformers, device)
    eval_start_pytorch = evaluate(model_pytorch, device)
    print("starting point: ", eval_start_pytorch, eval_start_xformer)
    train(model_pytorch, optim_pytorch, "pytorch", 500, device)
    train(model_xformers, optim_xformers, "xformers", 500, device)

    # Check that we can classify this dummy example
    # Arbitrary threshold
    eval_stop_xformer = evaluate(model_xformers, device)
    eval_stop_pytorch = evaluate(model_pytorch, device)
    print("end point: ", eval_stop_pytorch, eval_stop_xformer)

    fit_ratio_xformer = eval_start_xformer / eval_stop_xformer
    fit_ratio_pytorch = eval_start_pytorch / eval_stop_pytorch

    print(fit_ratio_pytorch, fit_ratio_xformer)

    # Catch a broken training
    assert fit_ratio_xformer > 60
    assert fit_ratio_pytorch > 60

    # Catch a significant difference in between the two
    assert (
        abs(eval_start_xformer - eval_start_pytorch) < 1e-1
    )  # initial eval is about 50, arbitrary limits
    assert (
        abs(eval_stop_xformer - eval_stop_pytorch) < 1e-1
    )  # final eval is about 0.74, arbitrary limits


@pytest.mark.parametrize("device", _devices)
def test_pytorch_tranformer_parity(device):
    # Build both a xFormers and Pytorch model
    reset_seeds()
    model_xformers = xFormer.from_config(xFormerConfig(_test_config)).to(device)
    print(model_xformers)

    model_pytorch = torch.nn.Transformer(
        d_model=EMB,
        nhead=HEADS,
        num_encoder_layers=LAYERS,
        num_decoder_layers=LAYERS,
        dim_feedforward=4 * EMB,
        dropout=DROP,
        activation=ACTIVATION,
        layer_norm_eps=1e-05,
        batch_first=True,  # (batch, seq, feature)
        device=device,
    )
    print(model_pytorch)

    optim_xformers = torch.optim.SGD(model_xformers.parameters(), lr=1e-3, momentum=0.9)
    optim_pytorch = torch.optim.SGD(model_pytorch.parameters(), lr=1e-3, momentum=0.9)

    # Check that both models can be trained to comparable results
    eval_start_xformer = evaluate(model_xformers, device)
    eval_start_pytorch = evaluate(model_pytorch, device)
    print("starting point: ", eval_start_pytorch, eval_start_xformer)
    train(model_xformers, optim_xformers, "xformers", 100, device)
    train(model_pytorch, optim_pytorch, "pytorch", 100, device)

    # Check that we can classify this dummy example
    # Arbitrary threshold
    eval_stop_xformer = evaluate(model_xformers, device)
    eval_stop_pytorch = evaluate(model_pytorch, device)
    print("end point: ", eval_stop_pytorch, eval_stop_xformer)

    fit_ratio_xformer = eval_start_xformer / eval_stop_xformer
    fit_ratio_pytorch = eval_start_pytorch / eval_stop_pytorch

    print(fit_ratio_pytorch, fit_ratio_xformer)

    # FIXME: Should not have a discrenpancy here.
    assert fit_ratio_xformer > 30
    assert fit_ratio_pytorch > 30
