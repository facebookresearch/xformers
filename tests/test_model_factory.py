import pytest
import torch

from xformers.factory.model_factory import xFormer, xFormerConfig

BATCH = 20
SEQ = 512
DEVICES = (
    [torch.device("cpu")]
    if not torch.cuda.is_available()
    else [
        torch.device("cuda")
    ]  # save a bit on CI for now, we have seperate cpu and gpu jobs
)

test_configs = [
    {
        "block_configs": [
            {
                "block_type": "encoder",
                "dim_model": 384,
                "position_encoding_config": {
                    "name": "vocab",
                    "dim_model": 384,
                    "seq_len": SEQ,
                    "vocab_size": 64,
                },
                "num_layers": 3,
                "multi_head_config": {
                    "num_heads": 4,
                    "dim_model": 384,
                    "residual_dropout": 0,
                    "attention": {
                        "name": "linformer",
                        "dropout": 0,
                        "causal": True,
                        "seq_len": 512,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dim_model": 384,
                    "dropout": 0,
                    "activation": "relu",
                    "hidden_layer_multiplier": 4,
                },
            },
            {
                "block_type": "decoder",
                "dim_model": 384,
                "position_encoding_config": {
                    "name": "vocab",
                    "dim_model": 384,
                    "seq_len": SEQ,
                    "vocab_size": 64,
                },
                "num_layers": 2,
                "multi_head_config_pre_encoder": {
                    "num_heads": 4,
                    "dim_model": 384,
                    "residual_dropout": 0,
                    "attention": {
                        "name": "linformer",
                        "dropout": 0,
                        "causal": True,
                        "seq_len": 512,
                    },
                },
                "multi_head_config_post_encoder": {
                    "num_heads": 4,
                    "dim_model": 384,
                    "residual_dropout": 0,
                    "attention": {
                        "name": "linformer",
                        "dropout": 0,
                        "causal": True,
                        "seq_len": 512,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dim_model": 384,
                    "dropout": 0,
                    "activation": "relu",
                    "hidden_layer_multiplier": 4,
                },
            },
        ]
    }
]


""" Test all the model configurations saved in model_presets. """


@pytest.mark.parametrize("config", test_configs)
@pytest.mark.parametrize("device", DEVICES)
def test_presets(config, device):
    # Build the model
    model = xFormer.from_config(xFormerConfig(**config)).to(device)

    # Dummy inputs, test a forward
    inputs = (torch.rand(BATCH, SEQ, device=device) * 10).abs().to(torch.int)

    input_mask = torch.randn(SEQ, dtype=torch.float, device=device)
    input_mask[input_mask < 0.0] = -float("inf")
    _ = model(inputs, encoder_input_mask=input_mask, decoder_input_mask=input_mask)
