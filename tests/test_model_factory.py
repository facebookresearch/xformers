import pytest
import torch

from xformers.factory.model_factory import xFormer, xFormerConfig

BATCH = 20
SEQ = 512


test_configs = [
    {
        "block_configs": [
            {
                "block_type": "encoder",
                "dim_model": 384,
                "position_encoding_config": {
                    "name": "vocab",
                    "dim_model": 384,
                    "max_sequence_len": SEQ,
                    "vocab_size": 64,
                },
                "num_layers": 3,
                "attention_config": {
                    "name": "linformer",
                    "dropout": 0,
                    "causal": True,
                    "from_seq_dim": 512,
                },
                "multi_head_config": {
                    "n_heads": 4,
                    "from_seq_dim": 512,
                    "dim_model": 384,
                    "residual_dropout": 0,
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dim_latent": 384,
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
                    "max_sequence_len": SEQ,
                    "vocab_size": 64,
                },
                "num_layers": 2,
                "attention_configs": [
                    {
                        "name": "linformer",
                        "dropout": 0,
                        "causal": True,
                        "from_seq_dim": 512,
                    },
                    {
                        "name": "linformer",
                        "dropout": 0,
                        "causal": False,
                        "from_seq_dim": 512,
                    },
                ],
                "multi_head_configs": [
                    {
                        "n_heads": 4,
                        "from_seq_dim": 512,
                        "dim_model": 384,
                        "residual_dropout": 0,
                    },
                    {
                        "n_heads": 4,
                        "from_seq_dim": 512,
                        "dim_model": 384,
                        "residual_dropout": 0,
                    },
                ],
                "feedforward_config": {
                    "name": "MLP",
                    "dim_latent": 384,
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
def test_presets(config):
    # Build the model
    model = xFormer.from_config(xFormerConfig(**config))

    # Dummy inputs, test a forward
    inputs = (torch.rand(BATCH, SEQ) * 10).abs().to(torch.int)
    _ = model(inputs)
