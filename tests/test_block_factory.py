import pytest

from xformers.block_factory import (
    AttentionConfig,
    FeedforwardConfig,
    MultiHeadDispatchConfig,
    PositionEncodingConfig,
    xFormerBlock,
    xFormerConfig,
)

# Automatically fetch all registered attentions and Feedforwards
from xformers.components.attention import ATTENTION_REGISTRY
from xformers.components.feedforward import FEEDFORWARD_REGISTRY, Activations

BATCH = 20
SEQ = 1920
MODEL = 384
LATENT = 128
DROPOUT = 0.5


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("activation", [a.value for a in Activations])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
def test_xformer_block(
    attention_name: str,
    feedforward_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    activation: Activations,
):

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "window_size": SEQ // 10,
    }

    multi_head_config = {
        "n_heads": heads,
        "dim_in": MODEL,
        "dim_out": MODEL,
        "residual_dropout": residual_dropout,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_latent": LATENT,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {"name": "sine", "dim_model": MODEL, "seq_len": SEQ}

    block_config = xFormerConfig(
        MODEL,
        AttentionConfig(**attention_config),
        MultiHeadDispatchConfig(**multi_head_config),
        FeedforwardConfig(**feedforward_config),
        PositionEncodingConfig(**position_encoding_config),
    )

    _ = xFormerBlock.from_config(block_config)
