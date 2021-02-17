import pytest

from xformers.components.feedforward import Activations
from xformers.factory import (
    AttentionConfig,
    FeedforwardConfig,
    PositionEncodingConfig,
    xFormerBlock,
    xFormerConfig,
)

BATCH = 20
SEQ = 512
EMBD = 384
LATENT = 128
DROPOUT = 0.5


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("activation", [a.value for a in Activations])
def test_xformer_block(
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    activation: Activations,
):

    attention_config = {
        "n_heads": heads,
        "dim_embd": EMBD,
        "dim_key": 64,
        "dim_value": 64,
        "attention_dropout": attn_dropout,
        "residual_dropout": residual_dropout,
        "causal": causal,
    }

    feedforward_config = {
        "dim_latent": LATENT,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {"dim_embd": EMBD, "seq_len": SEQ}

    block_config = xFormerConfig(
        EMBD,
        AttentionConfig(**attention_config),
        FeedforwardConfig(**feedforward_config),
        PositionEncodingConfig(**position_encoding_config),
    )

    _ = xFormerBlock.from_config(block_config)
