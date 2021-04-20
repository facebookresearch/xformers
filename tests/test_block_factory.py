import pytest
import torch

# Automatically fetch all registered attentions and Feedforwards
from xformers.components import Activation
from xformers.components.attention import ATTENTION_REGISTRY
from xformers.components.feedforward import FEEDFORWARD_REGISTRY
from xformers.factory import (
    AttentionConfig,
    FeedforwardConfig,
    MultiHeadDispatchConfig,
    PositionEncodingConfig,
    xFormerDecoderBlock,
    xFormerDecoderConfig,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
)

BATCH = 20
SEQ = 512
MODEL = 384
DROPOUT = 0.5
GLOBAL_ATTENTION_RATIO = 0.1  # 10% of the tokens have a global view


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("activation", [a.value for a in Activation])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
def test_xformer_encoder_block(
    attention_name: str,
    feedforward_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    activation: Activation,
):

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "window_size": SEQ // 8 + 1,
        "from_seq_dim": SEQ,
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
    }

    multi_head_config = {
        "n_heads": heads,
        "from_seq_dim": SEQ,
        "dim_model": MODEL,
        "residual_dropout": residual_dropout,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_latent": MODEL,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {"name": "sine", "dim_model": MODEL, "seq_len": SEQ}

    block_config = xFormerEncoderConfig(
        dim_model=MODEL,
        attention_config=AttentionConfig(**attention_config),
        multi_head_config=MultiHeadDispatchConfig(**multi_head_config),
        feedforward_config=FeedforwardConfig(**feedforward_config),
        position_encoding_config=PositionEncodingConfig(**position_encoding_config),
    )

    # Test that the whole block can be instantiated
    block = xFormerEncoderBlock.from_config(block_config)

    # Check that the dimensions make sense, to a FW pass
    inputs = torch.rand(BATCH, SEQ, MODEL)
    _ = block(inputs)


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("activation", [a.value for a in Activation])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
def test_xformer_decoder_block(
    attention_name: str,
    feedforward_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    activation: Activation,
):

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "window_size": SEQ // 8 + 1,
        "from_seq_dim": SEQ,
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
    }

    multi_head_config = {
        "n_heads": heads,
        "from_seq_dim": SEQ,
        "dim_model": MODEL,
        "residual_dropout": residual_dropout,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_latent": MODEL,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {"name": "sine", "dim_model": MODEL, "seq_len": SEQ}

    encoder_block_config = xFormerEncoderConfig(
        dim_model=MODEL,
        attention_config=AttentionConfig(**attention_config),
        multi_head_config=MultiHeadDispatchConfig(**multi_head_config),
        feedforward_config=FeedforwardConfig(**feedforward_config),
        position_encoding_config=PositionEncodingConfig(**position_encoding_config),
    )

    decoder_block_config = xFormerDecoderConfig(
        dim_model=MODEL,
        attention_configs=(
            AttentionConfig(**attention_config),
            AttentionConfig(**attention_config),
        ),
        multi_head_configs=(
            MultiHeadDispatchConfig(**multi_head_config),
            MultiHeadDispatchConfig(**multi_head_config),
        ),
        feedforward_config=FeedforwardConfig(**feedforward_config),
        position_encoding_config=PositionEncodingConfig(**position_encoding_config),
    )

    # Test that the whole block can be instantiated
    encoder_block = xFormerEncoderBlock.from_config(encoder_block_config)
    decoder_block = xFormerDecoderBlock.from_config(decoder_block_config)

    # Check that the dimensions make sense, to a FW pass
    inputs = torch.rand(BATCH, SEQ, MODEL)
    encoded = encoder_block(inputs)
    _ = decoder_block(
        inputs, encoded
    )  # FIXME: does not make a lot of sense, just checking dimensions
