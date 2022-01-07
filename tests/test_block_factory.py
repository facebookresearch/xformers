# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

# Automatically fetch all registered attentions and Feedforwards
from xformers.components import Activation
from xformers.components.attention import ATTENTION_REGISTRY
from xformers.components.feedforward import FEEDFORWARD_REGISTRY
from xformers.factory import (
    xFormerDecoderBlock,
    xFormerDecoderConfig,
    xFormerEncoderBlock,
    xFormerEncoderConfig,
)

BATCH = 20
SEQ = 128
MODEL = 96
DROPOUT = 0.5
GLOBAL_ATTENTION_RATIO = 0.1  # 10% of the tokens have a global view
DEVICES = [torch.device("cuda")]
VOCAB_SIZE = 32


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("activation", [a.value for a in Activation])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
@pytest.mark.parametrize("layer_norm_style", ["pre", "post"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("reversible", [True, False])
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires a CUDA device"
)
def test_xformer_encoder_block(
    attention_name: str,
    feedforward_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    activation: Activation,
    layer_norm_style: str,
    device: torch.device,
    reversible: bool,
):
    block_size = 16

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": False,
        "window_size": SEQ // 8 + 1,
        "seq_len": SEQ,
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
        "num_heads": heads,
        "dim_head": MODEL // heads,
        "layout": torch.eye(SEQ // block_size, SEQ // block_size, dtype=torch.long),
        "block_size": block_size,
        "num_rules": 2,  # Compositional Attention
    }

    multi_head_config = {
        "num_heads": heads,
        "dim_model": MODEL,
        "residual_dropout": residual_dropout,
        "attention": attention_config,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_model": MODEL,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {
        "name": "sine",
        "dim_model": MODEL,
        "seq_len": SEQ,
        "vocab_size": VOCAB_SIZE,
    }

    block_config = xFormerEncoderConfig(
        dim_model=MODEL,
        multi_head_config=multi_head_config,
        feedforward_config=feedforward_config,
        position_encoding_config=position_encoding_config,
        layer_norm_style=layer_norm_style,
        reversible=reversible,
    )

    # Test that the whole block can be instantiated
    block = xFormerEncoderBlock.from_config(block_config).to(device)

    # Check that the dimensions make sense, to a FW pass
    inputs = torch.rand(BATCH, SEQ, device=device)
    _ = block(inputs)

    # Check that we support attention masking, at least interface wise (do not check correctness yet)
    att_mask = torch.ones(SEQ, SEQ, dtype=torch.bool, device=device)
    _ = block(inputs, att_mask=att_mask)

    # Check that we support input masking, at least interface wise (do not check correctness yet)
    input_mask = torch.randn(SEQ, dtype=torch.float, device=device)
    input_mask[input_mask < 0.0] = -float("inf")
    _ = block(inputs, input_mask=input_mask)


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("activation", [a.value for a in Activation])
@pytest.mark.parametrize("rotary_embeddings", [False, True])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("feedforward_name", FEEDFORWARD_REGISTRY.keys())
@pytest.mark.parametrize("layer_norm_style", ["pre", "post"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="This test requires a CUDA device"
)
def test_xformer_decoder_block(
    attention_name: str,
    rotary_embeddings: bool,
    feedforward_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    activation: Activation,
    layer_norm_style: str,
    device: torch.device,
):

    block_size = 16

    attention_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "window_size": SEQ // 8 + 1,
        "seq_len": SEQ,
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
        "num_heads": heads,
        "dim_head": MODEL / heads,
        "layout": torch.eye(SEQ // block_size, SEQ // block_size, dtype=torch.long),
        "block_size": block_size,
        "num_rules": 2,  # Compositional Attention
    }

    multi_head_config = {
        "num_heads": heads,
        "dim_model": MODEL,
        "residual_dropout": residual_dropout,
        "attention": attention_config,
        "use_rotary_embeddings": rotary_embeddings,
    }

    feedforward_config = {
        "name": feedforward_name,
        "dim_model": MODEL,
        "dropout": DROPOUT,
        "activation": activation,
        "hidden_layer_multiplier": 4,
    }

    position_encoding_config = {
        "name": "sine",
        "dim_model": MODEL,
        "seq_len": SEQ,
        "vocab_size": VOCAB_SIZE,
    }

    encoder_block_config = xFormerEncoderConfig(
        dim_model=MODEL,
        multi_head_config=multi_head_config,
        feedforward_config=feedforward_config,
        position_encoding_config=position_encoding_config,
        layer_norm_style=layer_norm_style,
    )

    decoder_block_config = xFormerDecoderConfig(
        dim_model=MODEL,
        multi_head_config_masked=multi_head_config,
        multi_head_config_cross=multi_head_config,
        feedforward_config=feedforward_config,
        position_encoding_config=position_encoding_config,
        layer_norm_style=layer_norm_style,
    )

    # Test that the whole block can be instantiated
    encoder_block = xFormerEncoderBlock.from_config(encoder_block_config).to(device)
    decoder_block = xFormerDecoderBlock.from_config(decoder_block_config).to(device)

    # Check that the dimensions make sense, to a FW pass
    inputs = torch.rand(BATCH, SEQ, device=device)
    encoded = encoder_block(inputs)
    _ = decoder_block(
        inputs, encoded
    )  # NOTE: does not make a lot of sense, just checking dimensions

    # Check that we support masking, at least interface wise (do not check correctness yet)
    att_mask = torch.ones(SEQ, SEQ, dtype=torch.bool, device=device)
    input_mask = torch.randn(SEQ, dtype=torch.float, device=device)
    input_mask[input_mask < 0.0] = -float("inf")

    encoded = encoder_block(inputs)
    _ = decoder_block(inputs, encoded, encoder_att_mask=att_mask, input_mask=input_mask)

    # Test different sequence lengths when encoding and decoding
    if not decoder_block.mha.attention.requires_same_k_q_dimensions:
        if not causal or not hasattr(decoder_block.mha.attention, "causal"):
            _ = decoder_block(inputs[:, :-16], encoded)
        else:
            # Check that we assert properly
            with pytest.raises(AssertionError):
                _ = decoder_block(inputs[:, :-16], encoded)
    else:
        # Check that we assert properly
        with pytest.raises(AssertionError):
            _ = decoder_block(inputs[:, :-16], encoded)
