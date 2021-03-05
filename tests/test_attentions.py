import pytest
import torch

# Automatically test all the registered attentions
from xformers.components.attention import (
    ATTENTION_REGISTRY,
    AttentionConfig,
    build_attention,
)

BATCH = 20
SEQ = 1920
MODEL = 384

assert ATTENTION_REGISTRY.keys(), "Attention layers should have been registered"


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
def test_order_invariance(
    attention_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
):
    test_config = {
        "name": attention_name,
        "n_heads": heads,
        "dim_in": MODEL,
        "dim_out": MODEL,
        "attention_dropout": attn_dropout,
        "residual_dropout": residual_dropout,
        "causal": causal,
        "window_size": SEQ // 10,
    }

    attention = build_attention(AttentionConfig(**test_config))

    # Check that a shuffled input produces the same results
    inputs = torch.rand(BATCH, SEQ, MODEL)
    shuffle = torch.randperm(inputs.shape[1])
    inputs_shuffled = inputs[:, shuffle, :]

    results = attention(inputs)
    results_shuffled = attention(inputs_shuffled)

    torch.allclose(results[:, shuffle, :], results_shuffled)


# TODO: way more unit tests..
