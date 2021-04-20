import pytest
import torch

from xformers.components import MultiHeadDispatch

# Automatically test all the registered attentions
from xformers.components.attention import (
    _SPARSITY_THRESHOLD,
    ATTENTION_REGISTRY,
    AttentionConfig,
    build_attention,
)

BATCH = 5
SEQ = 1024
MODEL = 384
GLOBAL_ATTENTION_RATIO = (
    _SPARSITY_THRESHOLD * 0.9
)  # Make sure that we test the sparse implementation, no matter the threshold

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
        "dropout": attn_dropout,
        "causal": causal,
        "from_seq_dim": SEQ,
        "window_size": SEQ // 8 + 1,
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
    }

    attention = build_attention(AttentionConfig(**test_config))

    # build a multi head dispatch to test this attention mechanism
    multi_head = MultiHeadDispatch(
        from_seq_dim=SEQ,
        dim_model=MODEL,
        residual_dropout=residual_dropout,
        n_heads=heads,
        attention=attention,
    )

    # Check that a shuffled input produces the same results
    inputs = torch.rand(BATCH, SEQ, MODEL)
    shuffle = torch.randperm(inputs.shape[1])
    inputs_shuffled = inputs[:, shuffle, :]

    results = multi_head(inputs, inputs, inputs)
    results_shuffled = multi_head(inputs_shuffled, inputs_shuffled, inputs_shuffled)

    torch.allclose(results[:, shuffle, :], results_shuffled)


# TODO: way more unit tests..
