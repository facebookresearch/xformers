import pytest
import torch

from xformers.components import MultiHeadDispatch

# Automatically test all the registered attentions
from xformers.components.attention import (
    _DENSITY_THRESHOLD,
    ATTENTION_REGISTRY,
    build_attention,
)

DEVICES = (
    [torch.device("cpu")] if not torch.cuda.is_available() else [torch.device("cuda")]
)

BATCH = 5
SEQ = 128
MODEL = 96
GLOBAL_ATTENTION_RATIO = (
    _DENSITY_THRESHOLD * 0.9
)  # Make sure that we test the sparse implementation, no matter the threshold

assert ATTENTION_REGISTRY.keys(), "Attention layers should have been registered"


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("device", DEVICES)
def test_order_invariance(
    attention_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    device: torch.device,
):

    test_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "seq_len": SEQ,
        "window_size": SEQ // 8 + 1,
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
        "num_heads": heads,
        "dim_head": MODEL / heads,
    }

    attention = build_attention(test_config)

    # build a multi head dispatch to test this attention mechanism
    multi_head = MultiHeadDispatch(
        seq_len=SEQ,
        dim_model=MODEL,
        residual_dropout=residual_dropout,
        num_heads=heads,
        attention=attention,
    ).to(device)

    # Check that a shuffled input produces the same results
    inputs = torch.rand(BATCH, SEQ, MODEL, device=device)
    shuffle = torch.randperm(inputs.shape[1])
    inputs_shuffled = inputs[:, shuffle, :]

    results = multi_head(inputs, inputs, inputs)
    results_shuffled = multi_head(inputs_shuffled, inputs_shuffled, inputs_shuffled)

    torch.allclose(results[:, shuffle, :], results_shuffled)


# TODO: way more unit tests..
