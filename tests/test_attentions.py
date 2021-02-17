import pytest
import torch

from xformers.components.attention import Attention
from xformers.components.attention.multi_head_attention import MultiHeadAttention

BATCH = 20
SEQ = 512
EMBD = 384

attentions = [MultiHeadAttention]  # TODO: list these automatically


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("attention_class", attentions)
def test_order_invariance(
    attention_class: Attention,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
):
    test_config = {
        "n_heads": heads,
        "dim_embd": EMBD,
        "dim_key": 64,
        "dim_value": 64,
        "attention_dropout": attn_dropout,
        "residual_dropout": residual_dropout,
        "causal": causal,
    }

    attention = attention_class(**test_config)

    # Check that a shuffled input produces the same results
    inputs = torch.rand(BATCH, SEQ, EMBD)
    shuffle = torch.randperm(inputs.shape[1])
    inputs_shuffled = inputs[:, shuffle, :]

    results = attention(inputs)
    results_shuffled = attention(inputs_shuffled)

    torch.allclose(results[:, shuffle, :], results_shuffled)


# TODO: test loading from config
# TODO: way more unit tests..
