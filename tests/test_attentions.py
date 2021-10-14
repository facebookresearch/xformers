# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

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

BATCH = 2
SEQ = 128 if torch.cuda.is_available() else 32
MODEL = 128 if torch.cuda.is_available() else 64
GLOBAL_ATTENTION_RATIO = (
    _DENSITY_THRESHOLD * 0.9
)  # Make sure that we test the sparse implementation, no matter the threshold

assert ATTENTION_REGISTRY.keys(), "Attention layers should have been registered"


def _get_multihead(attention_name, attn_dropout, res_dropout, causal, heads, device):
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

    # Add some blocksparse layout to test the corresponding attention
    block_size = 16
    test_config["layout"] = torch.eye(
        SEQ // block_size, SEQ // block_size, dtype=torch.long
    )
    test_config["block_size"] = block_size

    attention = build_attention(test_config)

    # build a multi head dispatch to test this attention mechanism
    multi_head = MultiHeadDispatch(
        seq_len=SEQ,
        dim_model=MODEL,
        residual_dropout=res_dropout,
        num_heads=heads,
        attention=attention,
    ).to(device)

    return multi_head


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 4])
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
    multi_head = _get_multihead(
        attention_name, attn_dropout, residual_dropout, causal, heads, device
    )

    # Check that a shuffled input produces the same results
    inputs = torch.rand(BATCH, SEQ, MODEL, device=device)
    shuffle = torch.randperm(inputs.shape[1])
    inputs_shuffled = inputs[:, shuffle, :].clone()

    results = multi_head(inputs, inputs, inputs)
    results_shuffled = multi_head(inputs_shuffled, inputs_shuffled, inputs_shuffled)

    torch.allclose(results[:, shuffle, :], results_shuffled)

    # Test the non-self-attention codepath
    _ = multi_head(inputs, inputs_shuffled, inputs)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("attention_name", ["scaled_dot_product"])
@pytest.mark.parametrize("device", DEVICES)
def test_kqv_ordering(
    attention_name: str,
    heads: int,
    device: torch.device,
):

    multi_head = _get_multihead(attention_name, 0.0, 0.0, False, heads, device)

    # Check kqv are not flipped
    # this will not catch all issues, but would catch a V being misplaced
    # make k and q complimentary, so that QKt is all zero and attention is uniform

    q = torch.cat(
        (
            torch.rand((1, MODEL // 2), device=device),
            torch.zeros((1, MODEL // 2), device=device),
        ),
        dim=1,
    ).expand((BATCH, SEQ, MODEL))

    k = torch.cat(
        (
            torch.zeros((1, MODEL // 2), device=device),
            torch.rand((1, MODEL // 2), device=device),
        ),
        dim=1,
    ).expand((BATCH, SEQ, MODEL))
    v = torch.rand(BATCH, SEQ, MODEL, device=device)

    # Normal call
    res = multi_head(query=q, key=k, value=v)
    for i in range(BATCH):
        assert torch.allclose(res[i, :, :], res[i, 0, :].unsqueeze(-2))

    assert not torch.allclose(res[0, :, :], res[1, :, :])

    # Flip qkv, and check that we invert the above check properly
    res_false = multi_head(query=v, key=k, value=q)
    assert torch.allclose(res_false[0, :, :], res_false[1, :, :])


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("device", DEVICES)
def test_different_kq_dimensions(
    attention_name: str,
    heads: int,
    device: torch.device,
):
    if attention_name in {
        "global",
        "local",
        "random",
        "lambda",
        "linformer",
        "blocksparse",
    }:
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip(f"{attention_name} does not support different k, q dimensions yet.")
    multi_head = _get_multihead(attention_name, 0.0, 0.0, False, heads, device)

    seq_q = SEQ - 16
    q = torch.rand((BATCH, seq_q, MODEL), device=device)
    k = torch.rand((BATCH, SEQ, MODEL), device=device)
    v = torch.rand((BATCH, SEQ, MODEL), device=device)

    res = multi_head(query=q, key=k, value=v)
    assert res.shape == torch.Size([BATCH, seq_q, MODEL])


# TODO: way more unit tests..
