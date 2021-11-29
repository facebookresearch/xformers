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


def _get_multihead(
    attention_name,
    attn_dropout,
    res_dropout,
    causal,
    heads,
    device,
    skip_output_projection=False,
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

    if skip_output_projection:

        def noop(x):
            return x

        test_config["out_proj"] = noop

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
    seqs = [SEQ, SEQ - 16] if (attention_name != "blocksparse") else [SEQ]

    for seq in seqs:
        # Check that we can pass a smaller sequence
        inputs = torch.rand(BATCH, seq, MODEL, device=device)
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


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("attention_name", ["scaled_dot_product", "favor"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA gpu")
def test_causal(
    attention_name: str,
    heads: int,
):
    """
    Make sure that the causal flag is respected.
    The input data is orthogonal by design if causal is respected, but if the attention looks ahead this will fail
    """

    torch.random.manual_seed(42)

    device = torch.device("cuda")

    multi_head = _get_multihead(
        attention_name,
        0.0,
        0.0,
        causal=True,
        heads=heads,
        device=device,
        skip_output_projection=True,
    )

    k = (
        torch.tril(torch.ones((SEQ, SEQ), device=device), diagonal=0)
        .unsqueeze(0)
        .expand(1, -1, -1)
    )
    q = (
        torch.triu(torch.ones((SEQ, SEQ), device=device), diagonal=0)
        .unsqueeze(0)
        .expand(1, -1, -1)
    )
    v = (
        torch.arange(SEQ, device=device)
        .float()
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(1, -1, SEQ)
    )

    # Make sure that we don´t project, to keep the embeddings orthogonal
    multi_head.attention.requires_input_projection = False

    res = multi_head(query=q, key=k, value=v).squeeze(0)

    # Consolidate along the embedding, if causal was respected the amplitude should be sorted already
    res_sum = torch.sum(res, dim=1).cpu()

    assert torch.allclose(torch.sort(res_sum)[1], torch.arange(SEQ)) or torch.allclose(
        torch.sort(res_sum, descending=True)[1], torch.arange(SEQ)
    ), res_sum


@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("device", DEVICES)
def test_torch_script_ability(
    attention_name: str,
    heads: int,
    device: torch.device,
):
    if attention_name in {
        "favor",
        "global",
        "local",
        "random",
    }:
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip(f"{attention_name} does not support scripting yet.")
    multi_head = _get_multihead(attention_name, 0.0, 0.0, False, heads, device)

    seq_q = SEQ - 16

    # input for tracing
    q = torch.rand((BATCH, seq_q, MODEL), device=device)
    k = torch.rand((BATCH, seq_q, MODEL), device=device)
    v = torch.rand((BATCH, seq_q, MODEL), device=device)

    # tracing the attention module
    traced_multi_head = torch.jit.trace(multi_head, (q, k, v))

    # create new random inputs for testing the eager model and traced model
    q = torch.rand((BATCH, seq_q, MODEL), device=device)
    k = torch.rand((BATCH, seq_q, MODEL), device=device)
    v = torch.rand((BATCH, seq_q, MODEL), device=device)

    res = multi_head(query=q, key=k, value=v)
    res_traced = traced_multi_head(query=q, key=k, value=v)

    assert torch.allclose(res, res_traced)


# TODO: way more unit tests..
