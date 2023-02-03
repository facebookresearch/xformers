# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Tuple

import pytest
import torch

from xformers.components import (
    InputProjection,
    InputProjectionConfig,
    MultiHeadDispatch,
)

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
SEQ = 128 if torch.cuda.is_available() else 36
MODEL = 128 if torch.cuda.is_available() else 16
GLOBAL_ATTENTION_RATIO = (
    _DENSITY_THRESHOLD * 0.9
)  # Make sure that we test the sparse implementation, no matter the threshold

assert ATTENTION_REGISTRY.keys(), "Attention layers should have been registered"

_non_order_invariant_attentions = ["visual", "pooling"]


def _get_multihead(
    attention_name,
    attn_dropout,
    res_dropout,
    causal,
    heads,
    device,
    skip_output_projection=False,
    use_seperate_proj_weights=True,
):
    test_config = {
        "name": attention_name,
        "dropout": attn_dropout,
        "causal": causal,
        "seq_len": SEQ,
        "window_size": SEQ // 8 + 1,  # local attention
        "attention_query_mask": torch.rand((SEQ, 1)) < GLOBAL_ATTENTION_RATIO,
        "dim_model": MODEL,
        "num_heads": heads,
        "dim_head": MODEL / heads,
        "num_rules": 2,  # Compositional Attention
        "r": 0.5,  # random attention, ratio of tokens that the attention can attend to
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
        use_separate_proj_weight=use_seperate_proj_weights,
    ).to(device)

    return multi_head


@pytest.mark.parametrize("attn_dropout", [0.0, 0.3])
@pytest.mark.parametrize("residual_dropout", [0.0, 0.1])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize(
    "attention_name", ATTENTION_REGISTRY.keys() - _non_order_invariant_attentions
)
@pytest.mark.parametrize("device", DEVICES)
def test_order_invariance(
    attention_name: str,
    heads: int,
    attn_dropout: float,
    residual_dropout: float,
    causal: bool,
    device: torch.device,
):

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    multi_head = _get_multihead(
        attention_name,
        attn_dropout,
        residual_dropout,
        causal,
        heads,
        device,
        use_seperate_proj_weights=False,
    )

    if (
        int(math.sqrt(SEQ)) ** 2 != SEQ
        and multi_head.attention.requires_squared_context
    ):
        pytest.skip(f"{attention_name} requires squared sequence lengths")

    # Check that we can pass a smaller sequence
    seqs = (
        [SEQ, SEQ // 2]
        if not multi_head.attention.requires_same_k_q_dimensions
        else [SEQ]
    )

    for seq in seqs:
        # Check that the attention is invariant to a permutation of K, V
        inputs = torch.rand(BATCH, seq, MODEL, device=device)
        shuffle = torch.randperm(inputs.shape[1])
        inputs_shuffled = inputs[:, shuffle, :].clone()

        results = multi_head(inputs, inputs, inputs)
        results_shuffled = multi_head(inputs, inputs_shuffled, inputs_shuffled)
        torch.allclose(results, results_shuffled)

        # Check that the attention is equivariant to a permutation of Q,
        # meaning that the result is permuted in the same way
        results_shuffled = multi_head(inputs_shuffled, inputs, inputs)
        torch.allclose(results[:, shuffle, :], results_shuffled)

        # Check that dropout actually drops some values
        if attn_dropout > 0:
            att_1 = multi_head(inputs, inputs_shuffled, inputs)
            att_2 = multi_head(inputs, inputs_shuffled, inputs)
            assert (att_1 != att_2).any()

        # Test AMP, if available
        if device.type == "cuda":
            with torch.cuda.amp.autocast(enabled=True):
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
@pytest.mark.parametrize("attention_name", ["scaled_dot_product"])
@pytest.mark.parametrize("device", DEVICES)
def test_different_seqlen(
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

    # Changing sequence length by dividing by two to simulate differing sequence length
    q2 = torch.cat(
        (
            torch.rand((1, MODEL // 2), device=device),
            torch.zeros((1, MODEL // 2), device=device),
        ),
        dim=1,
    ).expand((BATCH, SEQ // 2, MODEL))

    k2 = torch.cat(
        (
            torch.zeros((1, MODEL // 2), device=device),
            torch.rand((1, MODEL // 2), device=device),
        ),
        dim=1,
    ).expand((BATCH, SEQ // 2, MODEL))

    v2 = torch.rand(BATCH, SEQ // 2, MODEL, device=device)

    res2 = multi_head(query=q2, key=k2, value=v2)

    assert res.shape != res2.shape


@pytest.mark.parametrize("proj_bias", [False, True])
@pytest.mark.parametrize("same_sizes", [False, True])
@pytest.mark.parametrize("same_settings", [False, True])
def test_inproj(proj_bias: bool, same_sizes: bool, same_settings: bool):

    test_config = {
        "name": "scaled_dot_product",
        "dropout": 0.1,
        "causal": False,
        "seq_len": SEQ,
        "window_size": SEQ // 8 + 1,
        "num_heads": 1,
        "dim_head": MODEL,
    }

    attention = build_attention(test_config)

    # Construct the initial projection, test different options
    in_params = InputProjectionConfig(MODEL, MODEL, proj_bias)

    if same_settings:
        in_proj = InputProjection(in_params, None, None)
        out_features = MODEL
    else:
        out_features = MODEL if same_sizes else MODEL // 2
        in_params_flip = InputProjectionConfig(MODEL, out_features, proj_bias)
        in_proj = InputProjection(
            in_params_flip,  # Q proj
            in_params_flip,  # K proj
            in_params,  # V proj
        )

    # build a multi head dispatch to test this attention mechanism
    multi_head = MultiHeadDispatch(
        seq_len=SEQ,
        dim_model=MODEL,
        residual_dropout=0.1,
        num_heads=1,
        attention=attention,
        in_proj_container=in_proj,
        dim_key=out_features,
        dim_value=MODEL,
    )

    # Check kqv are not flipped
    # this will not catch all issues, but would catch a V being misplaced
    # make k and q complimentary, so that QKt is all zero and attention is uniform

    q = torch.cat(
        (
            torch.rand((1, MODEL // 2)),
            torch.zeros((1, MODEL // 2)),
        ),
        dim=1,
    ).expand((BATCH, SEQ, MODEL))

    k = torch.cat(
        (
            torch.zeros((1, MODEL // 2)),
            torch.rand((1, MODEL // 2)),
        ),
        dim=1,
    ).expand((BATCH, SEQ, MODEL))
    v = torch.rand(BATCH, SEQ, MODEL)

    # just check that a FW does not assert out
    _ = multi_head(query=q, key=k, value=v)


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("device", DEVICES)
def test_different_kq_dimensions(
    attention_name: str,
    heads: int,
    device: torch.device,
):

    multi_head = _get_multihead(attention_name, 0.0, 0.0, False, heads, device)

    if multi_head.attention.requires_same_k_q_dimensions:
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip(f"{attention_name} does not support different k, q dimensions yet.")

    seq_q = SEQ // 2
    q = torch.rand((BATCH, seq_q, MODEL), device=device)
    k = torch.rand((BATCH, SEQ, MODEL), device=device)
    v = torch.rand((BATCH, SEQ, MODEL), device=device)

    res = multi_head(query=q, key=k, value=v)
    assert res.shape == torch.Size([BATCH, seq_q, MODEL])


@pytest.mark.parametrize("heads", [1, 4])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "batch_sizes",
    [
        (1, BATCH, BATCH),
        (BATCH, 1, BATCH),
        (BATCH, BATCH, 1),
        (1, 1, BATCH),
        (BATCH, 1, 1),
        (1, BATCH, 1),
    ],
)
def test_broadcast_batch_dimension(
    attention_name: str,
    heads: int,
    device: torch.device,
    batch_sizes: Tuple[int, int, int],
):
    Q_BATCH, K_BATCH, V_BATCH = batch_sizes
    multi_head = _get_multihead(attention_name, 0.0, 0.0, False, heads, device)

    if (
        int(math.sqrt(SEQ)) ** 2 != SEQ
        and multi_head.attention.requires_squared_context
    ):
        pytest.skip(f"{attention_name} requires squared sequence lengths")

    if multi_head.attention.requires_same_k_q_dimensions:
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip(f"{attention_name} does not support different k, q dimensions yet.")

    q = torch.rand((Q_BATCH, SEQ, MODEL), device=device)
    k = torch.rand((K_BATCH, SEQ, MODEL), device=device)
    v = torch.rand((V_BATCH, SEQ, MODEL), device=device)

    res = multi_head(query=q, key=k, value=v)
    assert res.shape == torch.Size([BATCH, SEQ, MODEL])


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


@pytest.mark.parametrize("attn_dropout", [0.0, 0.1])
@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize("attention_name", ATTENTION_REGISTRY.keys())
@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA gpu not supported yet")
def test_torch_script_ability(
    attention_name: str,
    heads: int,
    attn_dropout: float,
):
    if attention_name in {"favor", "global", "local", "random"}:
        # pyre-fixme[29]: The library function `pytest.skip` is not supported by Pyre.
        pytest.skip(f"{attention_name} does not support scripting yet.")

    device = torch.device("cpu")

    multi_head = _get_multihead(attention_name, attn_dropout, 0.0, False, heads, device)

    if (
        int(math.sqrt(SEQ)) ** 2 != SEQ
        and multi_head.attention.requires_squared_context
    ):
        pytest.skip(f"{attention_name} requires squared sequence lengths")

    # input for tracing the function
    q = torch.rand((BATCH, SEQ, MODEL), device=device)
    k = torch.rand((BATCH, SEQ, MODEL), device=device)
    v = torch.rand((BATCH, SEQ, MODEL), device=device)

    # to make sure dropout behaves deterministically
    torch.random.manual_seed(42)
    # tracing the attention module
    traced_multi_head = torch.jit.trace(multi_head, (q, k, v))

    # create new random inputs for testing the eager model and traced model
    q = torch.rand((BATCH, SEQ, MODEL), device=device)
    k = torch.rand((BATCH, SEQ, MODEL), device=device)
    v = torch.rand((BATCH, SEQ, MODEL), device=device)

    # to make sure dropout behaves deterministically need to set the seed again
    torch.random.manual_seed(42)
    res = multi_head(query=q, key=k, value=v)

    # to make sure dropout behaves deterministically need to set the seed again
    torch.random.manual_seed(42)
    res_traced = traced_multi_head(query=q, key=k, value=v)

    assert torch.allclose(res, res_traced)


# TODO: way more unit tests..
