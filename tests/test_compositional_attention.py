# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from xformers.components import MultiHeadDispatch

# Automatically test all the registered attentions
from xformers.components.attention import ATTENTION_REGISTRY, build_attention

DEVICES = (
    [torch.device("cpu")] if not torch.cuda.is_available() else [torch.device("cuda")]
)

BATCH = 2
SEQ = 128 if torch.cuda.is_available() else 16
MODEL = 128 if torch.cuda.is_available() else 32

assert ATTENTION_REGISTRY.keys(), "Attention layers should have been registered"


@pytest.mark.parametrize("heads", [4])
@pytest.mark.parametrize("attn_dropout", [0.0, 0.3])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("rules", [4])
@pytest.mark.parametrize("q_compose", [False, True])
@pytest.mark.parametrize("dim_selection", [MODEL // 2, None])
@pytest.mark.parametrize("num_rules", [2])
@pytest.mark.parametrize("qk_rule", [True, False])
@pytest.mark.parametrize("nonlinear", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_build_and_run(
    heads: int,
    attn_dropout: float,
    causal: bool,
    rules: int,
    q_compose: bool,
    dim_selection: int,
    num_rules: int,
    qk_rule: bool,
    nonlinear: bool,
    device: torch.device,
):

    torch.manual_seed(42)

    test_config = {
        "name": "compositional",
        "dropout": attn_dropout,
        "causal": causal,
        "seq_len": SEQ,
        "dim_model": MODEL,
        "num_heads": heads,
        "num_rules": num_rules,
        "q_compose": q_compose,
        "rules": rules,
        "dim_selection": dim_selection,
        "qk_rule": qk_rule,
        "nonlinear": nonlinear,
    }

    attention = build_attention(test_config)

    # build a multi head dispatch to test this attention mechanism
    multi_head = MultiHeadDispatch(
        seq_len=SEQ,
        dim_model=MODEL,
        num_heads=heads,
        attention=attention,
        residual_dropout=0.0,
    ).to(device)

    # Check that a shuffled input produces the same results
    seqs = [SEQ, SEQ // 2]

    for seq in seqs:
        # Check that we can pass a smaller sequence
        inputs = torch.rand(BATCH, seq, MODEL, device=device)
        shuffle = torch.randperm(inputs.shape[1])
        inputs_shuffled = inputs[:, shuffle, :].clone()

        results = multi_head(inputs, inputs, inputs)
        results_shuffled = multi_head(inputs_shuffled, inputs_shuffled, inputs_shuffled)

        if attn_dropout == 0.0 and num_rules == 1 and not causal:
            assert (results[:, shuffle, :] - results_shuffled).abs().max() < 1e-3

        # Test the non-self-attention codepath
        att = multi_head(inputs, inputs_shuffled, inputs)

        # Check that dropout actually drops some values
        if attn_dropout > 0:
            att_2 = multi_head(inputs, inputs_shuffled, inputs)
            assert (att != att_2).any()
