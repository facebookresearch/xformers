# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import xformers.ops

_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def ref_attention(q, k, v):
    q = q * (1 / q.shape[-1] ** 0.5)
    return (q @ k.transpose(-2, -1)).softmax(-1) @ v


@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33])
@pytest.mark.parametrize("q_len", [2, 3, 5])
@pytest.mark.parametrize("device", _devices)
def test_memory_efficient_attention(device, q_len, kv_len, batch_size, k_len):
    scale = 3
    query = torch.randn((batch_size, q_len, k_len), device=device) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    out = xformers.ops.memory_efficient_attention(query, key, value)
    ref = ref_attention(query, key, value)

    assert torch.allclose(out, ref, atol=2e-4)


@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 512])
@pytest.mark.parametrize("q_len", [128, 512])
@pytest.mark.parametrize("device", _devices)
def test_key_query_all_ones(device, q_len, kv_len, batch_size, k_len):
    scale = 3
    query = torch.ones((batch_size, q_len, k_len), device=device)
    key = torch.ones((batch_size, kv_len, k_len), device=device)
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    out = xformers.ops.memory_efficient_attention(query, key, value)
    # this should be equivalent to the average over value
    ref = value.mean(1, keepdim=True).expand_as(query)

    assert torch.allclose(out, ref, atol=1e-5)
