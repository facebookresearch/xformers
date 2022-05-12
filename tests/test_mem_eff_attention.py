# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

import xformers.ops

_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def ref_attention(q, k, v, attn_bias=None):
    q = q * (1 / q.shape[-1] ** 0.5)
    if attn_bias is None:
        return (q @ k.transpose(-2, -1)).softmax(-1) @ v
    else:
        return (q @ k.transpose(-2, -1) + attn_bias).softmax(-1) @ v


@pytest.mark.parametrize("use_attn_bias", [False, True])
@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33])
@pytest.mark.parametrize("q_len", [2, 3, 5])
@pytest.mark.parametrize("device", _devices)
def test_memory_efficient_attention(
    device, q_len, kv_len, batch_size, k_len, use_attn_bias
):
    scale = 3
    query = torch.randn((batch_size, q_len, k_len), device=device) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    attn_bias = None
    if use_attn_bias:
        attn_bias = torch.randn((batch_size, 1, kv_len), device=device) * scale
        attn_bias = attn_bias.expand(batch_size, q_len, kv_len)

    out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias)
    ref = ref_attention(query, key, value, attn_bias)

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


@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33])
@pytest.mark.parametrize("q_len", [2, 3, 5])
@pytest.mark.parametrize("device", _devices)
def test_logsumexp(device, q_len, kv_len, batch_size, k_len):
    scale = 3
    query = torch.randn((batch_size, q_len, k_len), device=device) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    _, lse = torch.ops.xformers.efficient_attention(query, key, value, True, None)
    ref_lse = ((query / k_len**0.5) @ key.transpose(-2, -1)).logsumexp(-1)

    assert torch.allclose(lse, ref_lse, atol=2e-4)


@pytest.mark.parametrize("use_attn_bias", [False, True])
@pytest.mark.parametrize("grad_out_contiguous", [False, True])
@pytest.mark.parametrize("k_len", [5, 6, 32])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [3, 15, 32, 33])
@pytest.mark.parametrize("q_len", [2, 3, 5])
@pytest.mark.parametrize("device", _devices)
def test_memory_efficient_attention_backward(
    device, q_len, kv_len, batch_size, k_len, grad_out_contiguous, use_attn_bias
):
    scale = 3
    query = torch.randn((batch_size, q_len, k_len), device=device) * scale
    key = torch.randn((batch_size, kv_len, k_len), device=device) * scale
    value = torch.randn((batch_size, kv_len, k_len), device=device) * scale

    attn_bias = None
    if use_attn_bias:
        attn_bias = torch.randn((batch_size, 1, kv_len), device=device) * scale
        attn_bias = attn_bias.expand(batch_size, q_len, kv_len)

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    grad_out = torch.ones_like(query)
    if grad_out_contiguous is False:
        grad_out = torch.tensor([1.0], device=device)[None, None, :].expand_as(query)

    out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias)
    out.backward(grad_out)

    grad_q = query.grad
    grad_k = key.grad
    grad_v = value.grad

    query.grad = None
    key.grad = None
    value.grad = None

    ref = ref_attention(query, key, value, attn_bias)
    ref.backward(grad_out)

    # there is some extra precision loss in the CPU implementation due to an
    # extra accumulation step in grad_q, which is not present in the CUDA
    # implementation
    atol = 5e-4 if device == "cuda" else 6e-4
    assert torch.allclose(
        grad_q, query.grad, atol=atol
    ), f"grad_q doesn't match {(grad_q - query.grad).abs().max()}"
    assert torch.allclose(
        grad_k, key.grad, atol=atol
    ), f"grad_k doesn't match {(grad_k - key.grad).abs().max()}"
    assert torch.allclose(
        grad_v, value.grad, atol=atol
    ), f"grad_v doesn't match {(grad_v - value.grad).abs().max()}"
