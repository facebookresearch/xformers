# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import math
from functools import partial
from typing import Optional

import pytest
import torch

from xformers.ops import rope_padded
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalWithOffsetPaddedKeysMask

from .utils import assert_allclose

compute_capability = (0, 0)
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability("cuda")
cuda_sm80_only = pytest.mark.skipif(
    compute_capability < (8, 0), reason="requires sm80+"
)


def apply_scaling(
    freqs: torch.Tensor,
    old_context_len: float,
    low_freq_factor: float,
    high_freq_factor: float,
    dynamic_scale_factor: float,
):
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    assert low_freq_wavelen >= high_freq_wavelen

    for idx, freq in enumerate(freqs):
        wavelen = 2 * math.pi / freq
        if wavelen > low_freq_wavelen:
            freqs[idx] = freq / dynamic_scale_factor

        if high_freq_wavelen <= wavelen and wavelen <= low_freq_wavelen:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            freqs[idx] = (1 - smooth) * freqs[
                idx
            ] / dynamic_scale_factor + smooth * freqs[idx]
    return freqs


def _slow_rope(
    x: torch.Tensor,
    *,
    seqpos: Optional[torch.Tensor] = None,
    theta=10000,
    linear_scale=1,
    adjacents: bool = True,
    use_dynamic_scaling: bool = False,
    dynamic_old_context_len: float = 8192.0,
    dynamic_scale_factor: float = 16.0,
    dynamic_low_freq_factor: float = 1.0,
    dynamic_high_freq_factor: float = 32.0,
):
    """
    Simple rope calculation of rope of one tensor

    Args:
        x: input, shape (B, M, H, K).
        seqpos: gives the position of each sequence element in x in its sequence
            (shape (M,)).
    """
    x_shape = x.shape
    dim = x_shape[-1]
    seq_dim = 1
    M = x_shape[seq_dim]
    assert dim % 2 == 0
    if seqpos is None:
        seqpos = torch.arange(M, device=x.device)
    power = torch.arange(0, dim, 2, device=x.device)[: (dim // 2)].float() / dim
    freqs: torch.Tensor = 1.0 / (theta**power)  # type: ignore
    if use_dynamic_scaling:
        freqs = apply_scaling(
            freqs,
            dynamic_old_context_len,
            dynamic_low_freq_factor,
            dynamic_high_freq_factor,
            dynamic_scale_factor,
        )
    all_freqs = torch.outer(seqpos / linear_scale, freqs)
    freqs_cis = torch.polar(torch.ones_like(all_freqs), all_freqs)  # complex64
    for _ in range(x.ndim - seq_dim - 2):
        freqs_cis = freqs_cis[:, None]
    if adjacents:
        x_reshaped = x.float().unflatten(-1, (-1, 2))
        x_ = torch.view_as_complex(x_reshaped)
        x_out = torch.view_as_real(x_ * freqs_cis)
    else:
        x_reshaped = x.float().unflatten(-1, (2, -1)).transpose(-1, -2).contiguous()
        x_ = torch.view_as_complex(x_reshaped)
        x_out = torch.view_as_real(x_ * freqs_cis)
        x_out = x_out.transpose(-1, -2)
    return x_out.flatten(-2).type_as(x)


def _slow_rope2(
    x: torch.Tensor,
    *,
    seqpos: Optional[torch.Tensor] = None,
    theta=10000,
    linear_scale=1,
    adjacents: bool = True,
):
    """
    More flexible unused version of _slow_rope
    - allows varying dtypes.
    """
    internal_dtype = torch.float64
    dim = x.shape[-1]
    seq_dim = 1
    M = x.shape[seq_dim]
    assert dim % 2 == 0
    if seqpos is None:
        seqpos = torch.arange(M, device=x.device)
    power = (
        torch.arange(0, dim, 2, device=x.device)[: (dim // 2)].to(internal_dtype) / dim
    )
    # freqs = 1.0 / (theta**power)
    freqs = theta**-power
    f = torch.outer(seqpos / linear_scale, freqs)
    for _ in range(x.ndim - seq_dim - 2):
        f = f[:, None]
    if adjacents:
        x1, x2 = x.to(internal_dtype).unflatten(-1, (-1, 2)).unbind(-1)
        y1 = x1 * f.cos() - x2 * f.sin()
        y2 = x1 * f.sin() + x2 * f.cos()
        x_out = torch.stack([y1, y2], -1)
    else:
        x1, x2 = x.to(internal_dtype).unflatten(-1, (2, -1)).unbind(-2)
        y1 = x1 * f.cos() - x2 * f.sin()
        y2 = x1 * f.sin() + x2 * f.cos()
        x_out = torch.stack([y1, y2], -2)
    return x_out.flatten(-2).type_as(x)


DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}
ROPE_ATOL_RTOL = {
    "bf16": (5e-3, 8e-3),
    "f32": (5e-3, 1e-5),
}


@cuda_sm80_only
@pytest.mark.parametrize(
    "adjacents", [True, False], ids=lambda x: "adj" if x else "non-adj"
)
@pytest.mark.parametrize("dtype_str", ["bf16", "f32"])
@pytest.mark.parametrize("internal_dtype", ["", "f32", "f64"])
@pytest.mark.parametrize("dim", [100, 4098])
@pytest.mark.parametrize("padding", [87, 18300])
@pytest.mark.parametrize("groups", [1, 3])
@pytest.mark.parametrize(
    "linear_scale, use_dynamic_scaling", [(1.0, False), (4.0, False), (1.0, True)]
)
def test_consistency(
    adjacents: bool,
    dim: int,
    padding: int,
    groups: int,
    internal_dtype: str,
    dtype_str: str,
    linear_scale: float,
    use_dynamic_scaling: bool,
):
    torch.manual_seed(1)
    heads, kvheads = 10, 2
    nqueries = [2, 1, 1]
    cache_lens = [27, padding - 5, padding // 2]
    device = torch.device("cuda")
    dtype = DTYPES[dtype_str]

    # Can we make the internals of attn_bias be on the gpu.
    attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=nqueries, kv_padding=padding, kv_seqlen=cache_lens
    )

    total_cache_length = len(cache_lens) * padding
    total_nqueries = sum(nqueries)
    if groups == 1:
        cache_k = torch.rand(
            1, total_cache_length, kvheads, dim, device=device, dtype=dtype
        )
        cache_v = torch.rand(
            1, total_cache_length, kvheads, dim, device=device, dtype=dtype
        )
        xq = torch.rand(1, total_nqueries, heads, dim, device=device, dtype=dtype)
        xk = torch.rand(1, total_nqueries, kvheads, dim, device=device, dtype=dtype)
        xv = torch.rand(1, total_nqueries, kvheads, dim, device=device, dtype=dtype)
    else:
        cache_k = torch.rand(
            1, total_cache_length, groups, kvheads, dim, device=device, dtype=dtype
        )
        cache_v = torch.rand(
            1, total_cache_length, groups, kvheads, dim, device=device, dtype=dtype
        )
        xq = torch.rand(
            1, total_nqueries, groups, heads, dim, device=device, dtype=dtype
        )
        xk = torch.rand(
            1, total_nqueries, groups, kvheads, dim, device=device, dtype=dtype
        )
        xv = torch.rand(
            1, total_nqueries, groups, kvheads, dim, device=device, dtype=dtype
        )

    cache_k_orig = cache_k.clone()
    cache_v_orig = cache_v.clone()
    out = rope_padded(
        xq,
        xk,
        xv,
        cache_k,
        cache_v,
        attn_bias,
        linear_scale=linear_scale,
        adjacents=adjacents,
        internal_dtype=internal_dtype,
        use_dynamic_scaling=use_dynamic_scaling,
    )

    seqpos = torch.tensor(
        [cache_lens[0] - 2, cache_lens[0] - 1, cache_lens[1] - 1, cache_lens[2] - 1],
        device=device,
    )
    cache_locs = [seqpos[0], seqpos[1], padding + seqpos[2], 2 * padding + seqpos[3]]
    baseline = _slow_rope if dtype_str == "f32" else _slow_rope2
    if use_dynamic_scaling:
        baseline = partial(_slow_rope, use_dynamic_scaling=True)  # type: ignore
    expected_out = baseline(  # type: ignore
        xq, linear_scale=linear_scale, seqpos=seqpos, adjacents=adjacents
    )
    atol, rtol = ROPE_ATOL_RTOL[dtype_str]
    assert_allclose(out, expected_out, atol=atol, rtol=rtol)

    assert_allclose(cache_v[:, cache_locs], xv, atol=atol, rtol=rtol)
    cache_v[:, cache_locs] = cache_v_orig[:, cache_locs]
    assert torch.allclose(cache_v, cache_v_orig)

    slow_roped_xk = _slow_rope(
        xk,
        linear_scale=linear_scale,
        seqpos=seqpos,
        adjacents=adjacents,
        use_dynamic_scaling=use_dynamic_scaling,
    )
    assert_allclose(
        cache_k[:, cache_locs],
        slow_roped_xk,
        atol=atol,
        rtol=rtol,
    )
    cache_k[:, cache_locs] = cache_k_orig[:, cache_locs]
    assert torch.allclose(cache_k, cache_k_orig)


@cuda_sm80_only
@pytest.mark.parametrize("seqlen", [512, 2**16])
def test_rope_prefill(seqlen) -> None:
    heads, kvheads = 2, 1
    dim = 32
    device = "cuda"
    adjacents = True
    dtype = torch.bfloat16

    attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[seqlen], kv_padding=seqlen + 1, kv_seqlen=[seqlen]
    )
    cache_k = torch.rand(1, seqlen + 1, kvheads, dim, device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)
    xq = torch.rand(1, seqlen, heads, dim, device=device, dtype=dtype)
    xk = torch.rand(1, seqlen, kvheads, dim, device=device, dtype=dtype)
    xv = torch.rand(1, seqlen, kvheads, dim, device=device, dtype=dtype)

    out = rope_padded(
        xq,
        xk,
        xv,
        cache_k,
        cache_v,
        attn_bias,
        adjacents=adjacents,
    )

    seqpos = torch.arange(start=0, end=seqlen, device=device)
    expected_out = _slow_rope2(xq, seqpos=seqpos, adjacents=adjacents)
    atol, rtol = ROPE_ATOL_RTOL["bf16"]
    assert_allclose(out, expected_out, atol=atol, rtol=rtol)


@cuda_sm80_only
def test_rope_seqpos() -> None:
    heads, kvheads = 2, 1
    dim = 32
    device = "cuda"
    adjacents = True
    dtype = torch.bfloat16
    seqlen = 723

    attn_bias = BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[seqlen], kv_padding=seqlen + 1, kv_seqlen=[seqlen]
    )
    cache_k = torch.rand(1, seqlen + 1, kvheads, dim, device=device, dtype=dtype)
    cache_v = torch.randn_like(cache_k)
    xq = torch.rand(1, seqlen, heads, dim, device=device, dtype=dtype)
    xk = torch.rand(1, seqlen, kvheads, dim, device=device, dtype=dtype)
    xv = torch.rand(1, seqlen, kvheads, dim, device=device, dtype=dtype)

    def inner(seqpos, *, first_seqpos_input=None, seqpos_input=None):
        out = rope_padded(
            xq,
            xk,
            xv,
            cache_k,
            cache_v,
            attn_bias,
            adjacents=adjacents,
            first_seqpos=first_seqpos_input,
            seqpos=seqpos_input,
        )

        expected_out = _slow_rope2(xq, seqpos=seqpos, adjacents=adjacents)
        atol, rtol = ROPE_ATOL_RTOL["bf16"]
        assert_allclose(out, expected_out, atol=atol, rtol=rtol)

    inner(torch.arange(start=0, end=seqlen, device=device))
    inner(
        torch.arange(start=4, end=seqlen + 4, device=device),
        first_seqpos_input=torch.tensor([4], device=device),
    )
    custom_seqpos = torch.arange(start=0, end=seqlen, device=device)
    custom_seqpos[231] = 934
    custom_seqpos[423] = 134
    inner(custom_seqpos, seqpos_input=custom_seqpos)
