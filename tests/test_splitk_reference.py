# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import pytest
import torch

import xformers.ops
from xformers.ops import fmha

from .utils import assert_allclose, disable_tf32, ref_attention_for_test


@disable_tf32
def ref_attention_splitk_bmhk(
    q, k, v, attn_bias, scale=None, split_k=None, dtype=None
) -> torch.Tensor:
    assert q.ndim == 4

    def T(t):
        return t.permute((0, 2, 1, 3)).reshape(
            [t.shape[0] * t.shape[2], t.shape[1], t.shape[3]]
        )

    if isinstance(attn_bias, xformers.ops.AttentionBias):
        attn_bias = attn_bias.materialize(
            (q.shape[0], q.shape[2], q.shape[1], k.shape[1]),
            device=q.device,
            dtype=torch.float32,
        ).reshape([q.shape[0] * q.shape[2], q.shape[1], k.shape[1]])
    out = ref_attention_splitk(
        T(q), T(k), T(v), attn_bias, scale=scale, split_k=split_k, dtype=dtype
    )
    out = out.reshape([q.shape[0], q.shape[2], q.shape[1], v.shape[3]])
    return out.permute((0, 2, 1, 3))


@disable_tf32
def ref_attention_splitk(
    q, k, v, attn_bias, scale=None, split_k=2, dtype=None
) -> torch.Tensor:
    if q.ndim == 5:

        def attn_bias_group(group: int):
            if isinstance(attn_bias, torch.Tensor):
                return attn_bias[:, group]
            if isinstance(attn_bias, fmha.attn_bias.LowerTriangularMaskWithTensorBias):
                return fmha.attn_bias.LowerTriangularMaskWithTensorBias(
                    attn_bias._bias[:, group]
                )
            return attn_bias

        return torch.stack(
            [
                ref_attention_splitk_bmhk(
                    q[:, :, g],
                    k[:, :, g],
                    v[:, :, g],
                    attn_bias=attn_bias_group(g),
                    split_k=split_k,
                    dtype=dtype,
                )
                for g in range(q.shape[2])
            ],
            dim=2,
        )

    if q.ndim == 4:
        return ref_attention_splitk_bmhk(
            q, k, v, attn_bias=attn_bias, split_k=split_k, dtype=dtype
        )
    assert q.ndim == 3
    if dtype is None:
        dtype = torch.float32
    q = q.to(dtype=dtype)
    k = k.to(dtype=dtype)
    v = v.to(dtype=dtype)

    if scale is None:
        scale = q.shape[-1] ** -0.5
    assert not q.isnan().any()
    q = q * scale
    assert not q.isnan().any()

    if attn_bias is not None:
        if isinstance(attn_bias, xformers.ops.AttentionBias):
            # Always create in B,H,Mq,Mk format
            attn_bias_tensor = attn_bias.materialize(
                (q.shape[0], 1, q.shape[1], k.shape[1]),
                device=q.device,
                dtype=torch.float32,
            )
        else:
            attn_bias_tensor = attn_bias
        if attn_bias_tensor.ndim == 4:
            assert q.shape[0] == attn_bias_tensor.shape[0] * attn_bias_tensor.shape[1]
            attn_bias_tensor = attn_bias_tensor.reshape(
                [-1, *attn_bias_tensor.shape[2:]]
            )

    split_size = k.size(-2) // split_k
    split_config = {"dim": -2, "split_size_or_sections": split_size}
    k_split = torch.split(k, **split_config)
    v_split = torch.split(v, **split_config)
    attn_bias_split = torch.split(
        attn_bias_tensor, dim=-1, split_size_or_sections=split_size
    )

    def compute_attention_split(q_whole, k_slice, v_slice, attn_bias_slice):
        p_slice = q_whole @ k_slice.transpose(-2, -1)
        p_slice += attn_bias_slice
        row_max = torch.max(p_slice, dim=-1, keepdim=True).values
        p_slice_scaled = p_slice - row_max
        p_slice_scaled[p_slice_scaled.isnan()] = float("-inf")
        s = torch.exp(p_slice_scaled)
        row_sumexp = torch.sum(s, dim=-1, keepdim=True)
        attn_slice = s @ v_slice
        return {
            "attn_slice": attn_slice,
            "row_max": row_max,
            "row_sumexp": row_sumexp,
        }

    splits = list(zip(k_split, v_split, attn_bias_split))

    slices = list(map(lambda s: compute_attention_split(q, s[0], s[1], s[2]), splits))
    out = torch.zeros_like(q)

    # reduce out over split-k slices

    global_max = torch.zeros_like(slices[0]["row_max"]).fill_(float("-inf"))
    global_sumexp = torch.zeros_like(slices[0]["row_sumexp"])

    for s in slices:
        local_out = s["attn_slice"]
        local_max = s["row_max"]
        local_sumexp = s["row_sumexp"]

        log_alpha = -torch.abs(local_max - global_max)
        alpha = torch.exp(log_alpha)
        alpha.nan_to_num_(1.0)

        pick_new = local_max < global_max
        new_coef = torch.where(pick_new, alpha, 1.0)
        curr_coef = torch.where(pick_new, 1.0, alpha)

        out = out * curr_coef + local_out * new_coef
        global_sumexp = global_sumexp * curr_coef + local_sumexp * new_coef
        global_max = torch.max(local_max, global_max)
    out /= global_sumexp
    return out


def _kv_heads_label(kv_heads: Optional[int]) -> str:
    if kv_heads is None:
        return ""
    if kv_heads == 1:
        return "mq"
    return f"gqa{kv_heads}"


@pytest.mark.parametrize("dtype", ["f32"])
@pytest.mark.parametrize("kv_heads", [None, 1, 2], ids=_kv_heads_label)
@pytest.mark.parametrize("n_heads", [16])
@pytest.mark.parametrize("padding, bsz", [(32, 8), (4096, 1)])
@pytest.mark.parametrize("split_k", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu"])
def test_splitk_reference(
    kv_heads: int,
    n_heads: int,
    padding: int,
    bsz: int,
    dtype: str,
    device: str,
    split_k: int,
):
    dtype_ = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]
    torch.manual_seed(1)
    d = 256
    num_queries = 1
    if kv_heads is not None and kv_heads > 1:
        k_shape: Tuple[int, ...] = (1, bsz * padding, kv_heads, n_heads, d)
        q_shape: Tuple[int, ...] = (
            1,
            bsz * num_queries,
            kv_heads,
            n_heads,
            d,
        )
    else:
        k_shape = (1, bsz * padding, n_heads, d)
        q_shape = (1, bsz * num_queries, n_heads, d)

    k = torch.rand(k_shape, dtype=dtype_, device=device)
    k_seqlen = torch.randint(1, padding + 1, (bsz,)).tolist()
    v = torch.rand_like(k)
    q = torch.rand(q_shape, dtype=dtype_, device=device)
    causal_diagonal = torch.tensor(  # TODO: make unnecessary
        [i - 1 for i in k_seqlen], dtype=torch.int32, device=device
    )

    if kv_heads is not None:
        k = k[..., :1, :].expand(k_shape)
        v = v[..., :1, :].expand(k_shape)

    attn_bias = fmha.attn_bias.BlockDiagonalCausalWithOffsetPaddedKeysMask.from_seqlens(
        q_seqlen=[1] * bsz,
        kv_seqlen=k_seqlen,
        causal_diagonal=causal_diagonal,
        kv_padding=padding,
    )
    ref_out = ref_attention_for_test(q, k, v, attn_bias)
    splitk_out = ref_attention_splitk(q, k, v, attn_bias, None, split_k=split_k)
    assert_allclose(
        ref_out,
        splitk_out,
        atol=fmha.ck.FwOp.ERROR_ATOL[dtype_],
        rtol=fmha.ck.FwOp.ERROR_RTOL[dtype_],
    )
