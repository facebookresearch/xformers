# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

import xformers.ops as xops
from xformers.ops import indexing

from .utils import assert_allclose


@pytest.mark.skipif(
    not indexing.ScaledIndexAddFw.is_available(), reason="not available"
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_scaling", [False, True])
@pytest.mark.parametrize(
    "out_shape", [(48, 1, 257 * 1536), (48, 257, 1536), (192, 50, 1536)]
)
def test_scaled_index_add(out_shape, with_scaling: bool) -> None:
    torch.manual_seed(0)
    alpha = 0.73
    dtype = torch.float16
    B_out, M, D = out_shape
    B_src = int(B_out * 0.6)

    inp = torch.randn([B_out, M, D], device="cuda", dtype=dtype, requires_grad=True)
    src = torch.randn([B_src, M, D], device="cuda", dtype=dtype, requires_grad=True)
    TENSORS = {"inp": inp, "src": src}

    index_py = [i for i in range(src.shape[0])]
    random.Random(B_out).shuffle(index_py)
    index = torch.tensor(index_py, dtype=torch.int64, device="cuda")

    if with_scaling:
        scaling = torch.randn([D], device="cuda", dtype=dtype, requires_grad=True)
        TENSORS["scaling"] = scaling
        ref_src_scaled = scaling.float() * src.float()
    else:
        scaling = None
        ref_src_scaled = src.float()
    ref_out = torch.index_add(
        inp.float(), dim=0, source=ref_src_scaled, index=index, alpha=alpha
    ).to(dtype)
    grad_output = torch.randn_like(ref_out)
    ref_out.backward(grad_output)
    ref_grads = {k: v.grad for k, v in TENSORS.items()}
    for v in TENSORS.values():
        v.grad = None

    # Test FW
    out = xops.scaled_index_add(
        inp.clone(),
        index,
        src,
        scaling,
        alpha,
    )
    assert_allclose(out, ref_out, "fw", atol=4e-3, rtol=1e-3)
    # Test BW
    out.backward(grad_output)
    for k, v in TENSORS.items():
        atol = 1e-5
        rtol = 1e-5
        # NOTE: Ordering of operations is not 100% the same as PT, hence the small numeric diff
        if k == "scaling":
            atol, rtol = 5e-2, 1e-2
        assert_allclose(v.grad, ref_grads[k], f"{k}.grad", atol=atol, rtol=rtol)  # type: ignore


@pytest.mark.skipif(not indexing.IndexSelect.is_available(), reason="not available")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("D", [1536])
@pytest.mark.parametrize("batches", [((48, 25), (192, 50))])
def test_index_select_cat(D, batches) -> None:
    torch.manual_seed(0)
    dtype = torch.float16

    num_rows = 0
    for B, seqlen in batches:
        num_rows += B * seqlen

    src = torch.randn([num_rows, D], device="cuda", dtype=dtype, requires_grad=True)
    indices = []
    sources = []
    rows_begin = 0
    for B, seqlen in batches:
        index = [i for i in range(B)]
        random.Random(B).shuffle(index)
        indices.append(
            torch.tensor(index[: int(0.6 * B)], dtype=torch.int64, device="cuda")
        )
        sources.append(
            src[rows_begin : rows_begin + B * seqlen].reshape([B, seqlen * D])
        )
        rows_begin += B * seqlen

    # PT implem
    ref_out = torch.cat([s[i].flatten() for s, i in zip(sources, indices)], dim=0)
    gradient_out = torch.randn_like(ref_out)
    ref_out.backward(gradient_out)
    assert src.grad is not None
    ref_grad = src.grad.clone()
    src.grad = None

    # xFormers implem
    out = xops.index_select_cat(sources, indices)
    assert_allclose(out, ref_out, "fw")
    out.backward(gradient_out)
    assert src.grad is not None
    assert_allclose(src.grad, ref_grad, "src.grad")
