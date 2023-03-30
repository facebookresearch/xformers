# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random

import pytest
import torch

import xformers.ops as xops

from .utils import assert_allclose

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda_only
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
    if with_scaling:
        scaling = torch.randn([D], device="cuda", dtype=dtype, requires_grad=True)
        TENSORS["scaling"] = scaling
    else:
        scaling = torch.Tensor()

    index_py = [i for i in range(src.shape[0])]
    random.Random(B_out).shuffle(index_py)
    index = torch.tensor(index_py, dtype=torch.int64, device="cuda")

    if with_scaling:
        ref_src_scaled = scaling.float() * src.float()
    else:
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
        input=inp.clone(),
        index=index,
        source=src,
        scaling=scaling if with_scaling else None,
        alpha=alpha,
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


@cuda_only
@pytest.mark.parametrize("D", [1536])
def test_index_select_cat(D) -> None:
    torch.manual_seed(0)
    dtype = torch.float16
    srcs = [
        torch.randn([48, 25 * D]),
        torch.randn([192, 50 * D]),
    ]
    src = torch.cat([s.view([-1, D]) for s in srcs], dim=0).cuda().to(dtype)
    src.requires_grad_(True)

    indices = []
    sources = []
    elements_i = 0
    for source_i in srcs:
        index = [i for i in range(source_i.shape[0])]
        random.Random(source_i.shape[0]).shuffle(index)
        indices.append(
            torch.tensor(
                index[: int(0.6 * source_i.shape[0])], dtype=torch.int64, device="cuda"
            )
        )
        sources.append(
            src[
                elements_i : elements_i + source_i.shape[0] * source_i.shape[1] // D
            ].reshape(source_i.shape)
        )
        elements_i += source_i.shape[0] * source_i.shape[1] // D

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
