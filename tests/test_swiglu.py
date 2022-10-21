# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Optional

import pytest
import torch

import xformers.ops.swiglu as xsw

torch.backends.cuda.matmul.allow_tf32 = False
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
_devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


def assert_allclose(
    # The output of the tested function
    out: torch.Tensor,
    # The output of the reference implementation
    ref: torch.Tensor,
    # The output of the reference implementation in f32
    ref32: Optional[torch.Tensor] = None,
    msg: str = "failed",
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    """
    Improved version of
    ```
    assert torch.allclose(out, ref)
    ```

    Except that we provide useful error message, and also compare
    to the output of the f32 calculation.
    """
    out = out.float()
    ref = ref.float()
    if atol is None:
        atol = 1e-8
    if rtol is None:
        rtol = 1e-5
    assert out.shape == ref.shape
    compare_to = ref32 if ref32 is not None else ref
    assert out.shape == compare_to.shape
    if torch.allclose(out, ref, rtol=rtol, atol=atol) or (
        ref32 is not None and torch.allclose(out, ref32, rtol=rtol, atol=atol)
    ):
        return

    flatten_diff = ((out - compare_to).abs() - atol - compare_to.abs() * rtol).flatten()
    max_pos = flatten_diff.argmax()

    if ref32 is not None:
        flatten_diff_vsf32 = ((ref - ref32).abs() - atol - ref32.abs() * rtol).flatten()
        max_pos_vsf32 = flatten_diff_vsf32.argmax()
        assert False, (
            f"{msg}: "
            f"out={out.flatten()[max_pos]} and ref32={compare_to.flatten()[max_pos]} (diff={flatten_diff[max_pos]} > 0)"
            f"/ atol={atol}, rtol={rtol}.\n"
            f"NOTE: ref vs ref_f32:\n"
            f"ref={ref.flatten()[max_pos_vsf32]} and ref32={ref32.flatten()[max_pos_vsf32]} "
            f"(diff={flatten_diff_vsf32[max_pos_vsf32]})"
        )
    else:
        assert False, (
            f"{msg}: "
            f"out={out.flatten()[max_pos]} and ref={compare_to.flatten()[max_pos]} (diff={flatten_diff[max_pos]} > 0)"
            f"/ atol={atol}, rtol={rtol}"
        )


def generate_test_shapes():
    shapes = [
        # Format: [inp.shape[0], inp.shape[1], hidden.shape[1]]
        # ViT-Giant
        (9456, 1536, 4096),
        (4440, 1536, 4096),
        (4728, 1536, 4096),
    ]
    # Add some random shapes
    r = random.Random(0)
    for _ in range(20):
        shapes.append((r.randint(1, 5000), r.randint(1, 5000), r.randint(1, 512) * 8))
    return shapes


_test_shapes = list(generate_test_shapes())
_test_shapes_ids = [str(s) for s in _test_shapes]
_dtypes = [torch.float, torch.float16]


@pytest.mark.parametrize("autocast", [False])  # TODO: Enable autocast testing
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize(
    "shape",
    _test_shapes,
    ids=_test_shapes_ids,
)
def test_forward_backward(
    shape,
    device,
    dtype,
    autocast: bool,
):
    FORWARD_ATOL = {torch.float: 2e-6, torch.half: 1e-3}
    BACKWARD_ATOL = {
        torch.float: 3e-4,
        torch.half: 0.5,
    }
    BACKWARD_RTOL = {
        torch.float: 2e-3,
        torch.half: 1e-2,
    }

    if device == "cpu" and dtype is not torch.float:
        pytest.skip("Half not supported on CPU")
    if autocast and (device == "cpu" or dtype is not torch.half):
        pytest.skip("Autocast only supported for CUDA+Half")

    inp_model_dtype = torch.float if autocast else dtype
    x = torch.randn(shape[:2], device=device, dtype=inp_model_dtype)
    op = xsw._SwiGLUDecomposedOp

    module = xsw._SwiGLUModule(in_features=shape[1], hidden_features=shape[2])
    x_f32: Optional[torch.Tensor]
    ref_f32: Optional[torch.Tensor]
    module_f32: Optional[torch.nn.Module]
    if dtype != torch.float:
        x_f32, module_f32 = x.to(device).to(torch.float), module.to(device)
        x_f32.requires_grad_()
        ref_f32 = module_f32(x_f32)
    else:
        x_f32, module_f32, ref_f32 = None, None, None

    x, module = x.to(device).to(inp_model_dtype), module.to(device).to(inp_model_dtype)
    x.requires_grad_()

    # Forward
    if autocast:
        with torch.autocast("cuda", dtype=dtype):
            ref = module(x)
    else:
        ref = module(x)
    out = xsw.functional_swiglu(x, *module._ordered_params_for_op(), op=op)
    if ref_f32 is None:
        ref_f32 = ref

    assert_allclose(
        out,
        ref,
        ref_f32,
        "fw",
        atol=FORWARD_ATOL[dtype],
    )

    # Backward
    grad = torch.randn_like(ref)

    def backward_gather_grads(inp, output):
        output.backward(grad.to(output.dtype))
        grads = {}
        for name, param in module.named_parameters():
            grads[name] = param.grad.clone()
            param.grad = None
        grads["x"] = inp.grad.clone()
        inp.grad = None
        return grads

    grads_ref = backward_gather_grads(x, ref)
    grads_out = backward_gather_grads(x, out)
    grads_ref32 = (
        backward_gather_grads(x_f32, ref_f32) if module_f32 is not None else grads_ref
    )

    assert list(grads_ref.keys()) == list(grads_out.keys())
    for name, gref in grads_ref.items():
        gout = grads_out[name]
        assert_allclose(
            gout,
            gref,
            grads_ref32.get(name),
            f"{name}.grad",
            atol=BACKWARD_ATOL[dtype],
            rtol=BACKWARD_RTOL[dtype],
        )
        # Ensure `gout >> atol`, so that the test is meaningful
        assert gout.norm(2) > BACKWARD_ATOL[dtype] / BACKWARD_RTOL[dtype]
