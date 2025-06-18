# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
import random
from contextlib import nullcontext
from typing import cast, ContextManager, Optional, Sequence, Union

import pytest
import torch

import xformers
import xformers.ops.swiglu_op as xsw

from .utils import disable_tf32

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
if torch.cuda.is_available():
    _devices = ["cuda"]
    _is_sm80 = torch.cuda.get_device_capability(_devices[0])[0] >= 8
else:
    _devices = []
    _is_sm80 = False
cuda_sm80_only = pytest.mark.skipif(not _is_sm80, reason="requires sm80+")
disable_on_rocm = pytest.mark.skipif(
    not not torch.version.hip, reason="could not be done on ROCM"
)


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
        (9456, 1536, 2736),
        (4440, 1536, 2736),
        (4728, 1536, 2736),
        # GPT-3 (small)
        (2048, 2048, 5632),
        # TODO: Not enough memory for this shape in github CI.
        # restore it after rearrange the code (fw, fw, bw, bw) -> (fw, bw, fw, bw)
        # Chinchilla
        # (2048, 8192, 22016),
    ]
    # Add some random shapes
    r = random.Random(0)
    for _ in range(20):
        shapes.append(
            (r.randint(1, 1000) * 8, r.randint(1, 1000) * 8, r.randint(1, 512) * 8)
        )
    return shapes


# Switch between these shape initialisations ...
_test_shapes = list(generate_test_shapes())
_test_shapes_ids = [str(s) for s in _test_shapes]
_dtypes = [torch.float16]
if _is_sm80:
    _dtypes += [torch.bfloat16]
_ops: Sequence[Union[xsw.SwiGLUOp, None]] = [
    xsw.SwiGLUFusedOp,
    xsw.SwiGLUPackedFusedOp,
    None,
]

FORWARD_ATOL = {torch.float: 2e-6, torch.half: 1e-2, torch.bfloat16: 1e-2}
FORWARD_RTOL = {torch.float: 1e-5, torch.half: 4e-3, torch.bfloat16: 4e-3}

BACKWARD_ATOL = {
    torch.float: 3e-4,
    torch.half: 0.5,
    torch.bfloat16: 4.0,  # !!
}
BACKWARD_RTOL = {
    torch.float: 2e-3,
    torch.half: 3e-2,
    torch.bfloat16: 4e-2,
}


@functools.lru_cache(maxsize=1)
def create_module_cached(**kwargs) -> xsw.SwiGLU:
    return xsw.SwiGLU(**kwargs)


@disable_tf32
@disable_on_rocm
@pytest.mark.parametrize("autocast", [False, True], ids=["regular", "autocast"])
@pytest.mark.parametrize(
    "op", _ops, ids=[x.NAME if x is not None else "auto_selected_op" for x in _ops]
)
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("pack_weights", [False, True], ids=["regular", "packed"])
@pytest.mark.parametrize(
    "shape",
    _test_shapes,
    ids=_test_shapes_ids,
)
def test_forward_backward(
    shape,
    device,
    op,
    dtype,
    autocast: bool,
    pack_weights: bool,
    bias: bool,
):
    torch.manual_seed(shape[0] * shape[1] * shape[2])

    if op is not None and not op.supports(
        xsw.SwiGLUOpDispatch(
            device=device,
            dtype=dtype,
            dtype_autocast_gpu=dtype if autocast and device == "cuda" else None,
            packed_weights=pack_weights,
            bias_enabled=bias,
        )
    ):
        pytest.skip("Not supported by operator")

    if op is not None:
        if pack_weights and not op.PACKED_WEIGHTS:
            pytest.skip("Not supported combination when module.op is set manually")

    inp_model_dtype = torch.float if autocast else dtype
    x = torch.randn(shape[:2], device=device, dtype=inp_model_dtype)

    module = copy.deepcopy(
        create_module_cached(
            in_features=shape[1],
            hidden_features=shape[2],
            bias=bias,
            _pack_weights=pack_weights,
        )
    )
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
    cm = cast(
        ContextManager,
        torch.autocast("cuda", dtype=dtype) if autocast else nullcontext(),
    )
    with cm:
        ref = xsw._eager_functional_swiglu(x, *module._ordered_params())

        module.op = op
        out = module(x)

    if ref_f32 is None:
        ref_f32 = ref

    assert_allclose(
        out, ref, ref_f32, "fw", atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype]
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


@cuda_sm80_only
@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_swiglu_compile(
    device,
    dtype,
    bias: bool,
):
    op = xsw.SwiGLUPackedFusedOp
    shape = [2048, 2048, 5632]

    # Eager
    mod = copy.deepcopy(
        create_module_cached(
            in_features=shape[1],
            hidden_features=shape[2],
            bias=bias,
            _pack_weights=True,
        )
    )
    mod = cast(xsw.SwiGLU, mod)
    mod.op = op
    mod = mod.to(device).to(dtype)

    # Torch compile
    mod_c = cast(xsw.SwiGLU, torch.compile(mod, fullgraph=True, dynamic=True))

    assert mod.w12 is not None
    assert mod_c.w12 is not None

    x = torch.randn(shape[:2], device=device, dtype=dtype, requires_grad=True)
    x_c = x.detach().requires_grad_()
    grad = torch.randn(shape[:2], device=device, dtype=dtype, requires_grad=False) * 0.1

    # Forward passes
    output = mod(x)
    output_c = mod_c(x_c)

    # Backward passes
    output.backward(grad)
    output_c.backward(grad)

    assert_allclose(
        output, output_c, msg="fw", atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype]
    )

    assert x_c.grad is not None and x.grad is not None
    assert_allclose(
        x_c.grad,
        x.grad,
        msg="grad",
        atol=BACKWARD_ATOL[dtype],
        rtol=BACKWARD_RTOL[dtype],
    )

    assert mod.w12.weight.grad is not None and mod_c.w12.weight.grad is not None
    assert_allclose(
        mod.w12.weight.grad,
        mod_c.w12.weight.grad,
        msg="w12.grad",
        atol=BACKWARD_ATOL[dtype],
        rtol=BACKWARD_RTOL[dtype],
    )

    assert mod.w3.weight.grad is not None and mod_c.w3.weight.grad is not None
    assert_allclose(
        mod.w3.weight.grad,
        mod_c.w3.weight.grad,
        msg="w3.grad",
        atol=BACKWARD_ATOL[dtype],
        rtol=BACKWARD_RTOL[dtype],
    )

    if bias:
        assert mod.w12.bias.grad is not None and mod_c.w12.bias.grad is not None
        assert_allclose(
            mod.w12.bias.grad,
            mod_c.w12.bias.grad,
            msg="w12.bias.grad",
            atol=BACKWARD_ATOL[dtype],
            rtol=BACKWARD_RTOL[dtype],
        )

        assert mod.w3.bias.grad is not None and mod_c.w3.bias.grad is not None
        assert_allclose(
            mod.w3.bias.grad,
            mod_c.w3.bias.grad,
            msg="w12.bias.grad",
            atol=BACKWARD_ATOL[dtype],
            rtol=BACKWARD_RTOL[dtype],
        )


@disable_tf32
@torch.inference_mode()
@cuda_sm80_only
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize("device", _devices)
@pytest.mark.parametrize("bias", [False, True], ids=["nobias", "bias"])
def test_dual_gemm_silu_identity_mul_compile(dtype, device, bias) -> None:
    N, M, H = (2048, 2048, 5632)

    x = torch.randn([N, M], device=device, dtype=dtype, requires_grad=False)
    w1 = torch.randn([H, M], device=device, dtype=dtype, requires_grad=False)
    w2 = torch.randn([H, M], device=device, dtype=dtype, requires_grad=False)

    b1: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None

    if bias:
        b1 = torch.randn([H], device=device, dtype=dtype, requires_grad=False)
        b2 = torch.randn([H], device=device, dtype=dtype, requires_grad=False)

    DualGemmSiluOp = xformers.ops.common.get_xformers_operator(
        "dual_gemm_silu_identity_mul"
    )

    def fn(x):
        x1, x2, x4 = DualGemmSiluOp(x, w1, b1, w2, b2)
        return [x1, x2, x4]

    # Eager
    output = fn(x)

    # Torch compile
    opt_output = torch.compile(fn, fullgraph=True, dynamic=True)(x)

    for a, b in zip(output, opt_output):
        assert_allclose(
            a, b, msg="fw", atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype]
        )


@disable_tf32
@torch.inference_mode()
@cuda_sm80_only
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize("device", _devices)
def test_gemm_fused_operand_sum_compile(dtype, device) -> None:
    shape = [2048, 2048, 5632]
    x = torch.randn(
        [shape[0], shape[2]], device=device, dtype=dtype, requires_grad=False
    )
    dy = torch.randn(shape[:2], device=device, dtype=dtype, requires_grad=False)

    GemmFusedSumOp = xformers.ops.common.get_xformers_operator("gemm_fused_operand_sum")

    def fn(x):
        return GemmFusedSumOp(dy.transpose(-2, -1), x)

    # Eager
    output = fn(x)

    # Torch compile
    opt_output = torch.compile(fn, fullgraph=True, dynamic=True)(x)

    for a, b in zip(output, opt_output):
        assert_allclose(
            a, b, msg="fw", atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype]
        )


@disable_tf32
@torch.inference_mode()
@pytest.mark.parametrize("dtype", _dtypes, ids=[str(x) for x in _dtypes])
@pytest.mark.parametrize("device", _devices)
def test_silu_bw_fused_compile(dtype, device) -> None:
    shape = [2048, 2048]
    x1 = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    x2 = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)
    dx4 = torch.randn(shape, device=device, dtype=dtype, requires_grad=False)

    SiluBWFusedOp = xformers.ops.common.get_xformers_operator("silu_bw_fused")

    def fn(x1, x2, dx4):
        dx1dx2, x4 = SiluBWFusedOp(x1, x2, dx4)
        return [dx1dx2, x4]

    # Eager
    with torch.autocast("cuda", dtype=dtype):
        output = fn(x1, x2, dx4)

    # Torch compile
    opt_output = torch.compile(fn, fullgraph=True, dynamic=True)(x1, x2, dx4)

    for a, b in zip(output, opt_output):
        assert_allclose(
            a, b, msg="fw", atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype]
        )


@disable_tf32
@cuda_only
@cuda_sm80_only
def test_autocast_silu_bw_fused_compile() -> None:
    shape = [2048, 2048]
    device = "cuda"
    dtype = torch.float32

    x1 = torch.randn(shape, device=device, dtype=dtype)
    x2 = torch.randn(shape, device=device, dtype=dtype)
    dx4 = torch.randn(shape, device=device, dtype=dtype)

    SiluBWFusedOp = xformers.ops.common.get_xformers_operator("silu_bw_fused")

    def fn(x1, x2, dx4):
        dx1dx2, x4 = SiluBWFusedOp(x1, x2, dx4)
        return [dx1dx2, x4]

    output = fn(x1, x2, dx4)

    # Autocast
    with torch.autocast("cuda", dtype=dtype):
        output_ac = fn(x1, x2, dx4)

    for a, b in zip(output, output_ac):
        assert_allclose(
            a, b, msg="fw", atol=FORWARD_ATOL[dtype], rtol=FORWARD_RTOL[dtype]
        )
