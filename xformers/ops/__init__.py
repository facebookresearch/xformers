# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import importlib.util
import logging
import os
import sys
from typing import Any, Optional

import torch
import torch.distributed

from .indexing import index_select_cat, scaled_index_add

# PyTorch can be built without distributed support, in which case
# torch.distributed.is_available() returns False and most of the
# torch.distributed namespace does not exist. That happens for instance in the
# NVIDIA PyTorch containers for Jetson iGPU devices. The model-parallel and
# sequence-parallel ops cannot work at all in that situation, so we skip them
# and keep the rest of xFormers usable, the same way we skip the ops that need
# mslk.
if _HAS_TORCH_DISTRIBUTED := torch.distributed.is_available():
    from .modpar_layers import ColumnParallelLinear, RowParallelLinear
from .rmsnorm import RMSNorm

if _HAS_TORCH_DISTRIBUTED:
    from .seqpar import (
        sequence_parallel_leading_matmul,
        sequence_parallel_trailing_matmul,
    )
    from .sequence_parallel_fused_ops import (
        fused_allgather_and_anything,
        fused_allgather_and_linear,
        fused_anything_and_reducescatter,
        fused_linear_and_reducescatter,
    )
from .swiglu_op import SwiGLU, swiglu, SwiGLUEagerOp, SwiGLUOp, SwiGLUOpDispatch
from .tiled_matmul import tiled_matmul
from .unbind import get_stack_strides, stack_or_none, unbind

logger = logging.getLogger("xformers")

_MSLK_COMPAT_URL = "https://github.com/meta-pytorch/MSLK#release-compatibility-table"


def _import_mslk() -> Optional[Exception]:
    """Import mslk, and return the error if that fails.

    mslk's native library only loads with the PyTorch version it was built
    for. If it doesn't load, fall back to mslk's python-only mode: the default
    attention backends (FlashAttention-3, PyTorch's own kernels, Triton) don't
    need the native library.
    """
    if importlib.util.find_spec("mslk") is None:
        return ModuleNotFoundError("No module named 'mslk'")
    try:
        import mslk  # noqa: F401

        return None
    except Exception as e:
        if os.environ.get("MSLK_PYTHON_ONLY") == "1":
            return e
        native_error = e
    # A failed import can leave submodules behind (e.g. mslk.version)
    for name in [m for m in sys.modules if m == "mslk" or m.startswith("mslk.")]:
        del sys.modules[name]
    os.environ["MSLK_PYTHON_ONLY"] = "1"
    try:
        import mslk  # noqa: F401, F811
    except Exception as e:
        return e
    logger.warning(
        "WARNING[XFORMERS]: mslk's native library could not be loaded, probably "
        "because it was built for a different version of PyTorch, so mslk is "
        "running in python-only mode. The default attention backends are "
        "unaffected on CUDA, but ops that need mslk's native kernels (e.g. "
        "MemoryEfficientAttentionCutlassBlackwellOp) won't work.\n"
        f"  See {_MSLK_COMPAT_URL}\n  Error: {native_error}"
    )
    return None


# The names that need mslk, and the module (relative to this one) they come from
_MSLK_NAMES = {
    "AttentionBias": ".fmha",
    "AttentionMask": ".fmha",
    "AttentionOp": ".fmha",
    "AttentionOpBase": ".fmha",
    "LowerTriangularMask": ".fmha",
    "memory_efficient_attention": ".fmha",
    "memory_efficient_attention_backward": ".fmha",
    "memory_efficient_attention_forward": ".fmha",
    "memory_efficient_attention_forward_requires_grad": ".fmha",
    "MemoryEfficientAttentionCkOp": ".fmha",
    "MemoryEfficientAttentionCutlassFwdFlashBwOp": ".fmha",
    "MemoryEfficientAttentionCutlassOp": ".fmha",
    "MemoryEfficientAttentionFlashAttentionOp": ".fmha",
    "MemoryEfficientAttentionSplitKCkOp": ".fmha",
    "rope_padded": ".rope_padded",
}
_MSLK_ALIASES = {"AttentionMask": "AttentionBias"}
# Without this annotation, which binds nothing at runtime, type checkers would
# resolve `xformers.ops.rope_padded` to the submodule of the same name
rope_padded: Any

_MSLK_ERROR = _import_mslk()
if _MSLK_ERROR is None:
    try:
        _mslk_objects = {
            name: getattr(
                importlib.import_module(module, __name__),
                _MSLK_ALIASES.get(name, name),
            )
            for name, module in _MSLK_NAMES.items()
        }
    except Exception as e:
        # E.g. this version of mslk lacks something that xformers re-exports
        _MSLK_ERROR = e
    else:
        # Only bind the names if all of them could be imported
        globals().update(_mslk_objects)
_HAS_MSLK = _MSLK_ERROR is None


def _mslk_unavailable_message() -> str:
    if torch.version.cuda is not None:
        index = "cu" + torch.version.cuda.replace(".", "")
    elif torch.version.hip is not None:
        index = "rocm" + ".".join(torch.version.hip.split(".")[:2])
    else:
        index = "<cuXXX or rocmX.Y, matching your PyTorch build>"
    return (
        "xformers.ops.memory_efficient_attention and the related symbols are "
        f"unavailable because mslk could not be imported: {_MSLK_ERROR}\n"
        "  Install mslk from the PyTorch index matching your PyTorch build, e.g.:\n"
        f"    pip install -U mslk --index-url https://download.pytorch.org/whl/{index}\n"
        f"  See {_MSLK_COMPAT_URL}"
    )


if not _HAS_MSLK:
    logger.warning(f"WARNING[XFORMERS]: {_mslk_unavailable_message()}")


def __getattr__(name: str) -> Any:
    # Only called for names that don't exist, i.e. never for the names that need
    # mslk when it's working. Raises AttributeError, like a missing name, so that
    # hasattr() keeps working. Type checkers also use it for the names bound
    # dynamically above, which they can't see.
    if name in _MSLK_NAMES and not _HAS_MSLK:
        raise AttributeError(_mslk_unavailable_message())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def masked_matmul(a, b, mask=None):
    if torch.overrides.has_torch_function((a, b, mask)):
        return torch.overrides.handle_torch_function(
            masked_matmul, (a, b, mask), a, b, mask
        )

    att = a @ b

    if mask is None:
        return att

    if mask.dtype == torch.bool:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).expand(att.shape[0], -1, -1)
        # mask is presumed false == ignore
        att[~mask] = float("-inf")
    else:
        # mask is presumed additive
        att += mask
    return att


__all__ = [
    # fmha and rope_padded, if available, so that `import *` works without them
    *(_MSLK_NAMES if _HAS_MSLK else ()),
    # indexing
    "index_select_cat",
    "scaled_index_add",
    # modpar_layers
    "ColumnParallelLinear",
    "RowParallelLinear",
    # rmsnorm
    "RMSNorm",
    # seqpar
    "sequence_parallel_leading_matmul",
    "sequence_parallel_trailing_matmul",
    # sequence_parallel_fused_ops
    "fused_allgather_and_anything",
    "fused_allgather_and_linear",
    "fused_anything_and_reducescatter",
    "fused_linear_and_reducescatter",
    # swiglu_op
    "SwiGLU",
    "SwiGLUEagerOp",
    "SwiGLUOp",
    "SwiGLUOpDispatch",
    "swiglu",
    # tiled_matmul
    "tiled_matmul",
    # unbind
    "get_stack_strides",
    "stack_or_none",
    "unbind",
    # .
    "masked_matmul",
]
