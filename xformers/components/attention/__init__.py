from pathlib import Path
from typing import Any, Callable, Dict, Set

import torch

from xformers.utils import get_registry_decorator, import_all_modules

from ._sputnik_sparse import SparseCS
from .base import Attention, AttentionConfig  # noqa

# Credits: Classy Vision registry mechanism

ATTENTION_REGISTRY: Dict[str, Any] = {}
ATTENTION_CLASS_NAMES: Set[str] = set()

# Arbitrary threshold for now,
# in between dense and sparse matrix algorithms for the attention mechanism
_DENSITY_THRESHOLD = 0.30  # noqa # from the sputnik paper, vs.
_USE_SPUTNIK = True


def build_attention(config: AttentionConfig):
    """Builds an attention from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_attention",
    "foo": "bar"}` will find a class that was registered as "my_attention"
    (see :func:`register_attention`) and call .from_config on it."""

    return ATTENTION_REGISTRY[config.name].constructor.from_config(config)


"""Registers an Attention subclass.

    This decorator allows xFormers to instantiate a subclass of Attention
    from a configuration file, even if the class itself is not part of the
    xFormers library. To use it, apply this decorator to an Attention
    subclass, like this:

    .. code-block:: python
        @dataclass
        class MyConfig:
            ...

        @register_attention('my_attention', MyConfig)
        class MyAttention(Attention):
            ...

    To instantiate an attention from a configuration file, see :func:`build_attention`."""
register_attention: Callable[[str, Any], Callable[[Any], Any]] = get_registry_decorator(
    ATTENTION_REGISTRY, ATTENTION_CLASS_NAMES, Attention, AttentionConfig
)


def maybe_sparsify(matrix):
    # Sparsify if that makes sense
    if torch.count_nonzero(matrix).item() / matrix.numel() > _DENSITY_THRESHOLD:
        return matrix

    return sparsify(matrix)


def sparsify(matrix):
    if _USE_SPUTNIK:
        return SparseCS(matrix)
    return matrix.to_sparse()


from .favor import FavorAttention  # noqa
from .global_tokens import GlobalAttention  # noqa
from .linformer import LinformerAttention  # noqa
from .local import LocalAttention  # noqa
from .nystrom import NystromAttention  # noqa
from .random import RandomAttention  # noqa
from .scaled_dot_product import ScaledDotProduct  # noqa

__all__ = [
    "ScaledDotProduct",
    "LocalAttention",
    "LinformerAttention",
    "NystromAttention",
    "RandomAttention",
    "GlobalAttention",
    "FavorAttention",
    "Attention",
    "build_attention",
    "register_attention",
]

# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components.attention")
