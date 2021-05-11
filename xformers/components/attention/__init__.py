from pathlib import Path
from typing import Any, Dict

import torch

from xformers.utils import import_all_modules

from ._sputnik_sparse import SparseCS
from .base import Attention, AttentionConfig  # noqa

# Credits: Classy Vision registry mechanism

ATTENTION_REGISTRY: Dict[str, Any] = {}
ATTENTION_CLASS_NAMES = set()

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

    return ATTENTION_REGISTRY[config.name].from_config(config)


def register_attention(name):
    """Registers an Attention subclass.

    This decorator allows xFormers to instantiate a subclass of Attention
    from a configuration file, even if the class itself is not part of the
    xFormers library. To use it, apply this decorator to an Attention
    subclass, like this:

    .. code-block:: python

        @register_attention('my_attention')
        class MyAttention(Attention):
            ...

    To instantiate an attention from a configuration file, see :func:`build_attention`."""

    def register_attention_cls(cls):
        if name in ATTENTION_REGISTRY:
            raise ValueError("Cannot register duplicate attention ({})".format(name))
        if not issubclass(cls, Attention):
            raise ValueError(
                "Attention ({}: {}) must extend the base Attention class".format(
                    name, cls.__name__
                )
            )
        if cls.__name__ in ATTENTION_CLASS_NAMES:
            raise ValueError(
                "Cannot register attention with duplicate class name ({})".format(
                    cls.__name__
                )
            )

        ATTENTION_REGISTRY[name] = cls
        ATTENTION_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_attention_cls


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
