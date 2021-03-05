from pathlib import Path
from typing import Any, Dict

from xformers.utils import import_all_modules

from .base import Attention, AttentionConfig  # noqa

# Credits: Classy Vision registry mechanism

ATTENTION_REGISTRY: Dict[str, Any] = {}
ATTENTION_CLASS_NAMES = set()


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


from .multi_head_attention import MultiHeadAttention  # noqa

__all__ = [
    "MultiHeadAttention",
    "Attention",
    "build_attention",
    "register_attention",
]

# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components.attention")
