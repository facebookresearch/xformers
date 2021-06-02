from pathlib import Path
from typing import Any, Callable, Dict, Set

from xformers.utils import get_registry_decorator, import_all_modules

from .base import PositionEmbedding, PositionEmbeddingConfig  # noqa

# Credits: Classy Vision registry mechanism

POSITION_EMBEDDING_REGISTRY: Dict[str, Any] = {}
POSITION_EMBEDDING_CLASS_NAMES: Set[str] = set()


def build_positional_embedding(config: PositionEmbeddingConfig):
    """Builds a position encoding from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_position_encoding",
    "foo": "bar"}` will find a class that was registered as "my_position_encoding"
    (see :func:`register_positional_embedding`) and call .from_config on it."""

    return POSITION_EMBEDDING_REGISTRY[config.name].constructor.from_config(config)


"""Registers a PositionEncoding subclass.

    This decorator allows xFormers to instantiate a subclass of PositionEncoding
    from a configuration file, even if the class itself is not part of the
    xFormers framework. To use it, apply this decorator to a `PositionEncoding`
    subclass, like this:

    .. code-block:: python
        @dataclass
        class MyConfig:
            ...

        @register_positional_embedding('my_encoding', MyConfig)
        class MyEncoding(PositionEncoding):
            ...

    To instantiate a position encoding from a configuration file, see :func:`build_positional_embedding`."""
register_positional_embedding: Callable[
    [str, Any], Callable[[Any], Any]
] = get_registry_decorator(
    POSITION_EMBEDDING_REGISTRY,
    POSITION_EMBEDDING_CLASS_NAMES,
    PositionEmbedding,
    PositionEmbeddingConfig,
)


from .sine import SinePositionalEmbedding  # noqa
from .vocab import VocabEmbedding  # noqa

__all__ = [
    "SinePositionalEmbedding",
    "VocabEmbedding",
    "build_positional_embedding",
    "register_positional_embedding",
]

# automatically import any Python files in the directory
import_all_modules(
    str(Path(__file__).parent), "xformers.components.positional_embedding"
)
