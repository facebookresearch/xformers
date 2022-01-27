# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import Any, Callable, Dict, Set, Union

from xformers.utils import (
    generate_matching_config,
    get_registry_decorator,
    import_all_modules,
)

from .base import PositionEmbedding, PositionEmbeddingConfig  # noqa

# CREDITS: Classy Vision registry mechanism

POSITION_EMBEDDING_REGISTRY: Dict[str, Any] = {}
POSITION_EMBEDDING_CLASS_NAMES: Set[str] = set()


def build_positional_embedding(config: Union[Dict[str, Any], PositionEmbeddingConfig]):
    """Builds a position encoding from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_position_encoding",
    "foo": "bar"}` will find a class that was registered as "my_position_encoding"
    (see :func:`register_positional_embedding`) and call .from_config on it."""

    if not isinstance(config, PositionEmbeddingConfig):
        config_instance = generate_matching_config(
            config, POSITION_EMBEDDING_REGISTRY[config["name"]].config
        )
    else:
        config_instance = config

    return POSITION_EMBEDDING_REGISTRY[config_instance.name].constructor.from_config(
        config_instance
    )


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


from .rotary import RotaryEmbedding  # noqa
from .sine import SinePositionalEmbedding  # type: ignore  # noqa
from .vocab import VocabEmbedding  # noqa

__all__ = [
    "RotaryEmbedding",
    "SinePositionalEmbedding",
    "VocabEmbedding",
    "build_positional_embedding",
    "register_positional_embedding",
]

# automatically import any Python files in the directory
import_all_modules(
    str(Path(__file__).parent), "xformers.components.positional_embedding"
)
