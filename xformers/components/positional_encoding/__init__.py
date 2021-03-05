from pathlib import Path
from typing import Any, Dict

from xformers.utils import import_all_modules

from .base import PositionEncoding, PositionEncodingConfig  # noqa

# Credits: Classy Vision registry mechanism

POSITION_ENCODING_REGISTRY: Dict[str, Any] = {}
POSITION_ENCODING_CLASS_NAMES = set()


def build_positional_encoding(config: PositionEncodingConfig):
    """Builds a position encoding from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_position_encoding",
    "foo": "bar"}` will find a class that was registered as "my_position_encoding"
    (see :func:`register_positional_encoding`) and call .from_config on it."""

    return POSITION_ENCODING_REGISTRY[config.name].from_config(config)


def register_positional_encoding(name):
    """Registers a PositionEncoding subclass.

    This decorator allows xFormers to instantiate a subclass of PositionEncoding
    from a configuration file, even if the class itself is not part of the
    xFormers framework. To use it, apply this decorator to a `PositionEncoding`
    subclass, like this:

    .. code-block:: python

        @register_positional_encoding('my_encoding')
        class MyEncoding(PositionEncoding):
            ...

    To instantiate a position encoding from a configuration file, see :func:`build_positional_encoding`."""

    def register_positional_encoding_cls(cls):
        if name in POSITION_ENCODING_REGISTRY:
            raise ValueError("Cannot register duplicate attention ({})".format(name))

        if not issubclass(cls, PositionEncoding):
            raise ValueError(
                "Feedforward ({}: {}) must extend the base Feedforward class".format(
                    name, cls.__name__
                )
            )

        if cls.__name__ in POSITION_ENCODING_CLASS_NAMES:
            raise ValueError(
                "Cannot register attention with duplicate class name ({})".format(
                    cls.__name__
                )
            )

        POSITION_ENCODING_REGISTRY[name] = cls
        POSITION_ENCODING_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_positional_encoding_cls


from .sine import SinePositionEncoding  # noqa

__all__ = [
    "SinePositionEncoding",
    "build_positional_encoding",
    "register_positional_encoding",
]

# automatically import any Python files in the directory
import_all_modules(
    str(Path(__file__).parent), "xformers.components.positional_encoding"
)
