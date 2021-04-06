from pathlib import Path
from typing import Any, Dict

from xformers.components import Activation  # noqa
from xformers.utils import import_all_modules

from .base import Feedforward, FeedforwardConfig  # noqa

# Credits: Classy Vision registry mechanism

FEEDFORWARD_REGISTRY: Dict[str, Any] = {}
FEEDFORWARD_CLASS_NAMES = set()


def build_feedforward(config: FeedforwardConfig):
    """Builds a feedforward from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_feedforward",
    "foo": "bar"}` will find a class that was registered as "my_feedforward"
    (see :func:`register_feedforward`) and call .from_config on it."""

    return FEEDFORWARD_REGISTRY[config.name].from_config(config)


def register_feedforward(name):
    """Registers a Feedforward subclass.

    This decorator allows xFormers to instantiate a subclass of Feedforward
    from a configuration file, even if the class itself is not part of the
    xFormers framework. To use it, apply this decorator to a Feedforward
    subclass, like this:

    .. code-block:: python

        @register_feedforward('my_ff')
        class MyFeedforward(Feedforward):
            ...

    To instantiate a feedforward from a configuration file, see :func:`build_feedforward`."""

    def register_feedforward_cls(cls):
        if name in FEEDFORWARD_REGISTRY:
            raise ValueError("Cannot register duplicate attention ({})".format(name))

        if not issubclass(cls, Feedforward):
            raise ValueError(
                "Feedforward ({}: {}) must extend the base Feedforward class".format(
                    name, cls.__name__
                )
            )

        if cls.__name__ in FEEDFORWARD_CLASS_NAMES:
            raise ValueError(
                "Cannot register attention with duplicate class name ({})".format(
                    cls.__name__
                )
            )

        FEEDFORWARD_REGISTRY[name] = cls
        FEEDFORWARD_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_feedforward_cls


from .mlp import MLP  # noqa

__all__ = [
    "MLP",
    "Feedforward",
    "build_feedforward",
    "register_feedforward",
]

# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components.feedforward")
