from pathlib import Path
from typing import Any, Callable, Dict, Set

from xformers.utils import get_registry_decorator, import_all_modules

from .base import Feedforward, FeedforwardConfig  # noqa

# Credits: Classy Vision registry mechanism

FEEDFORWARD_REGISTRY: Dict[str, Any] = {}
FEEDFORWARD_CLASS_NAMES: Set[str] = set()


def build_feedforward(config: FeedforwardConfig):
    """Builds a feedforward from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_feedforward",
    "foo": "bar"}` will find a class that was registered as "my_feedforward"
    (see :func:`register_feedforward`) and call .from_config on it."""

    return FEEDFORWARD_REGISTRY[config.name].constructor.from_config(config)


"""Registers a Feedforward subclass.

    This decorator allows xFormers to instantiate a subclass of Feedforward
    from a configuration file, even if the class itself is not part of the
    xFormers framework. To use it, apply this decorator to a Feedforward
    subclass, like this:

    .. code-block:: python
        @dataclass
        class MyConfig:
            ...

        @register_feedforward('my_ff', MyConfig)
        class MyFeedforward(Feedforward):
            ...

    To instantiate a feedforward from a configuration file, see :func:`build_feedforward`."""
register_feedforward: Callable[
    [str, Any], Callable[[Any], Any]
] = get_registry_decorator(
    FEEDFORWARD_REGISTRY, FEEDFORWARD_CLASS_NAMES, Feedforward, FeedforwardConfig
)

from .mlp import MLP  # noqa

__all__ = [
    "MLP",
    "Feedforward",
    "build_feedforward",
    "register_feedforward",
]

# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components.feedforward")
