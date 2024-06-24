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

from .base import Feedforward, FeedforwardConfig  # noqa

# CREDITS: Classy Vision registry mechanism

FEEDFORWARD_REGISTRY: Dict[str, Any] = {}
FEEDFORWARD_CLASS_NAMES: Set[str] = set()


def build_feedforward(config: Union[Dict[str, Any], FeedforwardConfig]):
    """Builds a feedforward from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_feedforward",
    "foo": "bar"}` will find a class that was registered as "my_feedforward"
    (see :func:`register_feedforward`) and call .from_config on it."""

    if not isinstance(config, FeedforwardConfig):
        config_instance = generate_matching_config(
            config, FEEDFORWARD_REGISTRY[config["name"]].config
        )
    else:
        config_instance = config

    return FEEDFORWARD_REGISTRY[config_instance.name].constructor.from_config(
        config_instance
    )


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
