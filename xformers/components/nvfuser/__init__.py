# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
from typing import Any, Callable, Dict, Set, Union

from xformers import _is_functorch_available
from xformers.utils import (
    generate_matching_config,
    get_registry_decorator,
    import_all_modules,
)

from .base import Fused, FusedConfig  # noqa

# CREDITS: Classy Vision registry mechanism

FUSED_REGISTRY: Dict[str, Any] = {}
FUSED_CLASS_NAMES: Set[str] = set()


def build_fused(config: Union[Dict[str, Any], FusedConfig]):
    """Builds a fused operation from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_fused",
    "foo": "bar"}` will find a class that was registered as "my_fused"
    (see :func:`register_fused`) and call .from_config on it."""

    if not isinstance(config, FusedConfig):
        config_instance = generate_matching_config(
            config, FUSED_REGISTRY[config["name"]].config
        )
    else:
        config_instance = config

    return FUSED_REGISTRY[config_instance.name].constructor.from_config(config_instance)


"""Registers a Fused subclass.

    This decorator allows xFormers to instantiate a subclass of Fused
    from a configuration file, even if the class itself is not part of the
    xFormers framework. To use it, apply this decorator to a Fused
    subclass, like this:

    .. code-block:: python

        @dataclass
        class MyConfig:
            ...

        @register_fused('my_fused', MyConfig)
        class MyFused(Fused):
            ...

    To instantiate a fused operation from a configuration file, see :func:`build_fused`."""
register_fused: Callable[[str, Any], Callable[[Any], Any]] = get_registry_decorator(
    FUSED_REGISTRY, FUSED_CLASS_NAMES, Fused, FusedConfig
)


__all__ = []

if _is_functorch_available:
    __all__ += ["fused_bias_relu_dropout"]

# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components.nvfuser")
