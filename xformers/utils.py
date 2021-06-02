import importlib
import os
import sys
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, List


# credit: snippet used in ClassyVision (and probably other places)
def import_all_modules(root: str, base_module: str) -> List[str]:
    modules: List[str] = []
    for file in os.listdir(root):
        if file.endswith((".py", ".pyc")) and not file.startswith("_"):
            module = file[: file.find(".py")]
            if module not in sys.modules:
                module_name = ".".join([base_module, module])
                importlib.import_module(module_name)
                modules.append(module_name)

    return modules


def get_registry_decorator(
    class_registry, name_registry, reference_class
) -> Callable[[str], Callable[[Any], Any]]:
    def register_item(name):
        """Registers a subclass.

        This decorator allows xFormers to instantiate a given subclass
        from a configuration file, even if the class itself is not part of the
        xFormers library."""

        def register_cls(cls):
            if name in class_registry:
                raise ValueError("Cannot register duplicate item ({})".format(name))
            if not issubclass(cls, reference_class):
                raise ValueError(
                    "Item ({}: {}) must extend the base class: {}".format(
                        name, cls.__name__, reference_class.__name__
                    )
                )
            if cls.__name__ in name_registry:
                raise ValueError(
                    "Cannot register item with duplicate class name ({})".format(
                        cls.__name__
                    )
                )

            class_registry[name] = cls
            name_registry.add(cls.__name__)
            return cls

        return register_cls

    return register_item


@dataclass(init=False)
class ExtensibleConfig:
    def __init__(self, *_, **kwargs):
        """ Accept any extra keyword at construction time, and store them """
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_other(cls, config: "ExtensibleConfig"):
        """Given another config -could be subtyped and not completely compatible-
        try to fill in the compatible fields"""

        names = {f.name for f in fields(cls)}
        kwargs = {
            n: getattr(config, n) for n in filter(lambda x: hasattr(config, x), names)
        }
        return cls(**kwargs)

    @classmethod
    def as_patchy_dict(cls, config: "ExtensibleConfig") -> Dict[str, Any]:
        """Return a dict which covers all the set fields in this class.
        .. warning: Note that this could be incomplete given the config definition"""

        names = {f.name for f in fields(cls)}
        return {
            n: getattr(config, n) for n in filter(lambda x: hasattr(config, x), names)
        }
