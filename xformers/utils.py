import importlib
import os
import sys
from typing import List


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
