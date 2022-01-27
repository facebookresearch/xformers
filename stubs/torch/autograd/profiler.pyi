# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from typing import Any

class record_function(contextlib.ContextDecorator):
    def __init__(self, name: str) -> None: ...
    def __enter__(self) -> Any: ...
    def __exit__(self, *exctype: Any) -> None: ...
