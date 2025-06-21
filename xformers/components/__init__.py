# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import warnings
from pathlib import Path

from xformers.utils import import_all_modules

from .attention import Attention, build_attention  # noqa
from .input_projection import InputProjection, InputProjectionConfig  # noqa
from .residual import (  # noqa
    NormalizationType,
    PostNorm,
    PreNorm,
    RequiresWrappedInputs,
    Residual,
    ResidualNormStyle,
)

warnings.warn(
    "xformers.components is deprecated and is not maintained anymore. "
    "It might be removed in a future version of xFormers ",
    FutureWarning,
    stacklevel=2,
)


# automatically import any Python files in the directory
import_all_modules(str(Path(__file__).parent), "xformers.components")
