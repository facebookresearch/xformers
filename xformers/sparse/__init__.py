# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import logging

try:
    from .blocksparse_tensor import BlockSparseTensor  # noqa: F401
except ImportError as e:
    logging.warning(
        f"Triton is not available, some optimizations will not be enabled.\nError {e}"
    )

from .csr_tensor import SparseCSRTensor  # noqa: F401
