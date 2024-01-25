# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# One reason this module is called `_triton` instead of just `triton` is this:
# https://github.com/openai/triton/commit/c6040bcbd8a046785462481b2830b3fff5fc4aab

from typing import TYPE_CHECKING

import xformers

if TYPE_CHECKING or xformers._is_triton_available():
    from .k_index_select_cat import index_select_cat_bwd, index_select_cat_fwd
    from .k_scaled_index_add import scaled_index_add_bwd, scaled_index_add_fwd
else:
    index_select_cat_fwd = index_select_cat_bwd = None
    scaled_index_add_fwd = scaled_index_add_bwd = None
