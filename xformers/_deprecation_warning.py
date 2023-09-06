# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: Inspired by https://github.com/pytorch/text/blob/master/torchtext/nn/modules/multiheadattention.py
# and the MultiHeadAttention implementation from PyTorch


import warnings


def deprecated_function(self):
    name = repr(self)  # self.__name__
    msg = f"{name} is deprecated and is not maintained anymore. It might be removed in a future version of xFormers"
    warnings.warn(msg, FutureWarning, stacklevel=2)
