# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum


class Activation(str, Enum):
    GeLU = "gelu"
    ReLU = "relu"
