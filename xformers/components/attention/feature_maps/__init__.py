# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from enum import Enum

from .base import FeatureMap, FeatureMapConfig
from .softmax import NormDistribution, SMHyperbolic, SMOrf, SMReg


class FeatureMapType(str, Enum):
    SMOrf = "sm_orf"
    SMHyp = "sm_hyp"
    SMReg = "sm_reg"  # regularized softmax kernel


__all__ = [
    "SMOrf",
    "SMReg",
    "SMHyperbolic",
    "NormDistribution",
    "FeatureMapConfig",
    "FeatureMap",
]
