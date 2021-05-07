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
