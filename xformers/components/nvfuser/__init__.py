# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


from xformers import _is_functorch_available

if _is_functorch_available:  # noqa
    try:
        from .bias_act_dropout import NVFusedBiasActivationDropout  # noqa
        from .bias_dropout_res import NVFusedBiasDropoutRes  # noqa
        from .bias_dropout_res_layernorm import NVFusedBiasDropoutResLayerNorm  # noqa

        __all__ = [
            "NVFusedBiasActivationDropout",
            "NVFusedBiasDropoutResLayerNorm",
            "NVFusedBiasDropoutRes",
        ]
    except ImportError:
        __all__ = []
