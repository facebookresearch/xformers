# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: Reusing a lot of code from the Timm repo
# main difference is probably the handling of deepnorm init, and adapting to some xformers specificities
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import logging
import math
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
from torch.nn.init import (
    _calculate_fan_in_and_fan_out,
    _no_grad_trunc_normal_,
    _no_grad_uniform_,
)

logger = logging.getLogger("xformers")


_assert_if_not_initialized = False


class xFormerWeightInit(str, Enum):
    Timm = "timm"
    ViT = "vit"
    Moco = "moco"
    Small = "small"


def get_weight_init_fn(init_choice: xFormerWeightInit):
    """
    Provide the xFormers factory with weight init routines.

    Supported initializations are:
    - Small: follow the method outlined in `Transformer Without Tears`_
    - ViT: follow the initialization in the reference ViT_ codebase
    - Timm: follow the initialization in the reference Timm_ codebase
    - Moco: follow the initialization in the reference MocoV3_ codebase

    .. _ViT: https://github.com/google-research/vision_transformer
    .. _Timm: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    .. _MocoV3: https://github.com/facebookresearch/moco-v3
    """
    return {
        xFormerWeightInit.Timm: _init_weights_vit_timm,
        xFormerWeightInit.ViT: _init_weights_vit_jax,
        xFormerWeightInit.Moco: _init_weights_vit_moco,
        xFormerWeightInit.Small: _init_weights_small,
    }[init_choice]


# Define pattern matches
def is_ffn(n):
    return "feedforward" in n or ("wrap_ff" in n and not n.endswith("norm"))


def is_mha_input_projection(n):
    return "q_proj" in n or "k_proj" in n or "v_proj" in n


# Define distribution helpers
def _small_init_(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    r"""Fills the input `Tensor` with values according to the method
    described in `Transformer Without Tears`_, using a uniform distribution.

    This is a variation of the Xavier init. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + 4 * \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    .. _`Transformer Without Tears`: https://arxiv.org/abs/1910.05895

    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def _lecun_normal(tensor, gain=1.0):
    fan_in, _ = _calculate_fan_in_and_fan_out(tensor)
    denom = fan_in
    variance = gain / denom

    # constant is stddev of standard normal truncated to (-2, 2)
    _no_grad_trunc_normal_(
        tensor,
        mean=0.0,
        std=math.sqrt(variance) / 0.87962566103423978,
        a=-2.0,
        b=2.0,
    )


# Helpers to keep all the functions typesafe, and handle corner cases and common behaviours in one place
def _maybe_init_tensor(module: nn.Module, attr: str, distribution_: Callable, **kwargs):
    #  Small helper to catch all the corner cases, while staying type safe
    if hasattr(module, attr):
        maybe_tensor = getattr(module, attr)
        if maybe_tensor is not None and isinstance(maybe_tensor, torch.Tensor):
            distribution_(maybe_tensor, **kwargs)


def _maybe_report_no_init(module, name):
    if len(list(module.named_children())) == 0 and (
        hasattr(module, "weight") or hasattr(module, "bias")
    ):
        # Skip layer norm, this is ok
        if isinstance(module, torch.nn.LayerNorm):
            return

        # Skip nn.Embedding, we typically initialize it one level up, else Pytorch has a valid default
        if isinstance(module, torch.nn.Embedding):
            return

        # This is unexpected, warn about a possible unhandled weight
        logger.warning(
            f"Not initializing weights in {name}, this could be a mistake.\nModule {module}"
        )

        if _assert_if_not_initialized:
            assert False, (
                f"Uninitialized weight found in {module}."
                + " If you have a custom module, please provide a `init_weights()` method"
            )


# Define the different initialization schemes
def _init_weights_vit_jax(
    module: nn.Module,
    name: str = "",
    head_bias: float = 0.0,
    gain: float = 1.0,
    deepnorm_style: bool = False,
    **kwargs,
):
    """ViT weight initialization, matching JAX (Flax) impl"""

    if is_ffn(name):
        _maybe_init_tensor(module, "bias", nn.init.normal_, std=1e-6)
        _maybe_init_tensor(module, "weight", torch.nn.init.xavier_uniform_, gain=gain)

    elif is_mha_input_projection(name) or isinstance(module, nn.Linear):
        if deepnorm_style and (
            "q_proj" in name.split(".") or "k_proj" in name.split(".")
        ):
            gain = 1.0

        _maybe_init_tensor(module, "weight", torch.nn.init.xavier_uniform_, gain=gain)
        _maybe_init_tensor(module, "bias", nn.init.zeros_)

    elif isinstance(module, nn.Conv2d):
        _maybe_init_tensor(module, "weight", _lecun_normal, gain=gain)
        _maybe_init_tensor(module, "bias", nn.init.zeros_)

    elif hasattr(module, "init_weights"):
        module.init_weights()  # type: ignore

    else:
        _maybe_report_no_init(module, name)

    # Recurse over the children, if the weight init is being handled here
    if not hasattr(module, "init_weights"):
        for child_name, child_module in module.named_children():
            _init_weights_vit_jax(child_module, f"{name}.{child_name}", head_bias, gain)


def _init_weights_vit_moco(
    module: nn.Module,
    name: str = "",
    gain: float = 1.0,
    **kwargs,
):
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""

    assert (
        "deepnorm_style" not in kwargs.keys()
    ), "This initialization method does not support deepnorm"

    if is_ffn(name):
        _maybe_init_tensor(module, "weight", torch.nn.init.xavier_uniform_, gain=gain)
        _maybe_init_tensor(module, "bias", nn.init.zeros_)

    elif is_mha_input_projection(name) or isinstance(module, nn.Linear):
        if isinstance(module.weight, torch.Tensor):
            val = (
                math.sqrt(6.0 / float(module.weight.shape[0] + module.weight.shape[1]))
                * gain
            )
            _maybe_init_tensor(module, "weight", nn.init.uniform_, a=-val, b=val)

        _maybe_init_tensor(module, "bias", nn.init.zeros_)

    elif hasattr(module, "init_weights"):
        module.init_weights(gain=gain)  # type: ignore

    else:
        _maybe_report_no_init(module, name)

    # Recurse over the children, if the weight init is being handled here
    if not hasattr(module, "init_weights"):
        for child_name, child_module in module.named_children():
            _init_weights_vit_moco(child_module, child_name, gain)


def _init_weights_small(
    module: nn.Module,
    name: str = "",
    head_bias: float = 0.0,
    gain: float = 1.0,
    deepnorm_style: bool = False,
    **kwargs,
):
    """Follow the `Transformer Without Tears`_ initialization for self-attention"""

    if is_ffn(name):
        _maybe_init_tensor(module, "weight", torch.nn.init.xavier_uniform_, gain=gain)
        _maybe_init_tensor(module, "bias", nn.init.normal_, std=1e-6)

    elif is_mha_input_projection(name) or isinstance(module, nn.Linear):
        # "small init" only scales the attention layers init, not the FFN
        if deepnorm_style and (
            "q_proj" in name.split(".") or "k_proj" in name.split(".")
        ):
            gain = 1.0

        _maybe_init_tensor(module, "weight", _small_init_, gain=gain)
        _maybe_init_tensor(module, "bias", nn.init.zeros_)

    elif isinstance(module, nn.Conv2d):
        _maybe_init_tensor(module, "weight", _lecun_normal)
        _maybe_init_tensor(module, "bias", nn.init.zeros_)
    elif hasattr(module, "init_weights"):
        module.init_weights()  # type: ignore
    else:
        _maybe_report_no_init(module, name)

    # Recurse over the children, if the weight init is being handled here
    if not hasattr(module, "init_weights"):
        for child_name, child_module in module.named_children():
            _init_weights_small(child_module, f"{name}.{child_name}", head_bias, gain)


def _init_weights_vit_timm(
    module: nn.Module,
    name: str = "",
    gain: float = 1.0,
    deepnorm_style: bool = False,
    **kwargs,
):
    """
    ViT weight initialization, original timm impl (for reproducibility).

    See DeepNet_ for all the DeepNorm specific codepaths
    """

    if isinstance(module, nn.Linear):
        if deepnorm_style and (
            "q_proj" in name.split(".") or "k_proj" in name.split(".")
        ):
            gain = 1

        std = 0.02 * gain
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

        _maybe_init_tensor(
            module, "weight", _no_grad_trunc_normal_, mean=0.0, std=std, a=-a, b=a
        )
        _maybe_init_tensor(module, "bias", nn.init.zeros_)

    elif hasattr(module, "init_weights"):
        module.init_weights(gain=gain)  # type: ignore
    else:
        _maybe_report_no_init(module, name)

    # Recurse over the children, if the weight init is being handled here
    if not hasattr(module, "init_weights"):
        for child_name, child_module in module.named_children():
            _init_weights_vit_timm(child_module, child_name, gain)
