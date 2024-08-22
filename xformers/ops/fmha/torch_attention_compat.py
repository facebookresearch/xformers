# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
from torch._C import parse_schema

try:
    # This function was added in https://github.com/pytorch/pytorch/pull/131894
    # (which hadn't landed yet at the time of writing), thus will only arrive in
    # PyTorch 2.5+. In the meantime we need a fallback.
    from torch.modules.cuda import is_flash_attention_available
except ImportError:

    def is_flash_attention_available():
        return sys.platform == "linux"


def is_pt_cutlass_compatible(force: bool) -> bool:
    compatible = True

    fwd_schema_str = (
        "aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value, "
        "Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt? max_seqlen_q, "
        "SymInt? max_seqlen_k, float dropout_p, int custom_mask_type, bool compute_log_sumexp=False, *, "
        "float? scale=None, Tensor? seqlen_k=None, int? window_size=None) -> "
        "(Tensor output, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, "
        "SymInt max_seqlen_batch_q, SymInt max_seqlen_batch_k)"
    )
    expected_fwd_schema = parse_schema(fwd_schema_str)

    current_schema = torch.ops.aten._efficient_attention_forward.default._schema
    if not current_schema.is_backward_compatible_with(expected_fwd_schema):
        compatible = False

        if force:
            raise ImportError(
                f"Current Torch CUTLASS doesnt have a compatible aten::_efficient_attention_forward schema\n"
                f"EXPECTED:\n{expected_fwd_schema}\n"
                f"but GOT:\n{current_schema}"
            )

    bwd_schema_str = (
        "aten::_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, "
        "Tensor? bias, Tensor out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q, "
        "SymInt max_seqlen_k, Tensor logsumexp, float dropout_p, Tensor philox_seed, Tensor philox_offset, "
        "int custom_mask_type, bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None, "
        "int? window_size=None, bool shared_storage_dqdkdv=False) -> (Tensor, Tensor, Tensor, Tensor)"
    )

    expected_bwd_schema = parse_schema(bwd_schema_str)

    current_schema = torch.ops.aten._efficient_attention_backward.default._schema
    if not current_schema.is_backward_compatible_with(expected_bwd_schema):
        compatible = False

        if force:
            raise ImportError(
                f"Current Torch CUTLASS doesnt have a compatible aten::_efficient_attention_backward schema\n"
                f"EXPECTED:\n{expected_bwd_schema}\n"
                f"but GOT:\n{current_schema}"
            )

    return compatible


def is_pt_flash_compatible(force: bool) -> bool:
    if not is_flash_attention_available():
        if force:
            raise ImportError("Flash SDP backend is disabled")
        return False

    if not hasattr(torch.nn, "attention") or not hasattr(
        torch.nn.attention, "_get_flash_version"
    ):
        if force:
            raise ImportError(
                f"Current Torch {torch.__version__} doesnt implement "
                "torch.nn.attention._get_flash_version()"
            )
        return False

    FLASH_VERSION = torch.nn.attention._get_flash_version()

    compatible = True

    fwd_schema_str = (
        "aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value, "
        "Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, "
        "bool is_causal, bool return_debug_mask, *, float? scale=None, "
        "SymInt? window_size_left=None, SymInt? window_size_right=None, "
        "Tensor? seqused_k=None, Tensor? alibi_slopes=None) -> (Tensor output, Tensor softmax_logsumexp, "
        "Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)"
    )
    expected_fwd_schema = parse_schema(fwd_schema_str)

    current_schema = torch.ops.aten._flash_attention_forward.default._schema
    if not current_schema.is_backward_compatible_with(expected_fwd_schema):
        compatible = False

        if force:
            raise ImportError(
                f"Current Torch with Flash-Attention {FLASH_VERSION} doesnt have "
                "a compatible aten::_flash_attention_forward schema\n"
                f"EXPECTED:\n{expected_fwd_schema}\n"
                f"but GOT:\n{current_schema}"
            )

    bwd_schema_str = (
        "aten::_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, "
        "Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, "
        "float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None, "
        "SymInt? window_size_left=None, SymInt? window_size_right=None) -> (Tensor, Tensor, Tensor)"
    )

    expected_bwd_schema = parse_schema(bwd_schema_str)

    current_schema = torch.ops.aten._flash_attention_backward.default._schema
    if not current_schema.is_backward_compatible_with(expected_bwd_schema):
        compatible = False

        if force:
            raise ImportError(
                f"Current Torch with Flash-Attention {FLASH_VERSION} doesnt have "
                "a compatible aten::_flash_attention_backward schema\n"
                f"EXPECTED:\n{expected_bwd_schema}\n"
                f"but GOT:\n{current_schema}"
            )

    return compatible
