/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
#if !defined(USE_ROCM)
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_small_k(Tensor query, Tensor key, Tensor value, "
      "bool compute_logsumexp, Tensor? attn_bias, float p) -> (Tensor, Tensor, int, int)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_cutlass(Tensor query, Tensor key, Tensor value, "
      "Tensor? attn_bias, Tensor? seqstart_q, Tensor? seqstart_k, int? max_seqlen_q, float "
      "dropout_p, bool compute_logsumexp, int custom_mask_type, float? scale, Tensor? seqlen_k, "
      "int? window_size) -> (Tensor, Tensor, int, int)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder(Tensor query, Tensor "
      "key, Tensor value, Tensor seq_positions, float scale) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward_small_k(Tensor grad_out, Tensor query, Tensor key, "
      "Tensor value, Tensor logsumexp, Tensor output, Tensor? attn_bias, float p, int rng_seed, "
      "int rng_offset) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward_cutlass(Tensor grad_out, Tensor query, Tensor key, "
      "Tensor value, Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, int max_seqlen_q, "
      "int max_seqlen_k, Tensor logsumexp, Tensor output, float dropout_p, int rng_seed, int "
      "rng_offset, int custom_mask_type, float? scale, int num_splits_key, int? window_size) -> "
      "(Tensor, Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_temp_dropout(Tensor out, float p) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_cutlass_rand_uniform(float p, Tensor out) -> Tensor"));
#endif
#if defined(USE_ROCM)
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_ck(Tensor query, "
      "Tensor key, Tensor value, Tensor? attn_bias, Tensor? seqstart_q, "
      "Tensor? seqstart_k, int? max_seqlen_q, float dropout_p, "
      "bool compute_logsumexp, int custom_mask_type, float? scale, Tensor? seqlen_k, int? window_size) -> (Tensor, Tensor, int, int)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder_ck(Tensor query, "
      "Tensor key, Tensor value, Tensor? seq_positions, float scale) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder_splitk_ck(Tensor query, Tensor key, "
      " Tensor value, Tensor? seq_positions, float scale, int split_k) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward_ck(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor? attn_bias, Tensor? seqstart_q, Tensor? seqstart_k, int? max_seqlen_q, Tensor? seqlen_k, Tensor logsumexp, Tensor output, float dropout_p, int rng_seed, int rng_offset, int custom_mask_type, float? scale) -> (Tensor, Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_ck_rand_uniform(float p, Tensor out) -> Tensor"));
#endif
}
