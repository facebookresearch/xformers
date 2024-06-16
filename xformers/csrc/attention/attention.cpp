/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <torch/types.h>

// If we are in a Windows environment, we need to define
// initialization functions for the _custom_ops extension.
// For PyMODINIT_FUNC to work, we need to include Python.h
// https://github.com/pytorch/vision/blob/main/torchvision/csrc/vision.cpp#L17
// Fixes error LNK2001: unresolved external symbol PyInit__C
#if defined(_WIN32)
#include <Python.h>
PyMODINIT_FUNC PyInit__C(void) {
  // No need to do anything.
  return NULL;
}
#endif // defined(_WIN32)

TORCH_LIBRARY_FRAGMENT(xformers, m) {
#if !defined(USE_ROCM)
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_small_k(Tensor query, Tensor key, Tensor value, "
      "bool compute_logsumexp, Tensor? attn_bias, float p) -> (Tensor, Tensor, int, int)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_cutlass(Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, int? max_seqlen_q, int? max_seqlen_k, float dropout_p, int custom_mask_type, bool compute_log_sumexp=False, *, float? scale=None, Tensor? seqlen_k=None, int? window_size=None) -> (Tensor output, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, SymInt max_seqlen_batch_q, SymInt max_seqlen_batch_k)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder(Tensor query, Tensor "
      "key, Tensor value, Tensor seq_positions, float scale) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward_small_k(Tensor grad_out, Tensor query, Tensor key, "
      "Tensor value, Tensor logsumexp, Tensor output, Tensor? attn_bias, float p, int rng_seed, "
      "int rng_offset) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward_cutlass(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q, SymInt max_seqlen_k, Tensor logsumexp, float dropout_p, Tensor philox_seed, Tensor philox_offset, int custom_mask_type, bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None, int? window_size=None) -> (Tensor, Tensor, Tensor, Tensor)"));
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
      "xformers::efficient_attention_backward_ck(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor? attn_bias, Tensor? seqstart_q, Tensor? seqstart_k, int? max_seqlen_q, int? max_seqlen_k, Tensor? seqlen_k, Tensor logsumexp, Tensor output, float dropout_p, int rng_seed, int rng_offset, int custom_mask_type, float? scale, int? window_size) -> (Tensor, Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_ck_rand_uniform(float p, Tensor out) -> Tensor"));
#endif
}
