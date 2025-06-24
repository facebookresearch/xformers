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
#if defined(USE_ROCM)
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_ck(Tensor query, "
      "Tensor key, Tensor value, Tensor? attn_bias, Tensor? seqstart_q, "
      "Tensor? seqstart_k, int? max_seqlen_q, float dropout_p, "
      "bool compute_logsumexp, int custom_mask_type, float? scale, Tensor? seqlen_k, int? window_size, Tensor? block_tables, int? page_size) -> (Tensor, Tensor?, int, int)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_forward_decoder_splitk_ck(Tensor query, Tensor key, "
      " Tensor value, Tensor? seq_positions, float scale, int split_k) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention_backward_ck(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor? attn_bias, Tensor? seqstart_q, Tensor? seqstart_k, int? max_seqlen_q, int? max_seqlen_k, Tensor? seqlen_k, Tensor logsumexp, Tensor output, float dropout_p, int rng_seed, int rng_offset, int custom_mask_type, float? scale, int? window_size) -> (Tensor, Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_ck_rand_uniform(float p, Tensor out) -> Tensor"));
#endif
}
