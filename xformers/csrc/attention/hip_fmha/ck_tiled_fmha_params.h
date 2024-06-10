/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <array>
#include <cstdint>

struct BatchedInferParams {
  int B; // batch size
  int M; // seq_len for Query
  int N; // seq_len for Key and Value
  int Hq; // number of heads for Query
  int Hkv; // number of heads for Key and Value
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  float scale;
  bool has_attn_bias;

  // BMHK mode strides
  std::array<int, 4> q_strides;
  std::array<int, 4> k_strides;
  std::array<int, 4> v_strides;
  std::array<int, 4> out_strides;
  std::array<int, 4> attn_bias_strides; // 4d tensor_view [B, H, M, N]

  // BHM mode strides, completely contiguous
  std::array<int, 3> lse_strides;

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;

  int custom_mask_type;
  int window_size; // local-attention

  void* out_ptr;
};

struct BatchedForwardParams : public BatchedInferParams {
  bool compute_logsumexp;

  float dropout_prob;
  int64_t philox_seed;
  int64_t philox_offset;

  // completely contiguous
  void* logsumexp_ptr;
};

struct GroupedInferParams {
  int num_batches;
  int M; // total seq_len for all queries in the batch
  int N; // total seq_len for all keys/values in the batch
  int Hq; // number of heads for Query
  int Hkv; // number of heads for Key and Value
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  int max_seqlen_q;

  void* seqstart_q_dev_ptr;
  void* seqstart_k_dev_ptr;
  void* seqlen_k_dev_ptr;

  float scale;
  bool has_attn_bias;

  // MHK mode strides, last-dim contiguous
  std::array<int, 3> q_strides;
  std::array<int, 3> k_strides;
  std::array<int, 3> v_strides;
  std::array<int, 3> out_strides;

  // 4d tensor view [B, H, M, N]
  std::array<int, 4> attn_bias_strides;

  // BHM mode strides, completely contiguous
  std::array<int, 3> lse_strides;

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;

  int custom_mask_type;
  int window_size; // local-attention

  void* out_ptr;
};

struct GroupedForwardParams : public GroupedInferParams {
  bool use_dropout;
  bool compute_logsumexp;

  float dropout_prob;
  int64_t philox_seed;
  int64_t philox_offset;

  // completely contiguous
  void* logsumexp_ptr;
};

struct BatchedBackwardParams {
  int B; // batch size
  int M; // seq_len for Query
  int N; // seq_len for Key and Value
  int Hq; // number of heads for Query
  int Hkv; // number of heads for Key and Value
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  float scale;
  bool has_attn_bias;
  bool bias_has_grad;

  bool is_mqa_gqa;

  // BMHK mode strides, last-dim contiguous
  std::array<int, 4> q_strides;
  std::array<int, 4> k_strides;
  std::array<int, 4> v_strides;
  std::array<int, 4> attn_bias_strides; // 4d tensor_view [B, H, M, N]
  std::array<int, 4> out_strides;
  std::array<int, 4> grad_out_strides;

  std::array<int, 4> grad_k_strides;
  std::array<int, 4> grad_v_strides;

  // BHM mode strides, completely contiguous
  std::array<int, 3> lsed_strides;

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;
  const void* grad_out_ptr;
  const void* out_ptr;

  uint8_t custom_mask_type;
  int window_size; // local-attention

  void* grad_q_ptr;
  void* grad_k_ptr;
  void* grad_v_ptr;
  void* grad_bias_ptr;

  float dropout_prob;
  int64_t philox_seed;
  int64_t philox_offset;

  // BHM mode lengths, completely contiguous
  const void* logsumexp_ptr;
  void* dot_out_ptr;
};

struct GroupedBackwardParams {
  int num_batches;
  int M; // total seq_len for all queries in the batch
  int N; // total seq_len for all keys/values in the batch
  int Hq; // number of heads for Query
  int Hkv; // number of heads for Key and Value
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  int max_seqlen_q;
  int max_seqlen_k;

  void* seqstart_q_dev_ptr;
  void* seqstart_k_dev_ptr;
  void* seqlen_k_dev_ptr;

  float scale;
  bool has_attn_bias;
  bool bias_has_grad;

  bool is_mqa_gqa;

  // MHK mode strides, last-dim contiguous
  std::array<int, 3> q_strides;
  std::array<int, 3> k_strides;
  std::array<int, 3> v_strides;
  std::array<int, 3> out_strides;
  std::array<int, 3> grad_out_strides;
  // 4d tensor view [B, H, M, N]
  std::array<int, 4> attn_bias_strides;

  std::array<int, 3> grad_k_strides;
  std::array<int, 3> grad_v_strides;

  // BHM mode strides, completely contiguous
  std::array<int, 3> lsed_strides;

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;
  const void* grad_out_ptr;
  const void* out_ptr;

  uint8_t custom_mask_type;
  int window_size; // local-attention

  void* grad_q_ptr;
  void* grad_k_ptr;
  void* grad_v_ptr;
  void* grad_bias_ptr;

  float dropout_prob;
  int64_t philox_seed;
  int64_t philox_offset;

  // BHM mode lengths, completely contiguous
  const void* logsumexp_ptr;
  void* dot_out_ptr;
};
