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

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;

  int custom_mask_type;
  int window_size; // local-attention

  void* out_ptr;
};

struct BatchedForwardParams : public BatchedInferParams {
  bool use_dropout;
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

  // TODO: need remove this after dev-op fix
  std::vector<void*> randvals_ptrs;
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

  bool use_fp32_qkv_grad;
  bool is_mqa_gqa;

  // BMHK mode strides, last-dim contiguous
  std::array<int, 4> q_strides;
  std::array<int, 4> k_strides;
  std::array<int, 4> v_strides;
  std::array<int, 4> attn_bias_strides; // 4d tensor_view [B, H, M, N]
  std::array<int, 4> out_strides;

  std::array<int, 4> tmp_grad_k_strides;
  std::array<int, 4> tmp_grad_v_strides;

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;
  const void* grad_out_ptr;
  const void* out_ptr;

  uint8_t custom_mask_type;

  void* grad_q_ptr;
  void* grad_k_ptr;
  void* grad_v_ptr;
  void* grad_bias_ptr;

  float dropout_prob;
  int64_t philox_seed;
  int64_t philox_offset;

  // BHM mode lengths, completely contiguous
  const void* logsumexp_ptr;
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

  std::vector<int> host_seqstart_q;
  std::vector<int> host_seqstart_k;
  std::vector<int> host_seqlen_k;

  float scale;
  bool has_attn_bias;
  bool bias_has_grad;

  bool use_fp32_qkv_grad;
  bool is_mqa_gqa;

  // MHK mode strides, last-dim contiguous
  std::array<int, 3> q_strides;
  std::array<int, 3> k_strides;
  std::array<int, 3> v_strides;
  std::array<int, 3> out_strides;
  // 4d tensor view [B, H, M, N]
  std::array<int, 4> attn_bias_strides;

  std::array<int, 3> tmp_grad_k_strides;
  std::array<int, 3> tmp_grad_v_strides;

  std::vector<const void*> q_ptrs;
  std::vector<const void*> k_ptrs;
  std::vector<const void*> v_ptrs;
  std::vector<const void*> attn_bias_ptrs;
  std::vector<const void*> grad_out_ptrs;
  std::vector<const void*> out_ptrs;

  // used by the light_v2 kernel
  // TODO use these as workspace
  std::vector<void*> ydotdy_ptrs;

  uint8_t custom_mask_type;

  std::vector<void*> grad_q_ptrs;
  std::vector<void*> grad_k_ptrs;
  std::vector<void*> grad_v_ptrs;
  std::vector<void*> grad_bias_ptrs;

  float dropout_prob;
  int64_t philox_seed;
  int64_t philox_offset;

  // BHM mode lengths, completely contiguous
  std::vector<const void*> logsumexp_ptrs;

  // TODO: need remove this after dev-op fix
  std::vector<void*> randvals_ptrs;
};
