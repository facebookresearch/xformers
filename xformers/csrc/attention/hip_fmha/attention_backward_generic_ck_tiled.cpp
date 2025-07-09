/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <cstdlib>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>
#include <ATen/cuda/PhiloxUtils.cuh>

#include "ck_fmha_util.h"
#include "ck_tiled_fmha_params.h"

extern void batched_backward_fp16(
    BatchedBackwardParams& param,
    hipStream_t stream);
extern void batched_backward_bf16(
    BatchedBackwardParams& param,
    hipStream_t stream);
extern void grouped_backward_fp16(
    GroupedBackwardParams& param,
    hipStream_t stream);
extern void grouped_backward_bf16(
    GroupedBackwardParams& param,
    hipStream_t stream);

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
efficient_attention_backward_ck(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& bias, // additive attention bias
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_k_,
    const c10::optional<at::Tensor>& seqlen_k,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    double dropout_p, // dropout probability
    int64_t rng_seed, // seed using for generating random numbers for dropout
    int64_t rng_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const c10::optional<double> scale,
    const c10::optional<int64_t> window_size) {
  // ndim
  TORCH_CHECK(query.dim() == grad_out.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) % key.size(2) == 0);
  TORCH_CHECK(key.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out.size(3));

  TORCH_CHECK(out.sizes() == grad_out.sizes());

  // last dim is contiguous, device is CUDA
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(out);
  // CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(grad_out);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  // logsumexp should be completely contiguous
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(logsumexp);

  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  TORCH_CHECK(
      !(seqstart_q.has_value() && bias.has_value()),
      "seqstart_q + bias not supported");

  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "seqstart_q only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    TORCH_CHECK(max_seqlen_k_.has_value());
  }

  hipStream_t stream = c10::hip::getCurrentHIPStream().stream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t Hq = query.size(2);
  int64_t Hkv = key.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  auto opts = query.options();

  at::Tensor grad_q, grad_k, grad_v, grad_bias;

  if (query.size(1) == key.size(1) && query.size(3) == value.size(3) &&
      query.size(2) == key.size(2) &&
      query.storage().is_alias_of(key.storage()) &&
      query.storage().is_alias_of(value.storage())) {
    // Create one big contiguous chunk for grad_q, grad_k, grad_v
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    at::Tensor chunk = at::empty({B, M, 3, Hq, K}, opts);
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else if (
      key.size(3) == value.size(3) &&
      key.storage().is_alias_of(value.storage())) {
    // Create one big contiguous chunk for grad_k, grad_v
    // This is because k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    at::Tensor chunk = at::empty({B, N, 2, Hkv, Kv}, opts);
    grad_k = chunk.select(2, 0);
    grad_v = chunk.select(2, 1);

    grad_q = at::empty_strided(query.sizes(), query.strides(), query.options());
  } else {
    grad_q = at::empty_strided(query.sizes(), query.strides(), query.options());
    grad_k = at::empty(key.sizes(), key.options());
    grad_v = at::empty(value.sizes(), value.options());
  }

  at::Tensor grad_q_f32;
  const bool use_grad_q_f32 =
      (query.scalar_type() == at::ScalarType::BFloat16 ||
       query.scalar_type() == at::ScalarType::Half);

  if (use_grad_q_f32) {
    grad_q_f32 = at::empty(grad_q.sizes(), opts.dtype(at::kFloat));
    grad_q_f32.fill_(0);
  } else {
    grad_q.fill_(0);
  };

  // CK-FlashAttn requires q/k/v to have same shapes with dQ/dK/dV respectively
  TORCH_CHECK(query.sizes() == grad_q.sizes());
  TORCH_CHECK(query.strides() == grad_q.strides());
  TORCH_CHECK(key.sizes() == grad_k.sizes());
  TORCH_CHECK(value.sizes() == grad_v.sizes());

  const bool bias_requires_grad = bias.has_value() && bias->requires_grad();

  // even it is an output, the grad_bias is required to use the same data-type
  // as bias in CK-FlashAttn
  if (bias_requires_grad) {
    grad_bias =
        at::empty_strided(bias->sizes(), bias->strides(), bias->options());
    // cleaning is needed since masked tile does no outputting in our
    // implementation
    grad_bias.fill_(0);
  }

  bool is_mqa_gqa = (Hq > Hkv);

  at::Tensor tmp_grad_k, tmp_grad_v;

  if (is_mqa_gqa) {
    // allocate tmp_grad_k/tmp_grad_v which will be reduce to
    // grad_k/grad_v for returning
    tmp_grad_k = at::empty({B, N, Hq, K}, opts);
    tmp_grad_v = at::empty({B, N, Hq, Kv}, opts);
  }

  auto dot_out = at::empty_like(logsumexp);

  auto set_batched_backward_params = [&](BatchedBackwardParams& p) {
    p.B = B;
    p.M = M;
    p.N = N;
    p.Hq = Hq;
    p.Hkv = Hkv;
    p.K = K;
    p.Kv = Kv;

    p.is_mqa_gqa = is_mqa_gqa;

    TORCH_CHECK(p.B == logsumexp.size(0));
    TORCH_CHECK(p.Hq == logsumexp.size(1));
    TORCH_CHECK(p.M == logsumexp.size(2));

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(K)));
    }

    p.q_ptr = query.data_ptr();
    p.k_ptr = key.data_ptr();
    p.v_ptr = value.data_ptr();
    p.grad_out_ptr = grad_out.data_ptr();
    p.out_ptr = out.data_ptr();

    p.grad_q_ptr = grad_q.data_ptr();
    p.grad_k_ptr = is_mqa_gqa ? tmp_grad_k.data_ptr() : grad_k.data_ptr();
    p.grad_v_ptr = is_mqa_gqa ? tmp_grad_v.data_ptr() : grad_v.data_ptr();

    if (use_grad_q_f32)
      p.grad_q_f32_ptr = grad_q_f32.data_ptr();
    else
      p.grad_q_f32_ptr = nullptr;

    p.q_strides = {
        static_cast<int>(query.stride(0)),
        static_cast<int>(query.stride(1)),
        static_cast<int>(query.stride(2)),
        static_cast<int>(query.stride(3))};
    p.k_strides = {
        static_cast<int>(key.stride(0)),
        static_cast<int>(key.stride(1)),
        static_cast<int>(key.stride(2)),
        static_cast<int>(key.stride(3))};
    p.v_strides = {
        static_cast<int>(value.stride(0)),
        static_cast<int>(value.stride(1)),
        static_cast<int>(value.stride(2)),
        static_cast<int>(value.stride(3))};
    p.out_strides = {
        static_cast<int>(out.stride(0)),
        static_cast<int>(out.stride(1)),
        static_cast<int>(out.stride(2)),
        static_cast<int>(out.stride(3))};
    p.grad_out_strides = {
        static_cast<int>(grad_out.stride(0)),
        static_cast<int>(grad_out.stride(1)),
        static_cast<int>(grad_out.stride(2)),
        static_cast<int>(grad_out.stride(3))};

    p.lsed_strides = {
        static_cast<int>(logsumexp.stride(0)),
        static_cast<int>(logsumexp.stride(1)),
        static_cast<int>(logsumexp.stride(2))};

    if (use_grad_q_f32) {
      p.grad_q_f32_strides = {
          static_cast<int>(grad_q_f32.stride(0)),
          static_cast<int>(grad_q_f32.stride(1)),
          static_cast<int>(grad_q_f32.stride(2)),
          static_cast<int>(grad_q_f32.stride(3))};
    }

    if (is_mqa_gqa) {
      p.grad_k_strides = {
          static_cast<int>(tmp_grad_k.stride(0)),
          static_cast<int>(tmp_grad_k.stride(1)),
          static_cast<int>(tmp_grad_k.stride(2)),
          static_cast<int>(tmp_grad_k.stride(3))};
      p.grad_v_strides = {
          static_cast<int>(tmp_grad_v.stride(0)),
          static_cast<int>(tmp_grad_v.stride(1)),
          static_cast<int>(tmp_grad_v.stride(2)),
          static_cast<int>(tmp_grad_v.stride(3))};
    } else {
      p.grad_k_strides = {
          static_cast<int>(grad_k.stride(0)),
          static_cast<int>(grad_k.stride(1)),
          static_cast<int>(grad_k.stride(2)),
          static_cast<int>(grad_k.stride(3))};
      p.grad_v_strides = {
          static_cast<int>(grad_v.stride(0)),
          static_cast<int>(grad_v.stride(1)),
          static_cast<int>(grad_v.stride(2)),
          static_cast<int>(grad_v.stride(3))};
    };

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(bias->scalar_type() == query.scalar_type());

      p.has_attn_bias = true;
      p.attn_bias_ptr = bias->data_ptr();

      const at::Tensor bias_4d_view = get_bias_4d_view(*bias, B, Hq, M, N);

      p.attn_bias_strides = {
          static_cast<int>(bias_4d_view.stride(0)),
          static_cast<int>(bias_4d_view.stride(1)),
          static_cast<int>(bias_4d_view.stride(2)),
          static_cast<int>(bias_4d_view.stride(3))};

      if (bias_requires_grad)
        p.grad_bias_ptr = grad_bias.data_ptr();
    } else {
      p.has_attn_bias = false;
      p.attn_bias_ptr = nullptr;
      p.grad_bias_ptr = nullptr;
    }

    p.bias_has_grad = bias_requires_grad;

    p.custom_mask_type = custom_mask_type;
    p.window_size =
        window_size.has_value() ? (*window_size > 0 ? *window_size : 0) : 0;

    p.dropout_prob = static_cast<float>(dropout_p);
    p.philox_seed = rng_seed;
    p.philox_offset = rng_offset;

    p.logsumexp_ptr = logsumexp.data_ptr();
    p.dot_out_ptr = dot_out.data_ptr();
  };

  auto set_grouped_backward_params = [&](GroupedBackwardParams& p) {
    p.num_batches = seqstart_q->size(0) - 1;
    p.M = M;
    p.N = N;
    p.Hq = Hq;
    p.Hkv = Hkv;
    p.K = K;
    p.Kv = Kv;

    p.is_mqa_gqa = is_mqa_gqa;

    p.max_seqlen_q = *max_seqlen_q_;
    p.max_seqlen_k = *max_seqlen_k_;

    // unpadded lse layout required
    TORCH_CHECK(p.Hq == logsumexp.size(1));
    TORCH_CHECK(p.M == logsumexp.size(2));

    if (scale.has_value())
      p.scale = float(*scale);
    else
      p.scale = float(1.0 / std::sqrt(float(K)));

    p.q_strides = {
        static_cast<int>(query.stride(1)),
        static_cast<int>(query.stride(2)),
        static_cast<int>(query.stride(3))};
    p.k_strides = {
        static_cast<int>(key.stride(1)),
        static_cast<int>(key.stride(2)),
        static_cast<int>(key.stride(3))};
    p.v_strides = {
        static_cast<int>(value.stride(1)),
        static_cast<int>(value.stride(2)),
        static_cast<int>(value.stride(3))};
    p.out_strides = {
        static_cast<int>(out.stride(1)),
        static_cast<int>(out.stride(2)),
        static_cast<int>(out.stride(3))};
    p.grad_out_strides = {
        static_cast<int>(grad_out.stride(1)),
        static_cast<int>(grad_out.stride(2)),
        static_cast<int>(grad_out.stride(3))};

    p.lsed_strides = {
        static_cast<int>(logsumexp.stride(1)),
        static_cast<int>(logsumexp.stride(2))};

    if (use_grad_q_f32) {
      p.grad_q_f32_strides = {
          static_cast<int>(grad_q_f32.stride(1)),
          static_cast<int>(grad_q_f32.stride(2)),
          static_cast<int>(grad_q_f32.stride(3))};
    }

    if (is_mqa_gqa) {
      p.grad_k_strides = {
          static_cast<int>(tmp_grad_k.stride(1)),
          static_cast<int>(tmp_grad_k.stride(2)),
          static_cast<int>(tmp_grad_k.stride(3))};
      p.grad_v_strides = {
          static_cast<int>(tmp_grad_v.stride(1)),
          static_cast<int>(tmp_grad_v.stride(2)),
          static_cast<int>(tmp_grad_v.stride(3))};
    } else {
      p.grad_k_strides = {
          static_cast<int>(grad_k.stride(1)),
          static_cast<int>(grad_k.stride(2)),
          static_cast<int>(grad_k.stride(3))};
      p.grad_v_strides = {
          static_cast<int>(grad_v.stride(1)),
          static_cast<int>(grad_v.stride(2)),
          static_cast<int>(grad_v.stride(3))};
    };

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(bias->scalar_type() == query.scalar_type());

      p.has_attn_bias = true;
      const at::Tensor bias_4d_view = get_bias_4d_view(*bias, B, Hq, M, N);
      p.attn_bias_strides = {
          static_cast<int>(bias_4d_view.stride(0)),
          static_cast<int>(bias_4d_view.stride(1)),
          static_cast<int>(bias_4d_view.stride(2)),
          static_cast<int>(bias_4d_view.stride(3))};
    } else
      p.has_attn_bias = false;

    p.bias_has_grad = bias_requires_grad;

    p.custom_mask_type = custom_mask_type;
    p.window_size =
        window_size.has_value() ? (*window_size > 0 ? *window_size : 0) : 0;

    // interesting: the tensors have to be defined here, moving to more local
    // scope will cause issue
    at::Tensor dev_seqstart_q;
    at::Tensor dev_seqstart_k;
    at::Tensor dev_seqlen_k;

    if (seqstart_q->is_cpu()) {
      dev_seqstart_q = at::empty({p.num_batches + 1}, opts.dtype(at::kInt));
      p.seqstart_q_dev_ptr = dev_seqstart_q.data_ptr();
      HIP_CALL_CHECK(hipMemcpyAsync(
          p.seqstart_q_dev_ptr,
          seqstart_q->data_ptr(),
          (p.num_batches + 1) * sizeof(int),
          hipMemcpyHostToDevice,
          stream));
    } else
      p.seqstart_q_dev_ptr = seqstart_q->data_ptr();

    if (seqstart_k->is_cpu()) {
      dev_seqstart_k = at::empty({p.num_batches + 1}, opts.dtype(at::kInt));

      p.seqstart_k_dev_ptr = dev_seqstart_k.data_ptr();
      HIP_CALL_CHECK(hipMemcpyAsync(
          p.seqstart_k_dev_ptr,
          seqstart_k->data_ptr(),
          (p.num_batches + 1) * sizeof(int),
          hipMemcpyHostToDevice,
          stream));
    } else
      p.seqstart_k_dev_ptr = seqstart_k->data_ptr();

    if (seqlen_k.has_value()) {
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      TORCH_CHECK(seqlen_k->dim() == 1);
      TORCH_CHECK(seqlen_k->size(0) == p.num_batches)

      if (seqlen_k->is_cpu()) {
        dev_seqlen_k = at::empty({p.num_batches}, opts.dtype(at::kInt));

        p.seqlen_k_dev_ptr = dev_seqlen_k.data_ptr();
        HIP_CALL_CHECK(hipMemcpyAsync(
            p.seqlen_k_dev_ptr,
            seqlen_k->data_ptr(),
            p.num_batches * sizeof(int),
            hipMemcpyHostToDevice,
            stream));
      } else
        p.seqlen_k_dev_ptr = seqlen_k->data_ptr();
    } else
      p.seqlen_k_dev_ptr = nullptr;

    p.dropout_prob = static_cast<float>(dropout_p);
    p.philox_seed = rng_seed;
    p.philox_offset = rng_offset;

    p.q_ptr = query.data_ptr();
    p.k_ptr = key.data_ptr();
    p.v_ptr = value.data_ptr();

    p.out_ptr = out.data_ptr();
    p.grad_out_ptr = grad_out.data_ptr();
    p.attn_bias_ptr = bias.has_value() ? bias->data_ptr() : nullptr;

    p.logsumexp_ptr = logsumexp.data_ptr();
    p.dot_out_ptr = dot_out.data_ptr();

    p.grad_q_ptr = grad_q.data_ptr();
    p.grad_k_ptr = is_mqa_gqa ? tmp_grad_k.data_ptr() : grad_k.data_ptr();
    p.grad_v_ptr = is_mqa_gqa ? tmp_grad_v.data_ptr() : grad_v.data_ptr();
    p.grad_bias_ptr = bias_requires_grad ? grad_bias.data_ptr() : nullptr;

    if (use_grad_q_f32)
      p.grad_q_f32_ptr = grad_q_f32.data_ptr();
    else
      p.grad_q_f32_ptr = nullptr;
  };

  auto inDataType = query.scalar_type();

  if (!seqstart_q.has_value()) { // input is batched
    BatchedBackwardParams batched_backward_params;

    set_batched_backward_params(batched_backward_params);

    if (inDataType == at::ScalarType::Half) {
      batched_backward_fp16(batched_backward_params, stream);
    } else if (inDataType == at::ScalarType::BFloat16) {
      batched_backward_bf16(batched_backward_params, stream);
    } else
      throw std::runtime_error("input data-type is not supported");
  } else { // input is grouped
    GroupedBackwardParams grouped_backward_params;

    set_grouped_backward_params(grouped_backward_params);

    if (inDataType == at::ScalarType::Half) {
      grouped_backward_fp16(grouped_backward_params, stream);
    } else if (inDataType == at::ScalarType::BFloat16) {
      grouped_backward_bf16(grouped_backward_params, stream);
    } else
      throw std::runtime_error("input data-type is not supported");
  }

  if (is_mqa_gqa) {
    auto tmp_grad_k_view = tmp_grad_k.unflatten(2, {Hkv, Hq / Hkv});
    auto tmp_grad_v_view = tmp_grad_v.unflatten(2, {Hkv, Hq / Hkv});
    grad_k = tmp_grad_k_view.sum(3);
    grad_v = tmp_grad_v_view.sum(3);
  }

  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
efficient_attention_backward_ck_meta(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& bias, // additive attention bias
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_k_,
    const c10::optional<at::Tensor>& seqlen_k,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    double dropout_p, // dropout probability
    int64_t rng_seed, // seed using for generating random numbers for dropout
    int64_t rng_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const c10::optional<double> scale,
    const c10::optional<int64_t> window_size) {
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t Hq = query.size(2);
  int64_t Hkv = key.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  auto opts = query.options();

  at::Tensor grad_q, grad_k, grad_v, grad_bias;

  if (query.size(1) == key.size(1) && query.size(3) == value.size(3) &&
      query.size(2) == key.size(2) &&
      query.storage().is_alias_of(key.storage()) &&
      query.storage().is_alias_of(value.storage())) {
    // Create one big contiguous chunk for grad_q, grad_k, grad_v
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    at::Tensor chunk = at::empty({B, M, 3, Hq, K}, opts);
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else if (
      key.size(3) == value.size(3) &&
      key.storage().is_alias_of(value.storage())) {
    // Create one big contiguous chunk for grad_k, grad_v
    // This is because k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    at::Tensor chunk = at::empty({B, N, 2, Hkv, Kv}, opts);
    grad_k = chunk.select(2, 0);
    grad_v = chunk.select(2, 1);

    grad_q = at::empty_strided(query.sizes(), query.strides(), query.options());
  } else {
    grad_q = at::empty_strided(query.sizes(), query.strides(), query.options());
    grad_k = at::empty_strided(key.sizes(), key.strides(), key.options());
    grad_v = at::empty_strided(value.sizes(), value.strides(), value.options());
  }

  const bool bias_requires_grad = bias.has_value() && bias->requires_grad();
  // even it is an output, the grad_bias is required to use the same data-type
  // as bias in CK-FlashAttn
  if (bias_requires_grad) {
    grad_bias =
        at::empty_strided(bias->sizes(), bias->strides(), bias->options());
  }
  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_ck"),
      TORCH_FN(efficient_attention_backward_ck));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_ck"),
      TORCH_FN(efficient_attention_backward_ck_meta));
}
