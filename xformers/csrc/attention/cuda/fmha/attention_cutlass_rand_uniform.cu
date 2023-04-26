/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/core/TensorOptions.h>
#include <curand_kernel.h>
#include <torch/library.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

namespace {

/**
 * simple kernel that populates a tensor with rand uniform values.
 * currently only used for testing purposes, not much attention
 * is paid to performance.
 *
 * problem is partitioned as follows:
 * - (batch, head) is given by block coordinates
 * - each thread handles a row for a given (batch, head)
 */
template <typename mask_t>
__global__ void rand_uniform_kernel(
    int64_t n_heads,
    int64_t n_queries,
    int64_t n_keys,
    float dropout_prob,
    at::PhiloxCudaState rng_engine_inputs,
    mask_t* mask_out,
    int64_t mask_numel) {
  const int64_t batch_id = blockIdx.x;
  const int64_t head_id = blockIdx.y;
  const int64_t query_idx = threadIdx.x;

  const auto seeds = at::cuda::philox::unpack(rng_engine_inputs);

  const int dropout_seq_start = batch_id * (n_heads * n_queries * n_keys) +
      head_id * (n_queries * n_keys);

  curandStatePhilox4_32_10_t curand_state;
  curand_init(
      std::get<0>(seeds),
      0,
      std::get<1>(seeds) + dropout_seq_start + query_idx * n_keys,
      &curand_state);

  for (int key_start_idx = 0; key_start_idx < n_keys; key_start_idx += 4) {
    float4 rand_quad = curand_uniform4(&curand_state);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t linear_idx = batch_id * (n_heads * n_queries * n_keys) +
          head_id * (n_queries * n_keys) + query_idx * n_keys + key_start_idx +
          i;

      if (linear_idx < mask_numel) {
        mask_out[linear_idx] = (&rand_quad.x)[i];
      }
    }
  }
}

/**
 * fill tensor with random uniform values. only used for testing, not much
 * attention is paid to performance
 */
at::Tensor rand_uniform(double p, at::Tensor out) {
  const int64_t batch_sz = out.size(0);
  const int64_t n_heads = out.size(1);
  const int64_t n_queries = out.size(2);
  const int64_t n_keys = out.size(3);

  at::CUDAGeneratorImpl* gen =
      at::get_generator_or_default<at::CUDAGeneratorImpl>(
          c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
  at::PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs =
        gen->philox_cuda_state(batch_sz * n_heads * n_queries * n_keys);
  }

  rand_uniform_kernel<float><<<dim3(batch_sz, n_heads), n_queries>>>(
      n_heads,
      n_queries,
      n_keys,
      p,
      rng_engine_inputs,
      reinterpret_cast<float*>(out.data_ptr()),
      out.numel());

  return out;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_cutlass_rand_uniform"),
      TORCH_FN(rand_uniform));
}
