/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <torch/all.h>

#define XFORMERS_CHECK(COND, ERR)          \
  if (!(COND)) {                           \
    std::ostringstream ostr;               \
    ostr << "'" #COND "' failed: " << ERR; \
    throw std::runtime_error(ostr.str());  \
  }

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(TENSOR.is_contiguous(), #TENSOR " must be contiguous");

#define CHECK_NOSPARSE_CONTIGUOUS_CPU(TENSOR)                             \
  XFORMERS_CHECK(TENSOR.is_cpu(), #TENSOR " must be a CPU tensor");       \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(TENSOR.is_contiguous(), #TENSOR " must be contiguous");

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

#define HIP_CALL_CHECK(flag)                                                 \
  do {                                                                       \
    hipError_t _tmpVal;                                                      \
    if ((_tmpVal = flag) != hipSuccess) {                                    \
      std::ostringstream ostr;                                               \
      ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
           << hipGetErrorString(_tmpVal);                                    \
      throw std::runtime_error(ostr.str());                                  \
    }                                                                        \
  } while (0)

/**
 * kernels expect 4D bias/bias.grad with shape
 * (batch_sz, n_heads, n_queries, n_keys). common bias shapes users may pass
 * are:
 * - (n_queries, n_keys)
 * - (batch_sz * n_heads, n_queries, n_keys)
 * - (batch_sz, n_heads, n_queries, n_keys)
 *
 * expand the bias as needed - be careful to only create a view with different
 * shape/strides, no copies allowed.
 */
static inline at::Tensor get_bias_4d_view(
    const at::Tensor& bias,
    int batch_sz,
    int n_heads,
    int n_queries,
    int n_keys) {
  TORCH_CHECK(
      bias.size(-2) == n_queries,
      "bias.size(-2) != n_queries: ",
      bias.size(-2),
      " != ",
      n_queries);
  TORCH_CHECK(
      bias.size(-1) == n_keys,
      "bias.size(-1) != n_keys: ",
      bias.size(-1),
      " != ",
      n_keys);
  switch (bias.dim()) {
    case 2: // (n_queries, n_keys) - broadcast across all batches and heads
      return bias.unsqueeze(0).unsqueeze(0).expand(
          {batch_sz, n_heads, n_queries, n_keys});
    case 3: // (batch_sz * n_heads, n_queries, n_keys) - just reshape
      TORCH_CHECK(bias.size(0) == batch_sz * n_heads);
      return bias.view({batch_sz, n_heads, n_queries, n_keys});
    case 4: // (batch_sz, n_heads, n_queries, n_keys) - do nothing
      TORCH_CHECK(bias.size(0) == batch_sz);
      TORCH_CHECK(bias.size(1) == n_heads)
      return bias;
    default:
      TORCH_CHECK(false, "bias can only have ndims in {2, 3, 4}");
  }
}

static inline int get_number_of_cu() {
  int device;

  HIP_CALL_CHECK(hipGetDevice(&device));

  hipDeviceProp_t props;

  HIP_CALL_CHECK(hipGetDeviceProperties(&props, device));

  return props.multiProcessorCount;
}
