/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <mutex>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

#include "autogen/cutlassF.h"
#include "kernel_forward.h"
#include "pytorch_utils.h"

#define USE_MEM_EFF_ATTENTION

namespace {
using namespace at;
/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt>
efficient_attention_forward_cutlass(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const std::optional<at::Tensor>& bias, // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const std::optional<int64_t> max_seqlen_q_,
    const std::optional<int64_t> max_seqlen_k_,
    double dropout_p, // attention matrix dropout probability
    int64_t custom_mask_type,
    bool compute_logsumexp,
    std::optional<double> scale,
    const std::optional<at::Tensor>& seqlen_k,
    const std::optional<int64_t> window_size) {
#if defined(USE_MEM_EFF_ATTENTION)
  // TODO In theory it is possible to compile with _CUDA_ARCH < 5.0 and run on a
  // machine that is >= 5.0. In practice, this is not a problem but since
  // this would avoid runtime architecture checks, we should look into it

  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  int64_t max_seqlen_q = 0, max_seqlen_k = 0;
  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k =
        0; // TODO: is this actually being set inside the kernel anywhere?
           // see https://github.com/pytorch/pytorch/issues/115590s
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor res;
  at::Tensor logsumexp;
  at::Tensor seed_t, offset_t;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // Note [Seed and Offset Device]
  // If we are currently in graph capture mode, we need to create the seed and
  // offset tensors on the device. This is necessary for CUDA graph-safe random
  // number generation, which requires the seed and offset tensors to be single
  // element tensors on device. During graph capture, when the seed and offset
  // tensors are passed the pointers act as scratch space for storing the RNG
  // state for the backwards pass. When calling backwards, we either construct a
  // PhiloxState with the pointers or the actual values. For more information on
  // CUDA graph-safe RNG states, see Note [CUDA Graph-safe RNG states].

  at::PhiloxCudaState philox_state;
  const bool in_capture_stream =
      at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;
  auto device = in_capture_stream ? at::kCUDA : at::kCPU;
  if (use_dropout) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // if using dropout, we produce 1 random number for each element of the
    // attention tensor
    philox_state = gen->philox_cuda_state(B * num_heads * M * N);

    if (in_capture_stream) {
      // The seed and offset will be populated by the kernel
      seed_t = at::empty({}, at::dtype(at::kLong).device(device));
      offset_t = at::empty({}, at::dtype(at::kLong).device(device));
    } else {
      auto [seed, offset] = at::cuda::philox::unpack(philox_state);
      seed_t = at::scalar_tensor(
          at::Scalar(static_cast<int64_t>(seed)), at::dtype(at::kLong));
      offset_t = at::scalar_tensor(
          at::Scalar(static_cast<int64_t>(offset)), at::dtype(at::kLong));
    }
  } else {
    // Not using dropout
    seed_t = at::empty({}, at::dtype(at::kLong).device(device));
    offset_t = at::empty({}, at::dtype(at::kLong).device(device));
  }

  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = p->major * 10 + p->minor;

  bool kernel_launched = false;
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (!Kernel::kSupportsDropout && use_dropout) {
      return;
    }
    if (!Kernel::kSupportsBias && bias.has_value()) {
      return;
    }

    if (value.size(3) > Kernel::kMaxK || key.size(3) > Kernel::kMaxK) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kAlignmentQ) ||
        (key.stride(2) % Kernel::kAlignmentK) ||
        (value.stride(2) % Kernel::kAlignmentV)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }
    kernel_launched = true;

    res = at::empty(
        {B, M, num_heads, Kv},
        query.options().dtype(
            CutlassToAtenDtype<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B,
         num_heads,
         compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));
    typename Kernel::Params p;
    p.query_ptr = (const scalar_t*)query.const_data_ptr();
    p.key_ptr = (const scalar_t*)key.const_data_ptr();
    p.value_ptr = (const scalar_t*)value.const_data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, num_heads, Kv},
          query.options().dtype(
              CutlassToAtenDtype<
                  typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();

    if (seqstart_q.has_value()) {
      p.seqstart_q_ptr = (const int32_t*)seqstart_q->const_data_ptr();
      p.seqstart_k_ptr = (const int32_t*)seqstart_k->const_data_ptr();
    }

    p.num_heads = num_heads;
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B;
    p.custom_mask_type = custom_mask_type;

    p.seqlen_k_ptr = nullptr;
    if (seqlen_k.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(seqlen_k.value());
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      p.seqlen_k_ptr = (const int32_t*)seqlen_k->const_data_ptr();
    }

    if (window_size.has_value()) {
      p.window_size = *window_size;
    }

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(p.head_dim)));
    }

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.o_strideM, res.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");
      p.attn_bias_ptr = (const scalar_t*)bias->const_data_ptr();

      TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
      TORCH_CHECK(
          bias->size(0) == query.size(0),
          "attn_bias: wrong shape (batch dimension)");
      TORCH_CHECK(
          bias->size(1) == query.size(2),
          "attn_bias: wrong shape (head dimension)");
      TORCH_CHECK(
          bias->size(2) == query.size(1),
          "attn_bias: wrong shape (seqlenQ dimension)");
      TORCH_CHECK(
          bias->size(3) == key.size(1),
          "attn_bias: wrong shape (seqlenKV dimension)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
    }

    p.use_dropout = use_dropout;
    if (p.use_dropout) {
      p.rng_engine_inputs = philox_state;
      p.dropout_prob = dropout_p;
      p.seed = seed_t.data_ptr<int64_t>();
      p.extragraph_offset = offset_t.data_ptr<int64_t>();
    }

    if (smem_bytes > 0xc000) {
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      TORCH_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }
    auto blocks = p.getBlocksGrid();
    if (blocks.x * blocks.y * blocks.z == 0 || key.size(1) == 0) {
      res.zero_();
      return;
    }
    Kernel::check_supported(p);
    kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  // Dispatch to the right kernel
  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());

  return std::make_tuple(
      std::move(res),
      std::move(logsumexp),
      std::move(seed_t),
      std::move(offset_t),
      max_seqlen_q,
      // TODO: why isn't this being set in the kernel?
      max_seqlen_k_.has_value() ? max_seqlen_k_.value() : max_seqlen_k);
#endif
  TORCH_CHECK(false, "USE_MEM_EFF_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{}, 0, 0);
}

// For testing in xFormers
bool has_cutlassF_kernel_for(
    at::ScalarType dtype,
    int64_t cc,
    int64_t maxShmem,
    int64_t dim_k) {
  bool found = false;

  auto callback = [&](auto kernelCls, auto kernelFn) {
    using Kernel = decltype(kernelCls);

    if (found) {
      return;
    }
    if (dim_k > Kernel::kMaxK) {
      return;
    }
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }
    found = true;
  };
  if (dtype == at::ScalarType::Float) {
    dispatch_cutlassF<float>(callback, cc);
  } else if (dtype == at::ScalarType::Half) {
    dispatch_cutlassF<cutlass::half_t>(callback, cc);
  } else {
    TORCH_CHECK(dtype == at::ScalarType::BFloat16, "invalid data type");
    dispatch_cutlassF<cutlass::bfloat16_t>(callback, cc);
  }
  return found;
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_cutlass"),
      TORCH_FN(efficient_attention_forward_cutlass));
}

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_has_cutlassF_kernel_for(ScalarType dtype, int cc, int maxShmem, int maxK) -> bool"));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_has_cutlassF_kernel_for"),
      TORCH_FN(has_cutlassF_kernel_for));
}
