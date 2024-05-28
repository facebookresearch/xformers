/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "autogen/cutlassB.h"
#include "gemm_kernel_utils.h"
#include "kernel_backward.h"
#include "pytorch_utils.h"

#define USE_MEM_EFF_ATTENTION

namespace {
using namespace at;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_cutlass(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const std::optional<at::Tensor>& kernel_bias, // additive attention bias
    const at::Tensor& out,
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_q_dummy,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& cu_seqlens_k_dummy,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_q,
    // (Mode 1MHK only) Maximum sequence length across batches
    int64_t max_seqlen_k,
    const at::Tensor& logsumexp,
    double dropout_p, // dropout probability
    const at::Tensor&
        philox_seed, // seed using for generating random numbers for dropout
    const at::Tensor& philox_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const bool bias_requires_grad,
    const std::optional<double> scale,
    std::optional<int64_t> num_splits_key,
    const std::optional<int64_t> window_size) {
#if defined(USE_MEM_EFF_ATTENTION)
  if (!grad_out_.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
  }
  // This path is used when we directly call _efficient_attention_forward
  // from python.
  // This is needed because SaveVariable automatically converts
  // std::optional to undefined tensor
  std::optional<Tensor> bias, cu_seqlens_q, cu_seqlens_k;
  bias = kernel_bias.has_value() && !kernel_bias->defined() ? c10::nullopt
                                                            : kernel_bias;
  cu_seqlens_q =
      cu_seqlens_q_dummy.has_value() && !cu_seqlens_q_dummy->defined()
      ? c10::nullopt
      : cu_seqlens_q_dummy;
  cu_seqlens_k =
      cu_seqlens_k_dummy.has_value() && !cu_seqlens_k_dummy->defined()
      ? c10::nullopt
      : cu_seqlens_k_dummy;

  // ndim
  TORCH_CHECK(query.dim() == grad_out_.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 4);

  // batch size
  TORCH_CHECK(query.size(0) == grad_out_.size(0));
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // seqlen
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(query.size(1) == grad_out_.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out_.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out_.size(3));

  // handle potentially non-contiguous grad_out through a copy
  auto grad_out = grad_out_.contiguous();
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(grad_out);

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  TORCH_CHECK(
      !(cu_seqlens_q.has_value() && bias.has_value()),
      "cu seqlen + bias not supported");
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(cu_seqlens_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_k));
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q > 0, "max_seqlen_q required with `cu_seqlens_q`");
    TORCH_CHECK(max_seqlen_k > 0, "max_seqlen_k required with `cu_seqlens_k`");
    TORCH_CHECK(
        max_seqlen_k <= key.size(1), "Invalid max_seqlen_k:", max_seqlen_k);
    TORCH_CHECK(
        max_seqlen_q <= query.size(1), "Invalid max_seqlen_q:", max_seqlen_q);
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t nH = query.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  at::Tensor grad_q, grad_k, grad_v, grad_bias;
  if (query.size(1) == key.size(1) && query.size(3) == value.size(3) &&
      query.storage().is_alias_of(key.storage()) &&
      query.storage().is_alias_of(value.storage())) {
    // Create one big contiguous chunk
    // This is because q, k and v usually come from a single
    // output of a linear layer that is chunked.
    // Creating the gradients with the right layout saves us
    // a `torch.cat` call in the backward pass
    at::Tensor chunk = at::empty({B, M, 3, nH, K}, query.options());
    grad_q = chunk.select(2, 0);
    grad_k = chunk.select(2, 1);
    grad_v = chunk.select(2, 2);
  } else {
    grad_q = at::empty(query.sizes(), query.options());
    grad_k = at::empty(key.sizes(), key.options());
    grad_v = at::empty(value.sizes(), value.options());
  }

  if (bias_requires_grad) {
    // force alignment for the last dim
    std::vector<int64_t> sz = bias->sizes().vec();
    int64_t lastDim = sz[sz.size() - 1];
    int64_t alignTo = 16;
    sz[sz.size() - 1] = alignTo * ((lastDim + alignTo - 1) / alignTo);
    grad_bias = at::empty(sz, bias->options())
                    .slice(/*dim=*/-1, /*start=*/0, /*end=*/lastDim);
  }
  at::Tensor workspace;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // See Note [Seed and Offset Device]
  at::PhiloxCudaState rng_engine_inputs;
  if (use_dropout) {
    if (at::cuda::currentStreamCaptureStatus() ==
        at::cuda::CaptureStatus::None) {
      rng_engine_inputs = at::PhiloxCudaState(
          *philox_seed.data_ptr<int64_t>(), *philox_offset.data_ptr<int64_t>());
    } else { // dropout + capture
      rng_engine_inputs = at::PhiloxCudaState(
          philox_seed.data_ptr<int64_t>(),
          philox_offset.data_ptr<int64_t>(),
          0);
    }
  }

  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = p->major * 10 + p->minor;

  bool kernel_launched = false;
  const auto maxK = std::max(query.size(3), value.size(3));
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (Kernel::kMaxK < maxK) {
      return;
    }
    // Dropout must be supported if we need it
    if (use_dropout && !Kernel::kApplyDropout) {
      return;
    }
    if (Kernel::kKeysQueriesAlignedToBlockSize &&
        (cu_seqlens_q.has_value() || M % Kernel::kBlockSizeI ||
         N % Kernel::kBlockSizeJ)) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kMinimumAlignment) ||
        (key.stride(2) % Kernel::kMinimumAlignment) ||
        (value.stride(2) % Kernel::kMinimumAlignment)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }

    kernel_launched = true;

    // TODO: Fuse this into a kernel?
    // This is a bottleneck for smaller sequences (M <= 128)
    auto delta = Kernel::kKernelComputesDelta
        ? at::empty({B, nH, M}, query.options().dtype(at::ScalarType::Float))
        : (grad_out.to(at::kFloat) * out.to(at::kFloat))
              .sum(-1)
              .transpose(-2, -1)
              .contiguous();
    TORCH_INTERNAL_ASSERT(delta.size(0) == B);
    TORCH_INTERNAL_ASSERT(delta.size(1) == nH);
    TORCH_INTERNAL_ASSERT(delta.size(2) == M);

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query.data_ptr();
    p.key_ptr = (scalar_t*)key.data_ptr();
    p.value_ptr = (scalar_t*)value.data_ptr();
    p.logsumexp_ptr = (typename Kernel::lse_scalar_t*)logsumexp.data_ptr();
    p.output_ptr = (scalar_t*)out.data_ptr();
    p.grad_output_ptr = (scalar_t*)grad_out.data_ptr();
    p.grad_query_ptr = (scalar_t*)grad_q.data_ptr();
    p.grad_key_ptr = (scalar_t*)grad_k.data_ptr();
    p.grad_value_ptr = (scalar_t*)grad_v.data_ptr();
    p.delta_ptr = (float*)delta.data_ptr();
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = cu_seqlens_q.has_value() ? cu_seqlens_q->size(0) - 1 : B;
    p.num_heads = nH;
    p.custom_mask_type = custom_mask_type;
    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(p.head_dim)));
    }
    if (cu_seqlens_q.has_value()) {
      p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
    }

    if (window_size.has_value()) {
      p.window_size = *window_size;
    }

    ASSIGN_CHECK_OVERFLOW(p.lse_strideB, logsumexp.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.lse_strideH, logsumexp.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideB, grad_out.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideM, grad_out.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.gO_strideH, grad_out.stride(2));

    ASSIGN_CHECK_OVERFLOW(p.o_strideB, out.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.o_strideH, out.stride(2));

    ASSIGN_CHECK_OVERFLOW(p.gQ_strideB, grad_q.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gK_strideB, grad_k.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gV_strideB, grad_v.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.gQ_strideH, grad_q.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.gK_strideH, grad_k.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.gV_strideH, grad_v.stride(2));
    p.gQKV_strideM_multiplier = grad_q.is_contiguous() ? 1 : 3;
    TORCH_INTERNAL_ASSERT(p.gQ_strideM() == grad_q.stride(1));
    TORCH_INTERNAL_ASSERT(p.gK_strideM() == grad_k.stride(1));
    TORCH_INTERNAL_ASSERT(p.gV_strideM() == grad_v.stride(1));

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.delta_strideB, delta.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.delta_strideH, delta.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");

      p.bias_ptr = (scalar_t*)bias->data_ptr();

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
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));

      if (bias_requires_grad) {
        p.grad_bias_ptr = (scalar_t*)grad_bias.data_ptr();

        ASSIGN_CHECK_OVERFLOW(p.gB_strideB, grad_bias.stride(0));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideH, grad_bias.stride(1));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideM, grad_bias.stride(2));
      }
    }

    if (use_dropout) {
      p.rng_engine_inputs = rng_engine_inputs;
      p.dropout_prob = dropout_p;
    }

    // Heuristic for finding optimal number of splits
    auto parallelism_without_split_key =
        p.getBlocksGrid().x * p.getBlocksGrid().y * p.getBlocksGrid().z;
    p.num_splits_key = cutlass::ceil_div(p.num_keys, Kernel::kBlockSizeJ);
    if (num_splits_key.has_value()) {
      p.num_splits_key =
          std::min<int64_t>(p.num_splits_key, num_splits_key.value());
    } else {
      // Keys splitting heuristic

      // If we already have enough parallelism, split-keys can help
      // better use L2 cache.
      // This is negligible when the seqlen is too small tho
      if (parallelism_without_split_key >= 256 &&
          p.num_keys <= 2 * Kernel::kBlockSizeJ) {
        p.num_splits_key = 1;
      }
      // Increasing `split_keys` leads to using more gmem for temporary storage
      // when we need a staging area for gK/gV. let's avoid that
      if (Kernel::kNeedsAccumGradK || Kernel::kNeedsAccumGradV) {
        p.num_splits_key = std::min(
            int(p.num_splits_key), 200 / (p.num_batches * p.num_heads));
      }
    }
    if (!Kernel::kEnableSplitKeys || p.num_splits_key < 1) {
      p.num_splits_key = 1;
    }

    auto& ctx = at::globalContext();
    if (ctx.deterministicAlgorithms()) {
      if (ctx.deterministicAlgorithmsWarnOnly()) {
        TORCH_WARN_ONCE(
            "Memory Efficient attention defaults to a non-deterministic algorithm. ",
            "To explicitly enable determinism call torch.use_deterministic_algorithms(True, warn_only=False).");
      } else {
        TORCH_CHECK(
            num_splits_key.value_or(1) <= 1,
            "Using `num_splits_key > 1` makes the algorithm non-deterministic, and pytorch's deterministic mode is enabled");
        p.num_splits_key = 1;
      }
    }
    int64_t size_bytes = p.workspace_size();
    if (size_bytes) {
      workspace =
          at::empty({size_bytes}, query.options().dtype(at::ScalarType::Byte));
      p.workspace = (float*)workspace.data_ptr();
      if (p.should_zero_workspace()) {
        workspace.zero_();
      }
    }

    // Handle the edge-cases where some tensors are empty
    if (p.num_queries == 0 || p.num_keys == 0 || p.num_batches == 0 ||
        p.num_heads == 0) {
      grad_k.zero_();
      grad_v.zero_();
      grad_q.zero_();
      return;
    }
    Kernel::check_supported(p);

    if (smem_bytes > 0xc000) {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      TORCH_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }

    // second syntax resulted in the error below on windows
    // error C3495: 'kernel_fn': a simple capture must be a variable
    // with automatic storage duration declared
    // in the reaching scope of the lambda
#ifdef _WIN32
    cudaFuncAttributes attr;
    AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
    TORCH_INTERNAL_ASSERT(
        attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability,
        "Something went wrong in the build process");
#else
    auto checkBinaryArchMatches = [&]() {
      cudaFuncAttributes attr;
      AT_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
      return attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability;
    };
    TORCH_INTERNAL_ASSERT(
        checkBinaryArchMatches(), "Something went wrong in the build process");
#endif

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassB<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassB: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(
      std::move(grad_q),
      std::move(grad_k),
      std::move(grad_v),
      std::move(grad_bias));
#endif
  TORCH_CHECK(false, "USE_MEM_EFF_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{});
}

bool has_cutlassB_kernel_for(
    at::ScalarType dtype,
    int64_t cc,
    int64_t maxShmem,
    int64_t maxK) {
  bool found = false;

  auto callback = [&](auto kernelCls, auto kernelFn) {
    using Kernel = decltype(kernelCls);

    if (found) {
      return;
    }
    if (Kernel::kMaxK < maxK) {
      return;
    }
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }
    found = true;
  };
  if (dtype == at::ScalarType::Float) {
    dispatch_cutlassB<float>(callback, cc);
  } else if (dtype == at::ScalarType::Half) {
    dispatch_cutlassB<cutlass::half_t>(callback, cc);
  } else {
    TORCH_CHECK(dtype == at::ScalarType::BFloat16, "Valid data type");
    dispatch_cutlassB<cutlass::bfloat16_t>(callback, cc);
  }
  return found;
}

using IterationDataOutput =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

IterationDataOutput _cutlassB_iteration_data(
    at::ScalarType dtype,
    int64_t cc,
    int64_t maxK,
    int64_t num_queries,
    int64_t num_keys,
    int64_t num_splits_key,
    int64_t window_size,
    int64_t custom_mask_type,
    int64_t query_start,
    int64_t key_start) {
  bool found = false;

  IterationDataOutput output;
  auto callback = [&](auto kernelCls, auto kernelFn) {
    using Kernel = decltype(kernelCls);

    if (found) {
      return;
    }
    if (Kernel::kMaxK < maxK) {
      return;
    }
    found = true;

    typename Kernel::Params p;
    p.num_queries = num_queries;
    p.num_keys = num_keys;
    p.num_splits_key = num_splits_key;
    p.window_size = window_size;
    p.custom_mask_type = custom_mask_type;

    int32_t new_query_start, new_key_start, num_parallel_blocks,
        smallest_query_for_key;
    num_parallel_blocks = Kernel::getNumParallelBlocksForQuery(p, query_start);
    smallest_query_for_key = Kernel::getSmallestQueryForKey(p, key_start);
    Kernel::incrIteration(
        p, query_start, key_start, new_query_start, new_key_start);
    output = std::make_tuple(
        Kernel::kBlockSizeI,
        Kernel::kBlockSizeJ,
        smallest_query_for_key,
        new_query_start,
        new_key_start,
        int64_t(num_parallel_blocks));
  };
  if (dtype == at::ScalarType::Float) {
    dispatch_cutlassB<float>(callback, cc);
  } else if (dtype == at::ScalarType::Half) {
    dispatch_cutlassB<cutlass::half_t>(callback, cc);
  } else {
    TORCH_CHECK(dtype == at::ScalarType::BFloat16, "Valid data type");
    dispatch_cutlassB<cutlass::bfloat16_t>(callback, cc);
  }
  TORCH_CHECK(found, "No kernel found");
  return output;
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_cutlass"),
      TORCH_FN(mem_efficient_attention_backward_cutlass));
}

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_has_cutlassB_kernel_for(ScalarType dtype, int cc, int maxShmem, int maxK) -> bool"));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_has_cutlassB_kernel_for"),
      TORCH_FN(has_cutlassB_kernel_for));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_cutlassB_iteration_data("
      "ScalarType dtype, int cc, int maxK, "
      "int num_queries=0, int num_keys=0, int num_splits_key=1, int window_size=0, int custom_mask_type=0, "
      "int query_start=0, int key_start=0) -> (int, int, int, int, int, int)"));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_cutlassB_iteration_data"),
      TORCH_FN(_cutlassB_iteration_data));
}
