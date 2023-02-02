#include <cmath>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "ATen/ops/empty_like.h"

#include "autogen/cutlassB.h"
#include "gemm_kernel_utils.h"
#include "kernel_backward.h"
#include "pytorch_utils.h"

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mem_efficient_attention_backward_cutlass(
    const at::Tensor& grad_out_,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const c10::optional<at::Tensor>& bias, // additive attention bias
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    double dropout_p, // dropout probability
    int64_t rng_seed, // seed using for generating random numbers for dropout
    int64_t rng_offset, // offset into random number sequence
    bool causal,
    const c10::optional<double> scale) {
#ifdef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(
      false,
      "MemoryEfficient build has been disabled at build time with -DXFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD");
#else
  at::globalContext().alertNotDeterministic(
      "mem_efficient_attention_backward_cutlass");

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

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t Mkv = key.size(1);
  int64_t N = key.size(1);
  int64_t nH = query.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  // It does not make sense to use that in practice,
  // but let's still make sure we are correct
  // As we iterate through keys first, we skip
  // keys with no query associated, so they are not
  // initialized
  bool grad_kv_needs_init = causal && N > M;
  const bool bias_requires_grad = bias.has_value() && bias->requires_grad();

  at::Tensor grad_q, grad_k, grad_v, grad_bias;
  if (!grad_kv_needs_init && query.size(1) == key.size(1) &&
      query.size(3) == value.size(3) &&
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
    grad_k = grad_kv_needs_init ? at::zeros(key.sizes(), key.options())
                                : at::empty(key.sizes(), key.options());
    grad_v = grad_kv_needs_init ? at::zeros(value.sizes(), value.options())
                                : at::empty(value.sizes(), value.options());
  }
  if (bias_requires_grad) {
    grad_bias = at::empty(bias->sizes(), bias->options());
  }
  at::Tensor workspace;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  at::PhiloxCudaState rng_engine_inputs(rng_seed, rng_offset);

  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = p->major * 10 + p->minor;

  bool kernel_launched = false;
  const auto maxK = std::max(query.size(3), value.size(3));
  const auto maxShmem =
      getMaximumSharedMemoryPerBlockKb(computeCapability) * 1024;

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
    p.num_queries = query.size(1);
    p.num_keys = key.size(1);
    p.num_batches = B;
    p.num_heads = nH;
    p.causal = causal;
    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(p.head_dim)));
    }

    ASSIGN_CHECK_OVERFLOW(p.lse_strideM, logsumexp.stride(1));
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

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));

      p.bias_ptr = (scalar_t*)bias->data_ptr();

      // assign strides for bias, viewed as:
      // (batch_sz, n_heads, n_queries, n_keys)
      const at::Tensor bias_4d_view = get_bias_4d_view(*bias, B, nH, M, N);
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias_4d_view.stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias_4d_view.stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias_4d_view.stride(2));

      if (bias_requires_grad) {
        p.grad_bias_ptr = (scalar_t*)grad_bias.data_ptr();

        // assign strides for gB, viewed as
        // (batch_sz, n_heads, n_queries, n_keys). might have different strides
        // than B, for example if bias tensor was created with
        // torch.tensor((B * nH, 1, nK)).expand((B * nH, nQ, nK)),
        // different values of Q will point to the same memory
        // locations, meaning bias.stride(1) == 0, while we'd want
        // grad_bias.stride(1) == nK
        const at::Tensor grad_bias_4d_view =
            get_bias_4d_view(grad_bias, B, nH, M, N);
        ASSIGN_CHECK_OVERFLOW(p.gB_strideB, grad_bias_4d_view.stride(0));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideH, grad_bias_4d_view.stride(1));
        ASSIGN_CHECK_OVERFLOW(p.gB_strideM, grad_bias_4d_view.stride(2));
      }
    }

    if (use_dropout) {
      p.rng_engine_inputs = rng_engine_inputs;
      p.dropout_prob = dropout_p;
    }

    int64_t size_bytes = p.workspace_size();
    if (size_bytes) {
      workspace =
          at::empty({size_bytes}, query.options().dtype(at::ScalarType::Byte));
      p.workspace = (float*)workspace.data_ptr();
    }
    Kernel::check_supported(p);

    if (smem_bytes > 0xc000) {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      XFORMERS_CHECK(
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
  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
#endif
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_cutlass"),
      TORCH_FN(mem_efficient_attention_backward_cutlass));
}
