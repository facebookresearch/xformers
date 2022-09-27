#include "kernel_forward.h"

#define DISPATCH_BLOCKSIZE(VALUE_HEAD_DIM, FN)        \
  {                                                   \
    if (VALUE_HEAD_DIM <= 64) {                       \
      constexpr bool kIs64x64 = true;                 \
      constexpr bool kSingleValueIteration = true;    \
      FN();                                           \
    } else {                                          \
      constexpr bool kIs64x64 = false;                \
      if (VALUE_HEAD_DIM <= 128) {                    \
        constexpr bool kSingleValueIteration = true;  \
        FN();                                         \
      } else {                                        \
        constexpr bool kSingleValueIteration = false; \
        FN();                                         \
      }                                               \
    }                                                 \
  }

#define DISPATCH_KERNEL(QUERY, KEY, VALUE, FUNC)                              \
  {                                                                           \
    cudaDeviceProp* properties =                                              \
        at::cuda::getDeviceProperties(QUERY.device().index());                \
    const int computeCapability = properties->major * 10 + properties->minor; \
    DISPATCH_BLOCKSIZE(                                                       \
        VALUE.size(2), ([&]() {                                               \
          static constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;     \
          static constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;       \
          DISPATCH_TYPES(                                                     \
              QUERY, ([&]() {                                                 \
                DISPATCH_ARCHTAG(                                             \
                    computeCapability, ([&]() {                               \
                      using AlignedAK = AttentionKernel<                      \
                          scalar_t,                                           \
                          ArchTag,                                            \
                          true,                                               \
                          kQueriesPerBlock,                                   \
                          kKeysPerBlock,                                      \
                          kSingleValueIteration>;                             \
                      /* Run a more efficient kernel (with `isAligned=True`)  \
                      if memory is correctly aligned*/                        \
                      bool isAligned =                                        \
                          (QUERY.stride(1) % AlignedAK::kAlignmentQ == 0 &&   \
                           KEY.stride(1) % AlignedAK::kAlignmentK == 0 &&     \
                           VALUE.stride(1) % AlignedAK::kAlignmentV == 0);    \
                      /* TODO: Should we warn or log somewhere when we use a  \
                      less efficient kernel due to wrong alignment? */        \
                      DISPATCH_BOOL(isAligned, kIsAligned, ([&]() {           \
                                      using Kernel = AttentionKernel<         \
                                          scalar_t,                           \
                                          ArchTag,                            \
                                          kIsAligned,                         \
                                          kQueriesPerBlock,                   \
                                          kKeysPerBlock,                      \
                                          kSingleValueIteration>;             \
                                      FUNC();                                 \
                                    }))                                       \
                    }))                                                       \
              }));                                                            \
        }));                                                                  \
  }

namespace {
std::tuple<at::Tensor, at::Tensor> efficient_attention_forward_cutlass(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    bool causal) {
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(key.dim() == 3);
  TORCH_CHECK(value.dim() == 3);

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  CHECK_NOSPARSE_CONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res;
  at::Tensor logsumexp;

  auto launchKernel = [&](auto _k, int computeCapability) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    res = at::empty(
        {B, M, K},
        query.options().dtype(
            TypeTraits<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {B, compute_logsumexp ? ceil_div(M, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query.data_ptr();
    p.key_ptr = (scalar_t*)key.data_ptr();
    p.value_ptr = (scalar_t*)value.data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, K},
          query.options().dtype(
              TypeTraits<typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();
    p.head_dim = query.size(2);
    p.head_dim_value = value.size(2);
    p.num_queries = query.size(1);
    p.num_keys = key.size(1);
    p.num_batches = B;
    p.causal = causal;

    constexpr auto kernel_fn = attention_kernel_batched<Kernel>;
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > 0xc000) {
      TORCH_INTERNAL_ASSERT(
          computeCapability >= 70,
          "This kernel requires too much shared memory on this machine!");
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    Kernel::check_supported(p);
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes>>>(p);
  };
  // Dispatch to the right kernel
  DISPATCH_KERNEL(query, key, value, ([&]() {
                    launchKernel(Kernel{}, computeCapability);
                  }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp);
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_cutlass"),
      TORCH_FN(efficient_attention_forward_cutlass));
}
