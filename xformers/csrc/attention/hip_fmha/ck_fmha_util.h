#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <torch/torch.h>

#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/sequence.hpp>
#include <ATen/cuda/CUDAGeneratorImpl.h>

// Here flag can be a constant, variable or function call
#define FMHA_HIP_CHECK(ret_or_call)                                          \
  do {                                                                       \
    hipError_t _tmpVal;                                                      \
    if ((_tmpVal = ret_or_call) != hipSuccess) {                             \
      std::ostringstream ostr;                                               \
      ostr << "HIP Function Failed (" << __FILE__ << "," << __LINE__ << ") " \
           << hipGetErrorString(_tmpVal);                                    \
      throw std::runtime_error(ostr.str());                                  \
    }                                                                        \
  } while (0)

#define XFORMERS_CHECK(COND, ERR)          \
  if (!(COND)) {                           \
    std::ostringstream ostr;               \
    ostr << "'" #COND "' failed: " << ERR; \
    throw std::runtime_error(ostr.str());  \
  }

#define DISPATCH_TYPES(InDataType, func)                                 \
  {                                                                      \
    if (InDataType == at::ScalarType::Half) {                            \
      using scalar_t = ck::half_t;                                       \
      func();                                                            \
    } else if (InDataType == at::ScalarType::BFloat16) {                 \
      using scalar_t = ck::bhalf_t;                                      \
      func();                                                            \
    } else {                                                             \
      XFORMERS_CHECK(                                                    \
          false, "Only half & bf16 input type supported at the moment"); \
    }                                                                    \
  }

template <typename scalar_t>
struct CkToAtenDtype;

template <>
struct CkToAtenDtype<ck::half_t> {
  using scalar_t = ck::half_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Half;
  }
};

template <>
struct CkToAtenDtype<ck::bhalf_t> {
  using scalar_t = ck::bhalf_t;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::BFloat16;
  }
};

template <>
struct CkToAtenDtype<float> {
  using scalar_t = float;

  static constexpr __host__ at::ScalarType atScalarType() {
    return at::ScalarType::Float;
  }
};

#define CHECK_NOSPARSE_CONTIGUOUS_CUDA(TENSOR)                            \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(TENSOR.is_contiguous(), #TENSOR " must be contiguous");

#define CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(TENSOR)                        \
  XFORMERS_CHECK(TENSOR.is_cuda(), #TENSOR " must be a CUDA tensor");     \
  XFORMERS_CHECK(!TENSOR.is_sparse(), #TENSOR " must be a dense tensor"); \
  XFORMERS_CHECK(                                                         \
      TENSOR.stride(-1) == 1, #TENSOR ": last dimension must be contiguous");

static inline size_t get_size_in_bytes(size_t n, at::ScalarType dtype) {
  if (dtype == at::ScalarType::Float) {
    return n * 4;
  } else if (dtype == at::ScalarType::Half) {
    return n * 2;
  } else if (dtype == at::ScalarType::BFloat16) {
    return n * 2;
  } else if (dtype == at::ScalarType::Short) {
    return n * 2;
  } else if (dtype == at::ScalarType::Int) {
    return n * 4;
  } else if (dtype == at::ScalarType::Byte) {
    return n;
  }
  return 0;
}

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
inline at::Tensor get_bias_4d_view(
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

template <typename scalar_t>
struct MaxVectorSizeForType {
  static constexpr int value = 4;
};

template <>
struct MaxVectorSizeForType<ck::half_t> {
  static constexpr int value = 8;
};

template <>
struct MaxVectorSizeForType<ck::bhalf_t> {
  static constexpr int value = 8;
};

struct SimpleDeviceMem {
  SimpleDeviceMem() = delete;
  SimpleDeviceMem(std::size_t mem_size) : p_mem_{} {
    FMHA_HIP_CHECK(hipMalloc(static_cast<void**>(&p_mem_), mem_size));
  }
  void* GetDeviceBuffer() {
    return p_mem_;
  }
  ~SimpleDeviceMem() {
    (void)hipFree(p_mem_);
  }

  void* p_mem_;
};

struct BatchedInferParams {
  int B; // batch size
  int M; // seq_len for Query
  int N; // seq_len for Key and Value
  int num_heads; //
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

  uint8_t custom_mask_type;

  void* out_ptr;
};

struct BatchedForwardParams : public BatchedInferParams {
  bool use_dropout;
  bool compute_logsumexp;

  float dropout_prob;
  at::PhiloxCudaState rng_engine_inputs;

  // BHMN mode strides, completely contiguous
  std::array<int32_t, 4> randvals_strides;
  void* randvals_ptr;

  // completely contiguous
  void* logsumexp_ptr;
};

struct GroupedInferParams {
  int num_batches;
  int M; // total seq_len for all queries in the batch
  int N; // total seq_len for all keys/values in the batch
  int num_heads; //
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  std::vector<int> host_seqstart_q;
  std::vector<int> host_seqstart_k;
  std::vector<int> host_seqlen_k;

  float scale;
  bool has_attn_bias;

  // MHK mode strides, last-dim contiguous
  std::array<int, 3> q_strides;
  std::array<int, 3> k_strides;
  std::array<int, 3> v_strides;
  std::array<int, 3> out_strides;

  // 4d tensor view [B, H, M, N]
  std::array<int, 4> attn_bias_strides;

  std::vector<const void*> q_ptrs;
  std::vector<const void*> k_ptrs;
  std::vector<const void*> v_ptrs;
  std::vector<const void*> attn_bias_ptrs;
  std::vector<void*> out_ptrs;

  uint8_t custom_mask_type;
};

struct GroupedForwardParams : public GroupedInferParams {
  bool use_dropout;
  bool compute_logsumexp;

  float dropout_prob;
  at::PhiloxCudaState rng_engine_inputs;

  // HMN mode strides, completely contiguous
  std::array<int, 3> randvals_strides;
  std::vector<void*> randvals_ptrs;

  // completely contiguous
  std::vector<void*> logsumexp_ptrs;
};

struct BatchedBackwardParams {
  int B; // batch size
  int M; // seq_len for Query
  int N; // seq_len for Key and Value
  int num_heads; //
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  float scale;

  // BMHK mode strides, last-dim contiguous
  std::array<int, 4> q_strides;
  std::array<int, 4> k_strides;
  std::array<int, 4> v_strides;
  std::array<int, 4> attn_bias_strides; // 4d tensor_view [B, H, M, N]
  std::array<int, 4> out_strides;

  const void* q_ptr;
  const void* k_ptr;
  const void* v_ptr;
  const void* attn_bias_ptr;
  const void* out_ptr;

  uint8_t custom_mask_type;

  std::array<int, 4> grad_out_strides;

  const void* grad_out_ptr;

  void* grad_q_ptr;
  void* grad_k_ptr;
  void* grad_v_ptr;
  // void* grad_bias_ptr;

  float dropout_prob;
  at::PhiloxCudaState rng_engine_inputs;

  // completely contiguous
  const void* logsumexp_ptr;

  // BHMN mode strides, completely contiguous
  std::array<int, 4> randvals_strides;
  void* randvals_ptr;

  int64_t rng_seed;
  int64_t rng_offset;
};

struct GroupedBackwardParams {
  int num_batches;
  int M; // total seq_len for all queries in the batch
  int N; // total seq_len for all keys/values in the batch
  int num_heads; //
  int K; // embed_dim for Query and Key
  int Kv; // embed_dim for Value

  std::vector<int> host_seqstart_q;
  std::vector<int> host_seqstart_k;
  std::vector<int> host_seqlen_k;

  float scale;

  // MHK mode strides, last-dim contiguous
  std::array<int, 3> q_strides;
  std::array<int, 3> k_strides;
  std::array<int, 3> v_strides;
  std::array<int, 3> out_strides;
  // 4d tensor view [B, H, M, N]
  std::array<int, 4> attn_bias_strides;

  std::vector<const void*> q_ptrs;
  std::vector<const void*> k_ptrs;
  std::vector<const void*> v_ptrs;
  std::vector<const void*> attn_bias_ptrs;
  std::vector<const void*> out_ptrs;

  uint8_t custom_mask_type;

  std::array<int, 3> grad_out_strides;

  std::vector<const void*> grad_out_ptrs;

  std::vector<void*> grad_q_ptrs;
  std::vector<void*> grad_k_ptrs;
  std::vector<void*> grad_v_ptrs;
  // std::vector<void *> grad_bias_ptrs;

  float dropout_prob;
  at::PhiloxCudaState rng_engine_inputs;

  // HM mode strides, completely contiguous
  std::vector<const void*> logsumexp_ptrs;

  // HMN mode strides, completely contiguous
  std::array<int, 3> randvals_strides;
  std::vector<void*> randvals_ptrs;

  int64_t rng_seed;
  int64_t rng_offset;
};

// useful aliasing for making the codes easy
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;
