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

#include "ck_fmha_params.h"
#include "ck_fmha_util.h"

extern void batched_forward_fp16(
    BatchedForwardParams& param,
    hipStream_t stream);
extern void batched_forward_bp16(
    BatchedForwardParams& param,
    hipStream_t stream);
extern void grouped_forward_fp16(
    GroupedForwardParams& param,
    hipStream_t stream);
extern void grouped_forward_bp16(
    GroupedForwardParams& param,
    hipStream_t stream);

namespace {

/*
  There are 2 modes for using this function.
  (Mode BMHK) With all the heads having the same seqlen
  (Mode 1MHK) `batch=1` with all tokens across batches concatenated
*/
std::tuple<at::Tensor, at::Tensor, int64_t, int64_t>
efficient_attention_forward_ck(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const c10::optional<at::Tensor>& bias, // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    double dropout_p, // attention matrix dropout probability
    bool compute_logsumexp,
    int64_t custom_mask_type,
    c10::optional<double> scale,
    const c10::optional<at::Tensor>& seqlen_k) {
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

  TORCH_CHECK(query.scalar_type() == key.scalar_type());
  TORCH_CHECK(query.scalar_type() == value.scalar_type());

  // Query, Key, Value must use the same CUDA device
  TORCH_CHECK(query.device() == key.device());
  TORCH_CHECK(query.device() == value.device());
  TORCH_CHECK(query.device().type() == torch::kCUDA)

  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
  };

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  hipStream_t stream = at::cuda::getCurrentHIPStream().stream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor out;
  at::Tensor logsumexp;
  at::Tensor randvals;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;
  at::PhiloxCudaState rng_engine_inputs;
  if (use_dropout) {
    at::CUDAGeneratorImpl* gen =
        at::get_generator_or_default<at::CUDAGeneratorImpl>(
            c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    std::lock_guard<std::mutex> lock(gen->mutex_);
    // if using dropout, we produce 1 random number for each element of the
    // attention tensor
    rng_engine_inputs = gen->philox_cuda_state(B * num_heads * M * N);
  }

  auto set_batched_forward_params = [&](BatchedForwardParams& p) {
    p.B = B;
    p.M = M;
    p.N = N;
    p.num_heads = num_heads;
    p.K = K;
    p.Kv = Kv;

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(K)));
    }

    p.q_ptr = query.data_ptr();
    p.k_ptr = key.data_ptr();
    p.v_ptr = value.data_ptr();
    p.out_ptr = out.data_ptr();

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

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(bias->scalar_type() == query.scalar_type());

      p.has_attn_bias = true;
      p.attn_bias_ptr = bias->data_ptr();

      const at::Tensor bias_4d_view =
          get_bias_4d_view(*bias, B, num_heads, M, N);
      p.attn_bias_strides = {
          static_cast<int>(bias_4d_view.stride(0)),
          static_cast<int>(bias_4d_view.stride(1)),
          static_cast<int>(bias_4d_view.stride(2)),
          static_cast<int>(bias_4d_view.stride(3))};
    } else
      p.has_attn_bias = false;

    p.custom_mask_type = custom_mask_type;

    p.use_dropout = use_dropout;
    p.compute_logsumexp = compute_logsumexp;

    // the following parameters are only used by training forward
    if (p.use_dropout) {
      p.dropout_prob = static_cast<float>(dropout_p);

      p.rng_engine_inputs = rng_engine_inputs;

      randvals = at::empty(
          {B, num_heads, M, N}, query.options().dtype(at::ScalarType::Short));
      p.randvals_strides = {
          static_cast<int>(randvals.stride(0)),
          static_cast<int>(randvals.stride(1)),
          static_cast<int>(randvals.stride(2)),
          static_cast<int>(randvals.stride(3))};
      p.randvals_ptr = randvals.data_ptr();
    } else {
      p.dropout_prob = 0.0f;
      p.randvals_ptr = nullptr;
    };

    if (p.compute_logsumexp) {
      logsumexp = at::empty(
          {B, num_heads, M}, query.options().dtype(at::ScalarType::Float));
      p.logsumexp_ptr = logsumexp.data_ptr();
    } else
      p.logsumexp_ptr = nullptr;
  };

  auto set_grouped_forward_params = [&](GroupedForwardParams& p) {
    p.num_batches = seqstart_q->size(0) - 1;
    p.M = M;
    p.N = N;
    p.num_heads = num_heads;
    p.K = K;
    p.Kv = Kv;

    if (scale.has_value()) {
      p.scale = float(*scale);
    } else {
      p.scale = float(1.0 / std::sqrt(float(K)));
    }

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

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(bias->scalar_type() == query.scalar_type());

      p.has_attn_bias = true;
      const at::Tensor bias_4d_view =
          get_bias_4d_view(*bias, B, num_heads, M, N);
      p.attn_bias_strides = {
          static_cast<int>(bias_4d_view.stride(0)),
          static_cast<int>(bias_4d_view.stride(1)),
          static_cast<int>(bias_4d_view.stride(2)),
          static_cast<int>(bias_4d_view.stride(3))};
    } else
      p.has_attn_bias = false;

    p.custom_mask_type = custom_mask_type;

    p.host_seqstart_q.resize(p.num_batches + 1);
    p.host_seqstart_k.resize(p.num_batches + 1);

    FMHA_HIP_CHECK(hipMemcpyAsync(
        p.host_seqstart_q.data(),
        seqstart_q->data_ptr(),
        (p.num_batches + 1) * sizeof(int32_t),
        hipMemcpyDeviceToHost,
        stream));
    FMHA_HIP_CHECK(hipMemcpyAsync(
        p.host_seqstart_k.data(),
        seqstart_k->data_ptr(),
        (p.num_batches + 1) * sizeof(int32_t),
        hipMemcpyDeviceToHost,
        stream));

    if (seqlen_k.has_value()) {
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      TORCH_CHECK(seqlen_k->dim() == 1);
      TORCH_CHECK(seqlen_k->size(0) == p.num_batches)
      CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqlen_k));

      p.host_seqlen_k.resize(p.num_batches);

      FMHA_HIP_CHECK(hipMemcpyAsync(
          p.host_seqlen_k.data(),
          seqlen_k->data_ptr(),
          p.num_batches * sizeof(int32_t),
          hipMemcpyDeviceToHost,
          stream));
    }

    char* q_ptr = reinterpret_cast<char*>(query.data_ptr());
    char* k_ptr = reinterpret_cast<char*>(key.data_ptr());
    char* v_ptr = reinterpret_cast<char*>(value.data_ptr());

    char* out_ptr = reinterpret_cast<char*>(out.data_ptr());
    char* attn_bias_ptr =
        bias.has_value() ? reinterpret_cast<char*>(bias->data_ptr()) : nullptr;

    for (int i = 0; i < p.num_batches; i++) {
      int32_t tmp_q_offset = get_size_in_bytes(
          p.host_seqstart_q[i] * p.q_strides[0], query.scalar_type());
      int32_t tmp_k_offset = get_size_in_bytes(
          p.host_seqstart_k[i] * p.k_strides[0], key.scalar_type());
      int32_t tmp_v_offset = get_size_in_bytes(
          p.host_seqstart_k[i] * p.v_strides[0], value.scalar_type());
      int32_t tmp_o_offset = get_size_in_bytes(
          p.host_seqstart_q[i] * p.out_strides[0], out.scalar_type());

      p.q_ptrs.push_back(reinterpret_cast<void*>(&q_ptr[tmp_q_offset]));
      p.k_ptrs.push_back(reinterpret_cast<void*>(&k_ptr[tmp_k_offset]));
      p.v_ptrs.push_back(reinterpret_cast<void*>(&v_ptr[tmp_v_offset]));
      p.out_ptrs.push_back(reinterpret_cast<void*>(&out_ptr[tmp_o_offset]));

      if (bias.has_value()) {
        int32_t tmp_bias_offset = get_size_in_bytes(
            p.host_seqstart_q[i] * p.attn_bias_strides[2] +
                p.host_seqstart_k[i] * p.attn_bias_strides[3],
            bias->scalar_type());

        p.attn_bias_ptrs.push_back(
            reinterpret_cast<void*>(&attn_bias_ptr[tmp_bias_offset]));
      };
    }

    p.use_dropout = use_dropout;
    p.compute_logsumexp = compute_logsumexp;

    // the following parameters are only used by training forward
    if (p.use_dropout) {
      p.dropout_prob = static_cast<float>(dropout_p);
      p.rng_engine_inputs = rng_engine_inputs;

      randvals = at::empty(
          {num_heads, M, N}, query.options().dtype(at::ScalarType::Short));
      p.randvals_strides = {
          static_cast<int>(randvals.stride(0)),
          static_cast<int>(randvals.stride(1)),
          static_cast<int>(randvals.stride(2))};
      char* randvals_ptr = reinterpret_cast<char*>(randvals.data_ptr());

      for (int i = 0; i < p.num_batches; i++) {
        int32_t tmp_randvals_stride = get_size_in_bytes(
            p.host_seqstart_q[i] * p.randvals_strides[1] +
                p.host_seqstart_k[i] * p.randvals_strides[2],
            randvals.scalar_type());

        p.randvals_ptrs.push_back(reinterpret_cast<void*>(randvals_ptr));
        randvals_ptr = randvals_ptr + tmp_randvals_stride;
      };
    } else
      p.dropout_prob = 0.0f;

    if (p.compute_logsumexp) {
      logsumexp = at::empty(
          {num_heads, M}, query.options().dtype(at::ScalarType::Float));
      char* logsumexp_ptr = reinterpret_cast<char*>(logsumexp.data_ptr());

      for (int i = 0; i < p.num_batches; i++) {
        int32_t tmp_logsumexp_stride =
            get_size_in_bytes(p.host_seqstart_q[i], logsumexp.scalar_type());

        p.logsumexp_ptrs.push_back(reinterpret_cast<void*>(logsumexp_ptr));
        logsumexp_ptr = logsumexp_ptr + tmp_logsumexp_stride;
      };
    };
  };

  // uint64_t -> int64_t bitwise casting as PyTorch don't support uint64_t
  // so just fake it as a int64_t
  int64_t seed, offset;

  DISPATCH_TYPES(query.scalar_type(), [&]() {
    out = at::zeros(
        {B, M, num_heads, Kv},
        query.options().dtype(CkToAtenDtype<scalar_t>::atScalarType()));

    if (!seqstart_q.has_value()) { // input is batched
      BatchedForwardParams batched_forward_params;

      set_batched_forward_params(batched_forward_params);

      if constexpr (std::is_same<scalar_t, ck::half_t>::value) {
        batched_forward_fp16(batched_forward_params, stream);
      } else if constexpr (std::is_same<scalar_t, ck::bhalf_t>::value) {
        batched_forward_bp16(batched_forward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported!");
    } else { // input is grouped
      GroupedForwardParams grouped_forward_params;

      set_grouped_forward_params(grouped_forward_params);

      if constexpr (std::is_same<scalar_t, ck::half_t>::value) {
        grouped_forward_fp16(grouped_forward_params, stream);
      } else if constexpr (std::is_same<scalar_t, ck::bhalf_t>::value) {
        grouped_forward_bp16(grouped_forward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported!");
    }
  });

  // torch::save(randvals, "randvals_dev.zip"); 

  std::memcpy(&seed, &rng_engine_inputs.seed_, sizeof(seed));
  std::memcpy(&offset, &rng_engine_inputs.offset_.val, sizeof(offset));

  return std::make_tuple(out, logsumexp, seed, offset);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_forward_ck"),
      TORCH_FN(efficient_attention_forward_ck));
}
