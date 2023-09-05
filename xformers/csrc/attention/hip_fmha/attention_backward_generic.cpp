#include <cmath>

#include <ATen/Context.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/TensorOperators.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "ck_fmha_params.h"
#include "ck_fmha_util.h"

extern void batched_backward_fp16(
    BatchedBackwardParams& param,
    hipStream_t stream);
extern void batched_backward_bp16(
    BatchedBackwardParams& param,
    hipStream_t stream);
extern void grouped_backward_fp16(
    GroupedBackwardParams& param,
    hipStream_t stream);
extern void grouped_backward_bp16(
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
    const c10::optional<at::Tensor>& seqlen_k,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    double dropout_p, // dropout probability
    int64_t rng_seed, // seed using for generating random numbers for dropout
    int64_t rng_offset, // offset into random number sequence
    int64_t custom_mask_type,
    const c10::optional<double> scale) {
#ifdef XFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD
  TORCH_CHECK(
      false,
      "MemoryEfficient build has been disabled at build time with -DXFORMERS_MEM_EFF_ATTENTION_DISABLE_BACKWARD");
#else
  at::globalContext().alertNotDeterministic(
      "mem_efficient_attention_backward_cutlass");

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
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));
  TORCH_CHECK(query.size(2) == grad_out.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));
  TORCH_CHECK(value.size(3) == grad_out.size(3));

  // Query, Key, Value must use the same CUDA device
  TORCH_CHECK(query.device() == key.device());
  TORCH_CHECK(query.device() == value.device());
  TORCH_CHECK(query.device().type() == torch::kCUDA)

  // handle potentially non-contiguous grad_out through a copy
  CHECK_NOSPARSE_CONTIGUOUS_CUDA(grad_out);

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

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
  }

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(2);
  int64_t K = query.size(3);
  int64_t Kv = value.size(3);

  at::Tensor grad_q, grad_k, grad_v, grad_bias;

  grad_q = at::empty(query.sizes(), query.options());
  grad_k = at::empty(key.sizes(), key.options());
  grad_v = at::empty(value.sizes(), value.options());

  at::Tensor randvals;

  auto set_batched_backward_params = [&](BatchedBackwardParams& p) {
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
    p.grad_out_ptr = grad_out.data_ptr();
    p.grad_q_ptr = grad_q.data_ptr();
    p.grad_k_ptr = grad_k.data_ptr();
    p.grad_v_ptr = grad_v.data_ptr();

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
    p.grad_out_strides = {
        static_cast<int>(grad_out.stride(0)),
        static_cast<int>(grad_out.stride(1)),
        static_cast<int>(grad_out.stride(2)),
        static_cast<int>(grad_out.stride(3))};

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
      p.attn_bias_ptr = nullptr;

    p.custom_mask_type = custom_mask_type;

    p.dropout_prob = static_cast<float>(dropout_p);
    p.philox_seed = rng_seed;
    p.philox_offset = rng_offset;

    randvals = at::empty(
        {B, num_heads, M, N}, query.options().dtype(at::ScalarType::Short));
    p.randvals_strides = {
        static_cast<int>(randvals.stride(0)),
        static_cast<int>(randvals.stride(1)),
        static_cast<int>(randvals.stride(2)),
        static_cast<int>(randvals.stride(3))};
    p.randvals_ptr = randvals.data_ptr();

    p.logsumexp_ptr = logsumexp.data_ptr();
  };

  auto set_grouped_backward_params = [&](GroupedBackwardParams& p) {
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

    p.grad_out_strides = {
        static_cast<int>(grad_out.stride(1)),
        static_cast<int>(grad_out.stride(2)),
        static_cast<int>(grad_out.stride(3))};

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

    p.dropout_prob = static_cast<float>(dropout_p);
    p.philox_seed = rng_seed;
    p.philox_offset = rng_offset;

    randvals = at::empty(
        {num_heads, M, N}, query.options().dtype(at::ScalarType::Short));
    p.randvals_strides = {
        static_cast<int>(randvals.stride(0)),
        static_cast<int>(randvals.stride(1)),
        static_cast<int>(randvals.stride(2))};

    p.custom_mask_type = custom_mask_type;

    p.host_seqstart_q.resize(p.num_batches + 1);
    p.host_seqstart_k.resize(p.num_batches + 1);

    FMHA_HIP_CHECK(hipMemcpyAsync(
        p.host_seqstart_q.data(),
        seqstart_q->data_ptr(),
        (p.num_batches + 1) * sizeof(int),
        hipMemcpyDeviceToHost,
        stream));
    FMHA_HIP_CHECK(hipMemcpyAsync(
        p.host_seqstart_k.data(),
        seqstart_k->data_ptr(),
        (p.num_batches + 1) * sizeof(int),
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
    char* grad_out_ptr = reinterpret_cast<char*>(grad_out.data_ptr());
    char* attn_bias_ptr = reinterpret_cast<char*>(bias->data_ptr());

    char* logsumexp_ptr = reinterpret_cast<char*>(logsumexp.data_ptr());
    char* randvals_ptr = reinterpret_cast<char*>(randvals.data_ptr());

    char* grad_q_ptr = reinterpret_cast<char*>(grad_q.data_ptr());
    char* grad_k_ptr = reinterpret_cast<char*>(grad_k.data_ptr());
    char* grad_v_ptr = reinterpret_cast<char*>(grad_v.data_ptr());

    for (int i = 0; i < p.num_batches; i++) {
      int32_t tmp_q_stride = get_size_in_bytes(
          p.host_seqstart_q[i] * p.q_strides[0], query.scalar_type());
      int32_t tmp_k_stride = get_size_in_bytes(
          p.host_seqstart_k[i] * p.k_strides[0], key.scalar_type());
      int32_t tmp_v_stride = get_size_in_bytes(
          p.host_seqstart_k[i] * p.v_strides[0], value.scalar_type());
      int32_t tmp_o_stride = get_size_in_bytes(
          p.host_seqstart_q[i] * p.out_strides[0], out.scalar_type());
      int32_t tmp_grad_o_stride = get_size_in_bytes(
          p.host_seqstart_q[i] * p.grad_out_strides[0], grad_out.scalar_type());
      int32_t tmp_logsumexp_stride =
          get_size_in_bytes(p.host_seqstart_q[i], logsumexp.scalar_type());
      int32_t tmp_randvals_stride = get_size_in_bytes(
          p.host_seqstart_q[i] * p.randvals_strides[1] +
              p.host_seqstart_k[i] * p.randvals_strides[2],
          randvals.scalar_type());

      p.q_ptrs.push_back(reinterpret_cast<void*>(&q_ptr[tmp_q_stride]));
      p.grad_q_ptrs.push_back(
          reinterpret_cast<void*>(&grad_q_ptr[tmp_q_stride]));
      p.k_ptrs.push_back(reinterpret_cast<void*>(&k_ptr[tmp_k_stride]));
      p.grad_k_ptrs.push_back(
          reinterpret_cast<void*>(&grad_k_ptr[tmp_k_stride]));
      p.v_ptrs.push_back(reinterpret_cast<void*>(&v_ptr[tmp_v_stride]));
      p.grad_v_ptrs.push_back(
          reinterpret_cast<void*>(&grad_v_ptr[tmp_v_stride]));
      p.out_ptrs.push_back(reinterpret_cast<void*>(&out_ptr[tmp_o_stride]));
      p.grad_out_ptrs.push_back(
          reinterpret_cast<void*>(&grad_out_ptr[tmp_grad_o_stride]));

      if (bias.has_value()) {
        int32_t tmp_bias_stride = get_size_in_bytes(
            p.host_seqstart_q[i] * p.attn_bias_strides[2] +
                p.host_seqstart_k[i] * p.attn_bias_strides[3],
            bias->scalar_type());

        p.attn_bias_ptrs.push_back(
            reinterpret_cast<void*>(&attn_bias_ptr[tmp_bias_stride]));
      };

      p.logsumexp_ptrs.push_back(
          reinterpret_cast<void*>(&logsumexp_ptr[tmp_logsumexp_stride]));
      p.randvals_ptrs.push_back(
          reinterpret_cast<void*>(&randvals_ptr[tmp_randvals_stride]));
    }
  };

  DISPATCH_TYPES(query.scalar_type(), [&]() {
    if (!seqstart_q.has_value()) { // input is batched
      BatchedBackwardParams batched_backward_params;

      set_batched_backward_params(batched_backward_params);

      if constexpr (std::is_same<scalar_t, ck::half_t>::value) {
        batched_backward_fp16(batched_backward_params, stream);
      } else if constexpr (std::is_same<scalar_t, ck::bhalf_t>::value) {
        batched_backward_bp16(batched_backward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported");
    } else { // input is grouped
      GroupedBackwardParams grouped_backward_params;

      set_grouped_backward_params(grouped_backward_params);

      if constexpr (std::is_same<scalar_t, ck::half_t>::value) {
        grouped_backward_fp16(grouped_backward_params, stream);
      } else if constexpr (std::is_same<scalar_t, ck::bhalf_t>::value) {
        grouped_backward_bp16(grouped_backward_params, stream);
      } else
        throw std::runtime_error("input data-type is not supported");
    }
  });

  return std::make_tuple(grad_q, grad_k, grad_v, grad_bias);
#endif
} // namespace

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward_ck"),
      TORCH_FN(efficient_attention_backward_ck));
}
