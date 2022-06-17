#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>

namespace {

template <typename scalar_t>
void fill_zero(scalar_t* buf, int64_t K) {
  for (int64_t k = 0; k < K; k++) {
    buf[k] = 0;
  }
}

template <typename scalar_t, int K>
scalar_t max(scalar_t* buf) {
  scalar_t m = buf[0];
  for (int64_t k = 1; k < K; k++) {
    m = buf[k] > m ? buf[k] : m;
  }
  return m;
}

template <typename scalar_t>
at::TensorAccessor<scalar_t, 3> _tensor_accessor_or_dummy(
    const at::Tensor& attn_bias,
    const std::array<int64_t, 3> zeros) {
  if (attn_bias.defined()) {
    return attn_bias.accessor<scalar_t, 3>();
  } else {
    return at::TensorAccessor<scalar_t, 3>(nullptr, zeros.data(), zeros.data());
  }
}

template <typename scalar_t>
void attention_kernel(
    at::TensorAccessor<scalar_t, 3> output,
    at::TensorAccessor<scalar_t, 2> logsumexp,
    at::TensorAccessor<scalar_t, 3> query,
    at::TensorAccessor<scalar_t, 3> key,
    at::TensorAccessor<scalar_t, 3> value,
    at::TensorAccessor<scalar_t, 3> buffer,
    bool compute_logsumexp,
    at::TensorAccessor<scalar_t, 3> attn_bias) {
  // TODO: optimize the code by adding blocking
  // over multiple dimensions. Doing this allows
  // the compiler to group reads and operations
  // for vectorization
  constexpr int64_t BLOCK = 1; // 8;
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t grain_size = 1;
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));
  at::parallel_for(0, B, grain_size, [&](int64_t start, int64_t end) {
    auto buf = buffer[at::get_thread_num()][0].data();
    for (int64_t i = start; i < end; i++) {
      for (int64_t j = 0; j < M; j++) {
        fill_zero<scalar_t>(buf, K);
        auto aar = query[i][j].data();
        scalar_t s_prime = 0;
        scalar_t m_prime = -std::numeric_limits<scalar_t>::infinity();
        for (int64_t l = 0; l < N; l += BLOCK) {
          auto bar = key[i][l].data();
          scalar_t si[BLOCK] = {0};
          for (int64_t k = 0; k < K; k++) {
            auto aaar = aar[k] * scale;
            for (int64_t rr = 0; rr < BLOCK; rr++)
              si[rr] += aaar * bar[k + K * rr];
          }
          if (attn_bias.data() != nullptr) {
            for (int64_t rr = 0; rr < BLOCK; rr++) {
              si[rr] += attn_bias[i][j][l + rr];
            }
          }

          scalar_t m_i = si[0] > m_prime ? si[0] : m_prime;
          for (int64_t rr = 1; rr < BLOCK; rr++) {
            m_i = si[rr] > m_i ? si[rr] : m_i;
          }

          auto vi = value[i][l].data();

          scalar_t m_delta;
          scalar_t s_delta[BLOCK];
          m_delta = std::exp(m_prime - m_i);

          for (int64_t rr = 0; rr < BLOCK; rr++)
            s_delta[rr] = std::exp(si[rr] - m_i);

          for (int64_t k = 0; k < K; k++) {
            buf[k] = buf[k] * m_delta;
            for (int64_t rr = 0; rr < BLOCK; rr++)
              buf[k] += vi[k + K * rr] * s_delta[rr];
          }
          s_prime = s_prime * m_delta;
          for (int64_t rr = 0; rr < BLOCK; rr++)
            s_prime += s_delta[rr];

          m_prime = m_i;
        }
        auto oo = output[i][j].data();
        for (int64_t k = 0; k < K; k++) {
          oo[k] = buf[k] / s_prime;
        }
        if (compute_logsumexp)
          logsumexp[i][j] = m_prime + std::log(s_prime);
      }
    }
  });
}

std::tuple<at::Tensor, at::Tensor, int64_t, int64_t> attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    const c10::optional<at::Tensor>& attn_bias_,
    double p) {
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  TORCH_CHECK(!query.is_cuda(), "query must be a CPU tensor");
  TORCH_CHECK(!key.is_cuda(), "key must be a CPU tensor");
  TORCH_CHECK(!value.is_cuda(), "value must be a CPU tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  TORCH_CHECK(p == 0, "CPU implementation does not support dropout");

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::empty({B, M, K}, query.options());
  at::Tensor logsumexp = at::empty({B, M}, query.options());

  at::Tensor buffer = at::empty({at::get_num_threads(), 1, K}, query.options());
  const std::array<int64_t, 3> zeros{{0}};

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "attention_kernel", [&] {
    attention_kernel<scalar_t>(
        res.accessor<scalar_t, 3>(),
        logsumexp.accessor<scalar_t, 2>(),
        query.accessor<scalar_t, 3>(),
        key.accessor<scalar_t, 3>(),
        value.accessor<scalar_t, 3>(),
        buffer.accessor<scalar_t, 3>(),
        compute_logsumexp,
        _tensor_accessor_or_dummy<scalar_t>(attn_bias, zeros));
  });

  return std::make_tuple(res, logsumexp, 1, 1);
}

template <typename scalar_t>
void attention_backward_kernel(
    at::TensorAccessor<scalar_t, 3> grad_q,
    at::TensorAccessor<scalar_t, 3> grad_k,
    at::TensorAccessor<scalar_t, 3> grad_v,
    at::TensorAccessor<scalar_t, 3> grad_out,
    at::TensorAccessor<scalar_t, 3> q,
    at::TensorAccessor<scalar_t, 3> k,
    at::TensorAccessor<scalar_t, 3> v,
    at::TensorAccessor<scalar_t, 2> logsumexp_normalizer,
    at::TensorAccessor<scalar_t, 3> buffer,
    at::TensorAccessor<scalar_t, 3> buffer2,
    at::TensorAccessor<scalar_t, 3> attn_bias) {
  int64_t K = q.size(2);
  int64_t B = q.size(0);
  int64_t M = q.size(1);
  int64_t N = k.size(1);
  int64_t grain_size = 1; // buffer.size(1);
  scalar_t scale = 1.0 / std::sqrt(scalar_t(K));
  at::parallel_for(0, B, grain_size, [&](int64_t start, int64_t end) {
    auto buf = buffer[at::get_thread_num()][0];
    auto buf2 = buffer2[at::get_thread_num()][0];
    for (int64_t i = start; i < end; i++) {
      for (int64_t j = 0; j < M; j++) {
        for (int64_t k = 0; k < K; k++) {
          buf[k] = 0;
        }
        auto query_i = q[i][j];
        auto normalizer = logsumexp_normalizer[i][j];
        scalar_t tmp_sum = 0;
        for (int64_t l = 0; l < N; l++) {
          auto key_j = k[i][l];
          scalar_t si = 0;
          for (int64_t k = 0; k < K; k++) {
            si += query_i[k] * key_j[k];
          }
          scalar_t attn_b =
              attn_bias.data() == nullptr ? scalar_t(0) : attn_bias[i][j][l];
          scalar_t attn_v = std::exp(si * scale - normalizer + attn_b);

          for (int64_t k = 0; k < K; k++) {
            grad_v[i][l][k] += attn_v * grad_out[i][j][k];
          }

          // now compute grad_q and grad_k
          // first compute the gradient for the self-attention
          // after softmax
          scalar_t grad_attn_v = 0;
          for (int64_t k = 0; k < K; k++) {
            grad_attn_v += grad_out[i][j][k] * v[i][l][k];
            // grad_attn_v[i][j][l] += grad_out[i][j][k] * v[i][l][k];
          }

          // those are temporaries for the gradient of the softmax
          scalar_t tmp = attn_v * grad_attn_v * scale;
          tmp_sum += tmp;

          // grad_q is easy
          for (int64_t k = 0; k < K; k++) {
            grad_q[i][j][k] += tmp * key_j[k];
            buf[k] += attn_v * key_j[k];
          }

          //  but grad_k is a bit trickier
          buf2[l] = attn_v;
          for (int64_t k = 0; k < K; k++) {
            grad_k[i][l][k] += tmp * query_i[k];
          }
        }
        for (int64_t l = 0; l < N; l++) {
          for (int64_t k = 0; k < K; k++) {
            grad_k[i][l][k] -= buf2[l] * query_i[k] * tmp_sum;
          }
        }
        for (int64_t k = 0; k < K; k++) {
          grad_q[i][j][k] -= buf[k] * tmp_sum;
        }
      }
    }
  });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> attention_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& output,
    const c10::optional<at::Tensor>& attn_bias_,
    double p,
    int64_t rng_seed,
    int64_t rng_offset) {
  TORCH_CHECK(query.dim() == grad_out.dim());
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  TORCH_CHECK(query.dim() == 3);

  TORCH_CHECK(query.size(0) == grad_out.size(0));
  TORCH_CHECK(query.size(1) == grad_out.size(1));
  TORCH_CHECK(query.size(2) == grad_out.size(2));

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  at::Tensor attn_bias;
  if (attn_bias_.has_value()) {
    attn_bias = *attn_bias_;
    TORCH_CHECK(query.dim() == attn_bias.dim());
    TORCH_CHECK(query.size(0) == attn_bias.size(0));
    TORCH_CHECK(query.size(1) == attn_bias.size(1));
    TORCH_CHECK(key.size(1) == attn_bias.size(2));
    TORCH_CHECK(attn_bias.stride(1) == 0);
  }

  TORCH_CHECK(!query.is_cuda(), "query must be a CPU tensor");
  TORCH_CHECK(!key.is_cuda(), "key must be a CPU tensor");
  TORCH_CHECK(!value.is_cuda(), "value must be a CPU tensor");
  TORCH_CHECK(!grad_out.is_cuda(), "grad_out must be a CPU tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");
  TORCH_CHECK(!grad_out.is_sparse(), "grad_out must be a dense tensor");

  TORCH_CHECK(p == 0, "CPU implementation does not support dropout");

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::empty({B, M, K}, query.options());
  at::Tensor grad_q = at::zeros_like(query);
  at::Tensor grad_k = at::zeros_like(key);
  at::Tensor grad_v = at::zeros_like(value);

  at::Tensor buffer = at::empty({at::get_num_threads(), 1, K}, query.options());
  at::Tensor buffer2 =
      at::zeros({at::get_num_threads(), 1, N}, query.options());

  const std::array<int64_t, 3> zeros{{0}};

  AT_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "attention_backward_kernel", [&] {
        attention_backward_kernel<scalar_t>(
            grad_q.accessor<scalar_t, 3>(),
            grad_k.accessor<scalar_t, 3>(),
            grad_v.accessor<scalar_t, 3>(),
            grad_out.accessor<scalar_t, 3>(),
            query.accessor<scalar_t, 3>(),
            key.accessor<scalar_t, 3>(),
            value.accessor<scalar_t, 3>(),
            logsumexp.accessor<scalar_t, 2>(),
            buffer.accessor<scalar_t, 3>(),
            buffer2.accessor<scalar_t, 3>(),
            _tensor_accessor_or_dummy<scalar_t>(attn_bias, zeros));
      });

  return std::make_tuple(grad_q, grad_k, grad_v);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention_backward"),
      TORCH_FN(attention_backward));
}
