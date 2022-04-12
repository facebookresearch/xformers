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
void attention_kernel(
    at::TensorAccessor<scalar_t, 3> output,
    at::TensorAccessor<scalar_t, 3> query,
    at::TensorAccessor<scalar_t, 3> key,
    at::TensorAccessor<scalar_t, 3> value,
    at::TensorAccessor<scalar_t, 3> buffer //,
    // at::TensorAccessor<int64_t, 2> mask
) {
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
      }
    }
  });
}

at::Tensor attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value
    // const at::Tensor& mask
) {
  TORCH_CHECK(query.dim() == key.dim());
  TORCH_CHECK(query.dim() == value.dim());
  // TORCH_CHECK(query.dim() == mask.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.size(0) == value.size(0));
  TORCH_CHECK(key.size(1) == value.size(1));
  TORCH_CHECK(
      query.size(2) ==
      value.size(2)); // TODO: drop this limitation in the future

  TORCH_CHECK(!query.is_cuda(), "query must be a CPU tensor");
  TORCH_CHECK(!key.is_cuda(), "key must be a CPU tensor");
  TORCH_CHECK(!value.is_cuda(), "value must be a CPU tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res = at::empty({B, M, K}, query.options());

  at::Tensor buffer = at::empty({at::get_num_threads(), 1, K}, query.options());

  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "attention_kernel", [&] {
    attention_kernel<scalar_t>(
        res.accessor<scalar_t, 3>(),
        query.accessor<scalar_t, 3>(),
        key.accessor<scalar_t, 3>(),
        value.accessor<scalar_t, 3>(),
        buffer.accessor<scalar_t, 3>());
  });

  return res;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
}
