#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <vector>

#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>


namespace {

template <typename scalar_t>
void attention_kernel(
    at::TensorAccessor<scalar_t, 3> output,
    at::TensorAccessor<scalar_t, 3> q,
    at::TensorAccessor<scalar_t, 3> k,
    at::TensorAccessor<scalar_t, 3> v,
    at::TensorAccessor<scalar_t, 3> buffer//,
    //at::TensorAccessor<int64_t, 2> mask
    ) {

  int64_t K = q.size(2);
  int64_t B = q.size(0);
  int64_t M = q.size(1);
  int64_t N = k.size(1);
  int64_t grain_size = 1;//buffer.size(1);
  at::parallel_for(0, B, grain_size, [&](int64_t start, int64_t end) {
    using Vec = at::vec::Vectorized<scalar_t>;

    auto buf = buffer[at::get_thread_num()][0].data();
    for (int64_t i = start; i < end; i++) {
    //for (int64_t ii = start; ii < end; ii++) {
      //int64_t i = ii / M;
      //int64_t j = ii % M;

      //{
      for (int64_t j = 0; j < M; j++) {
        //auto buf = buffer[i][at::get_thread_num()];
        for (int64_t k = 0; k < K; k++) {
          buf[k] = 0;
        }
        //std::memset(buf.data(), 0, K * sizeof(scalar_t));
        auto aar = q[i][j].data();
        scalar_t s_prime = 0;
        scalar_t m_prime = -std::numeric_limits<scalar_t>::infinity();
        for (int64_t l = 0; l < N; l++) {
          auto bar = k[i][l].data();
          scalar_t si = 0;
          for (int64_t k = 0; k < K; k++) {
            si += aar[k] * bar[k];
          }
          /*
          si = at::vec::map2_reduce_all<scalar_t>(
            [](Vec a, Vec b) { return a * b; },
            [](Vec x, Vec y) { return x + y; },
            aar,
            bar,
            K
          );
          */
          scalar_t m_i = si > m_prime ? si : m_prime;
          auto vi = v[i][l].data();

          scalar_t m_delta = std::exp(m_prime - m_i);
          scalar_t s_delta = std::exp(si - m_i);

          for (int64_t k = 0; k < K; k++) {
            buf[k] = buf[k] * m_delta + vi[k] * s_delta;
          }
          //at::vec::map2<scalar_t>(
          //  [m_delta, s_delta](Vec a, Vec b) { return a * Vec(m_delta) + b * Vec(s_delta); },
          //  buf,
          //  buf,
          //  vi,
          //  K
          //);
          s_prime = s_prime * m_delta + s_delta;

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
    //const at::Tensor& mask
    ) {
  TORCH_CHECK(query.dim() == key.dim());
  //TORCH_CHECK(query.dim() == mask.dim());
  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));
  //TORCH_CHECK(query.size(1) == mask.size(1));
  //TORCH_CHECK(query.size(2) == mask.size(2));
  //TORCH_CHECK(query.size(0) == mask.size(0));

  /*
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
  TORCH_CHECK(!mask.is_cuda(), "mask must be a CPU tensor");

  TORCH_CHECK(!a.is_sparse(), "a must be a dense tensor");
  TORCH_CHECK(!b.is_sparse(), "b must be a dense tensor");
  //TORCH_CHECK(mask.is_sparse(), "mask must be a sparse tensor");
  */
  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);


  at::Tensor res = at::empty({B, M, K}, query.options());

  int64_t grain_size = 32; // TODO: tune this
  //at::Tensor buffer = at::empty({B, grain_size, K}, query.options());
  at::Tensor buffer = at::empty({at::get_num_threads(), 1, K}, query.options());

  AT_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "attention_kernel", [&] {
        attention_kernel<scalar_t>(
            res.accessor<scalar_t, 3>(),
            query.accessor<scalar_t, 3>(),
            key.accessor<scalar_t, 3>(),
            value.accessor<scalar_t, 3>(),
            buffer.accessor<scalar_t, 3>()
            //idxs.accessor<int64_t, 2>()
            );
      });

  return res;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
}
