#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>
#include <cmath>
#include <vector>


#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


namespace {



#define CUDA_1D_KERNEL_LOOP(i, n)                                \
  for (int i = (blockIdx.x * blockDim.x) + threadIdx.x; i < (n); \
       i += (blockDim.x * gridDim.x))

template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) {
  return (n + m - 1) / m;
}

template <typename scalar_t>
__global__ void attention_kernel(
    at::PackedTensorAccessor<scalar_t, 3> output,
    at::PackedTensorAccessor<scalar_t, 3> query,
    at::PackedTensorAccessor<scalar_t, 3> key,
    at::PackedTensorAccessor<scalar_t, 3> value
    ) {
  constexpr int64_t BLOCK = 16;
  int64_t K = query.size(2);
  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);

  int64_t i = blockIdx.y;
  //int64_t j = blockIdx.x;
  int64_t j = blockIdx.x * blockDim.y + threadIdx.y;

      {{
        //auto aar = query[i][j].data();
        auto aar = reinterpret_cast<float4 *>(query[i][j].data());

        //auto oo = output[i][j].data();
        auto oo = reinterpret_cast<float4 *>(output[i][j].data());
        scalar_t s_prime = 0;
        scalar_t m_prime = -std::numeric_limits<scalar_t>::infinity();
        //for (int64_t l = threadIdx.x * BLOCK; l < N; l+=BLOCK * blockDim.x) {
        for (int64_t l = 0; l < N; l+=BLOCK) {
          //auto bar = key[i][l].data();
          auto bar = reinterpret_cast<float4 *>(key[i][l].data());
          scalar_t si[BLOCK] = {0};
          //for (int64_t k = threadIdx.x; k < K; k+=32) {
          for (int64_t k = 0; k < K / 4; k+=1) {
            //auto aaar = aar[k];
            //auto aaar = __ldg(aar + k);
            float4 aaar = __ldg(aar + k);
            for (int64_t rr = 0; rr < BLOCK; rr++) {
              float4 bbb = bar[k + K / 4 * rr];
              si[rr] += aaar.x * bbb.x + aaar.y * bbb.y + aaar.z * bbb.z + aaar.w * bbb.w;

              //si[rr] += aaar * bar[k + K * rr];
              //si[rr] += aaar * __ldg(bar + k + K * rr);
            }
          }

          //for (int64_t rr = 0; rr < BLOCK; rr++) {
          //  for (int stride = 16; stride > 0; stride >>= 1) {
          //    si[rr] += __shfl_xor_sync(0xffffffff, si[rr], stride, 32);
          //  }
          //}

          scalar_t m_i = si[0] > m_prime ? si[0] : m_prime;
          for (int64_t rr = 1; rr < BLOCK; rr++) {
            m_i = si[rr] > m_i ? si[rr] : m_i;
          }
          //s_prime = m_i;  // TODO: only for testing, remove!!!

          //auto vi = value[i][l].data();
          auto vi = reinterpret_cast<float4 *>(value[i][l].data());

          scalar_t m_delta;
          scalar_t s_delta[BLOCK];
          m_delta = std::exp(m_prime - m_i);

          for (int64_t rr = 0; rr < BLOCK; rr++)
            s_delta[rr] = std::exp(si[rr] - m_i);

          //for (int64_t k = threadIdx.x; k < K; k+=blockDim.x) {
          for (int64_t k = 0; k < K/4; k+=1) {
            //oo[k] = oo[k] * m_delta;
            float4 tmp = oo[k];
            tmp.x = tmp.x * m_delta;
            tmp.y = tmp.y * m_delta;
            tmp.z = tmp.z * m_delta;
            tmp.w = tmp.w * m_delta;
            for (int64_t rr = 0; rr < BLOCK; rr++) {
              //oo[k] += vi[k + K * rr] * s_delta[rr];
              float4 tmp2 = vi[k + K / 4 * rr];
              tmp.x += tmp2.x * s_delta[rr];
              tmp.y += tmp2.y * s_delta[rr];
              tmp.z += tmp2.z * s_delta[rr];
              tmp.w += tmp2.w * s_delta[rr];
              //oo[k] += __ldg(vi + k + K * rr) * s_delta[rr];
            }
            oo[k] = tmp;
          }
          s_prime = s_prime * m_delta;
          for (int64_t rr = 0; rr < BLOCK; rr++)
            s_prime += s_delta[rr];

          m_prime = m_i;
        }

        //for (int64_t k = threadIdx.x; k < K; k+=blockDim.x) {
        for (int64_t k = 0; k < K / 4; k+=1) {
          //oo[k] /= s_prime;
          float4 tmp = oo[k];
          tmp.x /= s_prime;
          tmp.y /= s_prime;
          tmp.z /= s_prime;
          tmp.w /= s_prime;
          oo[k] = tmp;
        }
      }
  }
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

  at::cuda::CUDAGuard device_guard(query.device());

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);


  at::Tensor res = at::zeros({B, M, K}, query.options());

  int64_t grain_size = 32; // TODO: tune this
  //at::Tensor buffer = at::empty({B, grain_size, K}, query.options());
  //at::Tensor buffer = at::empty({at::get_num_threads(), 1, K}, query.options());


  //dim3 grid(std::min(
  //    ceil_div(static_cast<int64_t>(B), static_cast<int64_t>(512)),
  //    static_cast<int64_t>(4096)));
  //dim3 block(512);
  //dim3 grid(M / 32, B);
  dim3 grid(M / 32, B);
  //dim3 block(32, 32);
  //dim3 block(4, 32);
  dim3 block(1, 32);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES(
      query.scalar_type(), "attention_kernel", [&] {
        attention_kernel<scalar_t><<<grid, block, 0, stream>>>(
            res.packed_accessor<scalar_t, 3>(),
            query.packed_accessor<scalar_t, 3>(),
            key.packed_accessor<scalar_t, 3>(),
            value.packed_accessor<scalar_t, 3>()
            //buffer.accessor<scalar_t, 3>()
            //idxs.accessor<int64_t, 2>()
            );
      });

  return res;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::efficient_attention"),
      TORCH_FN(attention));
}
