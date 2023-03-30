#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/array.h>
#include <cutlass/functional.h>

namespace {
template <typename scalar_t>
struct Params {
  // `output = source[index]`
  scalar_t* output; // [B_o, D]
  scalar_t* source; // [B_src, D] with B_o >= B_src
  int64_t* index; // [B_src]
  int output_stride0;
  int source_stride0;
  int d;
  int b_o;
};

template <typename scalar_t>
__global__ void index_select_cu(Params<scalar_t> p) {
  constexpr int kNumElementsPerAccess =
      128 / cutlass::sizeof_bits<scalar_t>::value;
  using AccessType = cutlass::Array<scalar_t, kNumElementsPerAccess>;
  constexpr int kWarpSize = 32;

  int b = blockIdx.x % p.b_o;
  int shift = blockIdx.x / p.b_o;
  int blocksPerLine = gridDim.x / p.b_o;
  int lane = threadIdx.x + shift * p.d / kNumElementsPerAccess;

  AccessType* output =
      reinterpret_cast<AccessType*>(p.output + b * p.output_stride0) + lane;
  AccessType const* source =
      reinterpret_cast<AccessType*>(p.source + p.index[b] * p.source_stride0) +
      lane;
  cutlass::plus<AccessType> add;

  int num_iters = p.d / (kWarpSize * kNumElementsPerAccess);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < num_iters; ++i) {
    AccessType out = source[i * kWarpSize];
    output[i * kWarpSize] = out;
  }
}

at::Tensor index_select(
    at::Tensor output,
    at::Tensor source,
    at::Tensor index) {
  using scalar_t = cutlass::half_t;
  constexpr int kNumElementsPerBlock =
      128 / cutlass::sizeof_bits<scalar_t>::value * 32;

  // dim
  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(source.dim() == 2);
  TORCH_CHECK(index.dim() == 1);

  // shapes
  TORCH_CHECK(output.size(1) == source.size(1));
  TORCH_CHECK(output.size(0) == index.size(0));

  // strides & alignment
  TORCH_CHECK(source.stride(1) == 1);
  TORCH_CHECK(output.stride(1) == 1);
  TORCH_CHECK(source.size(1) % kNumElementsPerBlock == 0);

  // TODO: Dispatch over those
  // dtypes
  TORCH_CHECK(output.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(source.scalar_type() == at::ScalarType::Half);
  TORCH_CHECK(index.scalar_type() == at::ScalarType::Long);

  int grid = output.size(0);
  int threads = 32;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  Params<scalar_t> p;
  p.output = (scalar_t*)output.data_ptr();
  p.source = (scalar_t*)source.data_ptr();
  p.index = (int64_t*)index.data_ptr();
  p.source_stride0 = source.stride(0);
  p.output_stride0 = output.stride(0);
  p.d = source.size(1);
  p.b_o = output.size(0);

  // Add more blocks to fill GPU
  if (p.d % (2 * kNumElementsPerBlock) == 0) {
    int multiplicity = p.d / (2 * kNumElementsPerBlock);
    p.d /= multiplicity;
    grid *= multiplicity;
  }

  index_select_cu<scalar_t><<<grid, threads, 0, stream>>>(p);
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::index_select"), TORCH_FN(index_select));
}
