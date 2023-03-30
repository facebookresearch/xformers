#include <ATen/ATen.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/array.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_conversion.h>
#include "../attention/cuda/fmha/gemm_kernel_utils.h"

// #################################################################################
// #################################### FW PASS
// ####################################
// #################################################################################
namespace {
template <typename scalar_t>
struct ParamsFW {
  // `output[index] = input[index] + alpha * source_scaling * source`
  scalar_t* output; // [B_o, M, D]
  scalar_t* input; // [B_o, M, D]
  scalar_t* source; // [B_src, M, D] with B_o >= B_src
  scalar_t* source_scaling; // [D]
  int64_t* index; // [B_src]
  int64_t stride0;
  int64_t stride1;
  float alpha = 1.0f;
  int d;
  int b_src;
};

template <typename scalar_t, bool kHasScaling, bool kHasInput>
__global__ void scaled_index_addF_cu(ParamsFW<scalar_t> p) {
  constexpr int kNumElementsPerAccess =
      128 / cutlass::sizeof_bits<scalar_t>::value;
  using ElementCompute = float;
  using AccessType = cutlass::Array<scalar_t, kNumElementsPerAccess>;
  using ComputeType = cutlass::Array<ElementCompute, kNumElementsPerAccess>;
  auto storage_to_compute = cutlass::
      NumericArrayConverter<ElementCompute, scalar_t, kNumElementsPerAccess>();
  auto compute_to_storage = cutlass::
      NumericArrayConverter<scalar_t, ElementCompute, kNumElementsPerAccess>();
  constexpr int kWarpSize = 32;

  int b = blockIdx.x % p.b_src;
  int pos_dim1 = blockIdx.x / p.b_src;
  int index_b = p.index[b];
  int lane = threadIdx.x;

  ComputeType alpha;
  alpha.fill(ElementCompute(p.alpha));
  AccessType* output =
      reinterpret_cast<AccessType*>(
          p.output + index_b * p.stride0 + pos_dim1 * p.stride1) +
      lane;
  AccessType const* input =
      reinterpret_cast<AccessType const*>(
          p.input + index_b * p.stride0 + pos_dim1 * p.stride1) +
      lane;
  AccessType const* source =
      reinterpret_cast<AccessType*>(
          p.source + b * p.stride0 + pos_dim1 * p.stride1) +
      lane;
  AccessType const* source_scaling =
      reinterpret_cast<AccessType*>(p.source_scaling) + lane;
  cutlass::plus<ComputeType> add;
  cutlass::multiplies<ComputeType> mul;

  int num_iters = p.d / (kWarpSize * kNumElementsPerAccess);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < num_iters; ++i) {
    ComputeType out = storage_to_compute(source[i * kWarpSize]);
    out = mul(out, alpha);
    if (kHasScaling) {
      ComputeType src_scaling =
          storage_to_compute(source_scaling[i * kWarpSize]);
      out = mul(out, src_scaling);
    }
    if (kHasInput) {
      ComputeType fragment_input = storage_to_compute(input[i * kWarpSize]);
      out = add(out, fragment_input);
    }
    output[i * kWarpSize] = compute_to_storage(out);
  }
}

at::Tensor scaled_index_addF(
    at::Tensor output,
    const c10::optional<at::Tensor>& input_,
    at::Tensor source,
    at::Tensor index,
    const c10::optional<at::Tensor>& source_scaling,
    double alpha) {
  torch::Tensor input = input_.has_value() ? *input_ : output;
  // dim
  TORCH_CHECK(output.dim() == 3);
  TORCH_CHECK(input.dim() == 3);
  TORCH_CHECK(source.dim() == 3);
  TORCH_CHECK(index.dim() == 1);
  TORCH_CHECK(!source_scaling.has_value() || source_scaling->dim() == 1);

  // shapes
  TORCH_CHECK(output.size(0) == input.size(0));
  TORCH_CHECK(output.size(1) == input.size(1));
  TORCH_CHECK(output.size(1) == source.size(1));
  TORCH_CHECK(output.size(2) == input.size(2));
  TORCH_CHECK(output.size(2) == source.size(2));
  TORCH_CHECK(source.size(0) == index.size(0));
  TORCH_CHECK(
      !source_scaling.has_value() || source_scaling->size(0) == output.size(2));

  // strides & alignment
  TORCH_CHECK(source.stride(-1) == 1);
  TORCH_CHECK(input.stride(-1) == 1);
  TORCH_CHECK(output.stride(-1) == 1);
  TORCH_CHECK(output.stride(0) == input.stride(0));
  TORCH_CHECK(output.stride(0) == source.stride(0));
  TORCH_CHECK(output.stride(1) == input.stride(1));
  TORCH_CHECK(output.stride(1) == source.stride(1));

  // dtypes
  TORCH_CHECK(output.scalar_type() == source.scalar_type());
  TORCH_CHECK(output.scalar_type() == input.scalar_type());
  TORCH_CHECK(output.scalar_type() == input.scalar_type());
  TORCH_CHECK(
      !source_scaling.has_value() ||
      source_scaling->scalar_type() == output.scalar_type());
  TORCH_CHECK(index.scalar_type() == at::ScalarType::Long);

  // TODO: Only half supported at the moment
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half);

  int grid = source.size(0) * source.size(1);
  int threads = 32;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using scalar_t = cutlass::half_t;
  constexpr int kNumElementsPerWarp =
      (128 / cutlass::sizeof_bits<scalar_t>::value) * 32;
  TORCH_CHECK(source.size(-1) % kNumElementsPerWarp == 0);

  ParamsFW<scalar_t> p;
  p.output = (scalar_t*)output.data_ptr();
  if (input_.has_value()) {
    p.input = (scalar_t*)input.data_ptr();
  }
  p.source = (scalar_t*)source.data_ptr();
  p.index = (int64_t*)index.data_ptr();
  p.stride0 = input.stride(0);
  p.stride1 = input.stride(1);
  p.b_src = source.size(0);
  p.d = source.size(2);
  p.alpha = float(alpha);

  if (source_scaling.has_value()) {
    p.source_scaling = (scalar_t*)source_scaling->data_ptr();
  } else if (source.size(1) == 1 && p.d % (kNumElementsPerWarp) == 0) {
    // if we don't have a scaling, we can "reshape" the inputs to spawn more
    // threads let's restrict this heuristic for cases when `shape[1] == 1` for
    // now
    int multiplicity = p.d / (kNumElementsPerWarp);
    p.d = p.d / multiplicity; // `kNumElementsPerWarp`
    p.stride1 = p.d;
    grid *= multiplicity;
  }

  // launch kernel
  DISPATCH_BOOL(input_.has_value(), kHasInput, ([&]() {
                  DISPATCH_BOOL(
                      source_scaling.has_value(), kHasScaling, ([&]() {
                        scaled_index_addF_cu<scalar_t, kHasScaling, kHasInput>
                            <<<grid, threads, 0, stream>>>(p);
                      }));
                }));
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

} // namespace

// #################################################################################
// #################################### BW PASS
// ####################################
// #################################################################################
namespace {
template <typename scalar_t>
struct ParamsBW : ParamsFW<scalar_t> {
  scalar_t* source; // [B_src, D] with B_o >= B_src
  scalar_t* source_scaling; // [D]
  int64_t* index; // [B_src]
  int d;
  int b_src;
  int stride0;
  int stride1;

  // grads:
  scalar_t* grad_output; // <- incoming grad
  scalar_t* grad_source; // [B_src, D]
  scalar_t* grad_source_scaling; // [D]
  int gout_stride0;
  int gout_stride1;
  float alpha = 1.0f;
};

template <typename scalar_t, bool kHasSourceScaling>
__global__ void scaled_index_addB_cu(ParamsBW<scalar_t> p) {
  constexpr int kNumElementsPerAccess =
      128 / cutlass::sizeof_bits<scalar_t>::value;
  using ElementCompute = float;
  using AccessType = cutlass::Array<scalar_t, kNumElementsPerAccess>;
  using ComputeType = cutlass::Array<ElementCompute, kNumElementsPerAccess>;
  auto storage_to_compute = cutlass::
      NumericArrayConverter<ElementCompute, scalar_t, kNumElementsPerAccess>();
  auto compute_to_storage = cutlass::
      NumericArrayConverter<scalar_t, ElementCompute, kNumElementsPerAccess>();
  constexpr int kWarpSize = 32;

  int b = blockIdx.x % p.b_src;
  int pos_dim1 = blockIdx.x / p.b_src;
  int lane = threadIdx.x;

  ComputeType alpha;
  alpha.fill(ElementCompute(p.alpha));
  AccessType const* grad_output =
      reinterpret_cast<AccessType const*>(
          p.grad_output + p.index[b] * p.gout_stride0 +
          pos_dim1 * p.gout_stride1) +
      lane;
  AccessType const* source =
      reinterpret_cast<AccessType const*>(
          p.source + b * p.stride0 + pos_dim1 * p.stride1) +
      lane;
  AccessType const* source_scaling =
      reinterpret_cast<AccessType const*>(p.source_scaling) + lane;
  AccessType* grad_source =
      reinterpret_cast<AccessType*>(
          p.grad_source + b * p.stride0 + pos_dim1 * p.stride1) +
      lane;
  AccessType* grad_source_scaling =
      reinterpret_cast<AccessType*>(
          p.grad_source_scaling + b * p.stride0 + pos_dim1 * p.stride1) +
      lane;
  cutlass::plus<ComputeType> add;
  cutlass::multiplies<ComputeType> mul;

  int num_iters = p.d / (kWarpSize * kNumElementsPerAccess);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < num_iters; ++i) {
    ComputeType gout = storage_to_compute(grad_output[i * kWarpSize]);
    ComputeType src = storage_to_compute(source[i * kWarpSize]);
    gout = mul(gout, alpha);
    ComputeType gsrc = gout;
    if (kHasSourceScaling) {
      ComputeType src_scaling =
          storage_to_compute(source_scaling[i * kWarpSize]);
      gsrc = mul(gout, src_scaling);
    }
    ComputeType gscaling = mul(gout, src);

    grad_source[i * kWarpSize] = compute_to_storage(gsrc);
    if (kHasSourceScaling) {
      grad_source_scaling[i * kWarpSize] = compute_to_storage(gscaling);
    }
  }
}

std::tuple<at::Tensor, const c10::optional<at::Tensor>> scaled_index_addB(
    // outputs:
    at::Tensor grad_source,
    const c10::optional<at::Tensor>& grad_source_scaling,
    // inputs:
    at::Tensor grad_output,
    at::Tensor source,
    at::Tensor index,
    const c10::optional<at::Tensor>& source_scaling,
    double alpha) {
  TORCH_CHECK(source_scaling.has_value() == grad_source_scaling.has_value());
  if (grad_source_scaling.has_value()) {
    TORCH_CHECK(grad_source_scaling->stride(-1) == 1);
    TORCH_CHECK(grad_source_scaling->dim() == 3);
    TORCH_CHECK(grad_source_scaling->size(0) == source.size(0));
    TORCH_CHECK(grad_source_scaling->size(1) == source.size(1));
    TORCH_CHECK(grad_source_scaling->size(2) == source.size(2));
    TORCH_CHECK(source.stride(0) == grad_source_scaling->stride(0));
    TORCH_CHECK(source.stride(1) == grad_source_scaling->stride(1));
    TORCH_CHECK(source.stride(2) == grad_source_scaling->stride(2));
    TORCH_CHECK(grad_source_scaling->scalar_type() == at::ScalarType::Half);
  }
  // TODO: Missing some checks there
  TORCH_CHECK(grad_source.dim() == 3);
  TORCH_CHECK(grad_output.dim() == 3);
  TORCH_CHECK(source.dim() == 3);
  TORCH_CHECK(source.stride(0) == grad_source.stride(0));
  TORCH_CHECK(source.stride(1) == grad_source.stride(1));
  TORCH_CHECK(source.stride(2) == grad_source.stride(2));

  TORCH_CHECK(index.scalar_type() == at::ScalarType::Long);
  // TODO: Only half supported at the moment
  TORCH_CHECK(source.scalar_type() == at::ScalarType::Half);

  int grid = source.size(0) * source.size(1);
  int threads = 32;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  using scalar_t = cutlass::half_t;
  ParamsBW<scalar_t> p;
  p.source = (scalar_t*)source.data_ptr();
  if (source_scaling.has_value()) {
    p.source_scaling = (scalar_t*)source_scaling->data_ptr();
  }
  p.index = (int64_t*)index.data_ptr();
  p.stride0 = source.stride(0);
  p.stride1 = source.stride(1);

  p.grad_output = (scalar_t*)grad_output.data_ptr();
  p.gout_stride0 = grad_output.stride(0);
  p.gout_stride1 = grad_output.stride(1);

  p.grad_source = (scalar_t*)grad_source.data_ptr();
  if (grad_source_scaling.has_value()) {
    p.grad_source_scaling = (scalar_t*)grad_source_scaling->data_ptr();
  }
  p.b_src = source.size(0);
  p.d = source.size(2);
  p.alpha = float(alpha);

  DISPATCH_BOOL(grad_source_scaling.has_value(), kHasScaling, ([&]() {
                  scaled_index_addB_cu<scalar_t, kHasScaling>
                      <<<grid, threads, 0, stream>>>(p);
                }));
  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(grad_source, grad_source_scaling);
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::scaled_index_addF"),
      TORCH_FN(scaled_index_addF));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::scaled_index_addB"),
      TORCH_FN(scaled_index_addB));
}
