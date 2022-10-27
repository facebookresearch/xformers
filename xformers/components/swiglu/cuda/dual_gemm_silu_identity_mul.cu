#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ScalarOps.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <43_dual_gemm/device/dual_gemm.h>
#include <43_dual_gemm/thread/left_silu_and_mul.h>

namespace {
template <typename scalar_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_silu_identity_mul_(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1
) {
  TORCH_CHECK(x.stride(-1) == 1);
  TORCH_CHECK(w0.stride(-1) == 1);
  TORCH_CHECK(w1.stride(-1) == 1);

  at::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = x.size(0);
  int64_t I = x.size(1);
  int64_t H = w0.size(0);

  at::Tensor d0 = at::empty({B, H}, x.options());
  at::Tensor d1 = at::empty({B, H}, x.options());
  at::Tensor d2 = at::empty({B, H}, x.options());

  // templati-ze the cutlass kernel
  cutlass::gemm::GemmCoord problem_size(B, H, I);

  constexpr int kStages = 3;
  constexpr bool kSplitKSerial = false;

  using ElementOutput = scalar_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using EpilogueOutputOp01 = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling
  >;
  using EpilogueOutputOp2 = cutlass::epilogue::thread::LeftSiLUAndMul<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementOutput,
    ElementCompute
  >;

  const ElementCompute alpha0 = ElementCompute(1);
  const ElementCompute beta0 = ElementCompute(1);
  const ElementCompute alpha1 = ElementCompute(1);
  const ElementCompute beta1 = ElementCompute(1);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;

  using DualGemm = cutlass::gemm::device::DualGemm<
    scalar_t,
    cutlass::layout::RowMajor,
    scalar_t,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp01,
    EpilogueOutputOp01,
    EpilogueOutputOp2,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    kStages,
    kStoreD0,
    kStoreD1,
    kSplitKSerial
  >;

  int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
  using RefA = typename cutlass::TensorRef<typename DualGemm::ElementA, typename DualGemm::LayoutA>;
  using RefB = typename cutlass::TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB>;
  using RefC = typename cutlass::TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>;
  typename DualGemm::Arguments arguments{
    problem_size,
    RefA{(scalar_t*)x.data_ptr(), typename DualGemm::LayoutA::Stride(x.stride(0))},
    RefB{(scalar_t*)w0.data_ptr(), typename DualGemm::LayoutB::Stride(w0.stride(0))},
    RefC{(scalar_t*)b0.data_ptr(), typename DualGemm::LayoutC::Stride(0)},
    RefC{(scalar_t*)d0.data_ptr(), typename DualGemm::LayoutC::Stride(d0.stride(0))},
    RefB{(scalar_t*)w1.data_ptr(), typename DualGemm::LayoutB::Stride(w1.stride(0))},
    RefC{(scalar_t*)b1.data_ptr(), typename DualGemm::LayoutC::Stride(0)},
    RefC{(scalar_t*)d1.data_ptr(), typename DualGemm::LayoutC::Stride(d1.stride(0))},
    RefC{(scalar_t*)d2.data_ptr(), typename DualGemm::LayoutC::Stride(d2.stride(0))},
    typename DualGemm::EpilogueOutputOp0::Params{alpha0, beta0},
    typename DualGemm::EpilogueOutputOp1::Params{alpha1, beta1},
    typename DualGemm::EpilogueOutputOp2::Params{},
    split_k_slices
  };
  DualGemm dual_gemm;
  at::Tensor workspace = at::empty({int64_t(dual_gemm.get_workspace_size(arguments))}, x.options().dtype(at::ScalarType::Byte));
  cutlass::Status status = dual_gemm.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "not supported by this kernel");
  status = dual_gemm.initialize(arguments, (uint8_t*)workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel initialize failed");
  status = dual_gemm(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel run failed");
  return std::make_tuple(d0, d1, d2);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_silu_identity_mul(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1
) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(x.dim() == 2);
  TORCH_CHECK(w0.dim() == 2);
  TORCH_CHECK(w1.dim() == 2);

  #define FWD_PARAMS x,w0,b0,w1,b1

  if (x.scalar_type() == at::ScalarType::Half) {
    return dual_gemm_silu_identity_mul_<cutlass::half_t>(FWD_PARAMS);
  } else {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "Only supports bf16/f16");
    return dual_gemm_silu_identity_mul_<cutlass::bfloat16_t>(FWD_PARAMS);
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_silu_identity_mul_autocast(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1
) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_gpu_dtype();
  return dual_gemm_silu_identity_mul(
    at::autocast::cached_cast(exec_type, x),
    at::autocast::cached_cast(exec_type, w0),
    at::autocast::cached_cast(exec_type, b0),
    at::autocast::cached_cast(exec_type, w1),
    at::autocast::cached_cast(exec_type, b1)
  );
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::dual_gemm_silu_identity_mul"),
      TORCH_FN(dual_gemm_silu_identity_mul));
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::dual_gemm_silu_identity_mul"),
      TORCH_FN(dual_gemm_silu_identity_mul_autocast));
}
