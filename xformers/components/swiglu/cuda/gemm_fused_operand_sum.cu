#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ScalarOps.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_with_k_reduction.h"
#include "cutlass/reduction/device/reduce_split_k.h"
#include "cutlass/reduction/kernel/reduce_split_k.h"
#include "cutlass/reduction/thread/reduction_operators.h"
#include "cutlass/matrix_coord.h"

#define WORKAROUND_CUTLASS_BUG

namespace {
template <typename scalar_t>
void gemm_fused_operand_sum_(
    const at::Tensor& a, // col-major
    const at::Tensor& b, // row-major
    at::Tensor& out_mm, // row-major
    at::Tensor& out_sum // row-major
) {
  at::cuda::CUDAGuard device_guard(a.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t M = a.size(0);
  int64_t N = b.size(0);
  int64_t K = a.size(1);

  // templati-ze the cutlass kernel
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  using ElementAccumulator = float;                  // Data type of accumulator
  using ElementComputeEpilogue = ElementAccumulator; // Data type of epilogue computation
  using ElementInputA = scalar_t;
  using ElementInputB = scalar_t;
  using ElementOutput = scalar_t;

  using LayoutInputA = cutlass::layout::ColumnMajor;
  TORCH_CHECK(a.stride(0) == 1);
  using LayoutInputB = cutlass::layout::RowMajor;
  TORCH_CHECK(b.stride(1) == 1);
  using LayoutOutput = cutlass::layout::RowMajor;
  TORCH_CHECK(out_mm.stride(1) == 1);

  // Layout of the output vector
  using LayoutGemmKReduction = cutlass::layout::PitchLinear;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;  // Threadblock tile shape

  // This code section describes tile size a warp will compute
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;         // Warp tile shape

  // This code section describes the size of MMA op
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;    // TensorCore instruction shape

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  // Number of pipelines you want to use
  constexpr int NumStages = 4;

  // Reduce A or B operand along the K dimension
#ifdef WORKAROUND_CUTLASS_BUG
  constexpr bool ReduceKForA = false;
#else
  constexpr bool ReduceKForA = true;
#endif

  // This code section describes the epilogue part of the kernel, we use default value
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                        // Data type of output matrix.
      128 / cutlass::sizeof_bits<ElementOutput>::value,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
      ElementAccumulator,                                   // Data type of accumulator
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithKReduction<
  ElementInputA, LayoutInputA, cutlass::ComplexTransform::kNone, 8,
  ElementInputB, LayoutInputB, cutlass::ComplexTransform::kNone, 8,
  ElementOutput, LayoutOutput,
  ElementAccumulator,
  MMAOp,
  ReduceKForA,
  SmArch,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOp,
  SwizzleThreadBlock,
  NumStages,
  cutlass::arch::OpMultiplyAdd
  >::GemmKernel;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Below is the reduction kernel used in the case of parallel split-k
  using ReduceGemmSplitKShape = cutlass::MatrixShape<4, 64>;;

  using ReduceOp = cutlass::reduction::thread::ReduceAdd<
      ElementAccumulator,
      ElementOutput,
      EpilogueOp::kCount
  >;

  using ReduceGemmSplitKKernel = cutlass::reduction::kernel::ReduceSplitK<
      ReduceGemmSplitKShape,
      EpilogueOp,
      ReduceOp
  >;

  using ReduceGemmSplitK = cutlass::reduction::device::ReduceSplitK<ReduceGemmSplitKKernel>;

  using ReduceVectorSplitKShape = cutlass::MatrixShape<1, 256>;;

  // This code section describes the epilogue part of the kernel, we use default value
  using DummyEpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                        // Data type of output matrix.
      128 / cutlass::sizeof_bits<ElementOutput>::value,     // The number of elements per vectorized.
                                                          // memory access. This becomes the vector width of
                                                          // math instructions in the epilogue too.
      ElementAccumulator,                                   // Data type of accumulator
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

  using ReduceVectorSplitKKernel = cutlass::reduction::kernel::ReduceSplitK<
      ReduceVectorSplitKShape,
      DummyEpilogueOp,
      ReduceOp
  >;

  using ReduceVectorSplitK = cutlass::reduction::device::ReduceSplitK<ReduceVectorSplitKKernel>;
  auto alpha = ElementComputeEpilogue(1);
  auto beta = ElementComputeEpilogue(0);

  using RefA = cutlass::TensorRef<ElementInputA, LayoutInputA>;
  using RefB = cutlass::TensorRef<ElementInputB, LayoutInputB>;
  using RefC = cutlass::TensorRef<ElementOutput, LayoutOutput>;
  int reduce_vector_length = ReduceKForA ? problem_size.m() : problem_size.n();
  int split_k_slices = 2;
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    problem_size,
    int(split_k_slices),
    {alpha, beta},
    (ElementInputA const*)a.data_ptr(),
    (ElementInputB const*)b.data_ptr(),
    (ElementOutput*)nullptr,
    (ElementOutput*)out_mm.data_ptr(),
    (ElementOutput*)out_sum.data_ptr(),
    problem_size.m() * problem_size.k(),
    problem_size.n() * problem_size.k(),
    problem_size.m() * problem_size.n(),
    problem_size.m() * problem_size.n(),
    reduce_vector_length,
    a.stride(1),
    b.stride(0),
    int64_t(0), // bias
    out_mm.stride(0),
    int64_t(1) // out_sum
  };

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace = at::empty({int64_t(gemm_op.get_workspace_size(arguments))}, a.options().dtype(at::ScalarType::Byte));
  cutlass::Status status = gemm_op.can_implement(arguments);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "not supported by this kernel");
  status = gemm_op.initialize(arguments, (uint8_t*)workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel initialize failed");
  status = gemm_op(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel run failed");
}

std::tuple<at::Tensor, at::Tensor> gemm_fused_operand_sum(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out_mm,
    at::Tensor& out_sum
) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(a.dim() == 2);
  TORCH_CHECK(b.dim() == 2);
  TORCH_CHECK(out_mm.dim() == 2);

  #define FWD_PARAMS a,b,out_mm,out_sum

  if (a.scalar_type() == at::ScalarType::Half) {
    gemm_fused_operand_sum_<cutlass::half_t>(FWD_PARAMS);
  } else {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::BFloat16, "Only supports bf16/f16");
    gemm_fused_operand_sum_<cutlass::bfloat16_t>(FWD_PARAMS);
  }
  return std::make_tuple(out_mm, out_sum);
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::gemm_fused_operand_sum"),
      TORCH_FN(gemm_fused_operand_sum));
}
