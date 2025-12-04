// BEGIN COPY-PASTE FROM PyTorch
// https://github.com/pytorch/pytorch/blob/b937510a3f254fe0223b9b29235e0eb6e6da912a/aten/src/ATen/native/sparse/cuda/StructuredSparseLinearCUTLASS.cu
// Some very small modifications, like we don't need to support uint8, and we
// always have the meta-reordered available
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>

#include <tuple>
#include <type_traits>

#include "pt_stable_utils.h"

namespace {
#define CUTLASS_STATUS_CHECK(status)         \
  {                                          \
    STD_TORCH_CHECK(                         \
        status == cutlass::Status::kSuccess, \
        "Got CUTLASS error: ",               \
        cutlassGetStatusString(status));     \
  }

// Wrapper function for CUTLASS sparse GEMM implementation, used
// solely to simplify dispatching from _structured_sparse_linear()
// function below.
template <
    bool kIsMeta,
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ElementComputeEpilogue,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOp,
    typename LayoutInputA,
    typename LayoutInputB>
torch::stable::Tensor two_four_sgemm_cutlass(
    const torch::stable::Tensor& tensor_a,
    const torch::headeronly::IntHeaderOnlyArrayRef::value_type& tensor_a_stride,
    const torch::stable::Tensor& tensor_b,
    const torch::headeronly::IntHeaderOnlyArrayRef::value_type& tensor_b_stride,
    const torch::stable::Tensor& meta_reordered) {
  // Fix CUTLASS sparse GEMM template arguments that are not
  // provided as template argument of this function, and create an
  // alias for particular instantiation of this template.
  using LayoutOutput =
      cutlass::layout::RowMajor; // Result of the operation will be provided in
                                 // row-major format.
  using MMAOp = cutlass::arch::OpClassTensorOp; // Tensor cores are to be used
                                                // for maximum performance.
  using SmArch =
      cutlass::arch::Sm80; // Only CC 8.x devices are suported at the moment.
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<
          3>; // This choice provides good performance
              // across wide range of operand sizes.
  constexpr int NumStages = 4; // This choice provides good performance across
                               // wide range of operand sizes.
  using Gemm = cutlass::gemm::device::SparseGemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages>;

  // Datatype and layout of metadata matrix are inferred from sparse
  // GEMM template.
  using ElementInputE = typename Gemm::ElementE;
  using ReorderedLayoutInputE = typename Gemm::LayoutE;

  constexpr auto kSparse = Gemm::kSparse;
  constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;

  // Operand sizes.
  const int length_m = tensor_a.size(0);
  const int length_k = tensor_b.size(0);
  const int length_n = tensor_b.size(1);
  const auto meta_ncols = length_k / kSparse / kElementsPerElementE;

  // Check for current CUTLASS limitations w.r.t. input sizes.
  constexpr auto input_a_is_half =
      std::is_same<ElementInputA, cutlass::half_t>::value ||
      std::is_same<ElementInputA, cutlass::bfloat16_t>::value;
  STD_TORCH_CHECK(
      length_m % 32 == 0,
      "torch._structured_sparse_linear: Number of rows of sparse matrix must "
      "be divisible by 32");
  STD_TORCH_CHECK(
      length_k % (input_a_is_half ? 64 : 128) == 0,
      "torch._structured_sparse_linear: Number of rows of dense matrix must "
      "be divisible by ",
      (input_a_is_half ? 64 : 128));
  STD_TORCH_CHECK(
      length_n % (input_a_is_half ? 8 : 16) == 0,
      "torch._structured_sparse_linear: Number of columns of dense matrix "
      "must be divisible by ",
      (input_a_is_half ? 8 : 16));

  // Determine PyTorch datatype for the output matrix.
  auto tensor_d_dtype = torch::headeronly::ScalarType::Char;
  if (std::is_same<ElementOutput, int32_t>::value) {
    tensor_d_dtype = torch::headeronly::ScalarType::Int;
  } else if (std::is_same<ElementOutput, cutlass::half_t>::value) {
    tensor_d_dtype = torch::headeronly::ScalarType::Half;
  } else if (std::is_same<ElementOutput, cutlass::bfloat16_t>::value) {
    tensor_d_dtype = torch::headeronly::ScalarType::BFloat16;
  } else {
    STD_TORCH_CHECK(
        false,
        "torch._structured_sparse_linear: invalid sparse GEMM output "
        "datatype encountered");
  }

  // Create output matrix.
  auto tensor_d = torch::stable::new_empty(
      tensor_a,
      {length_m, length_n},
      /*dtype=*/tensor_d_dtype);
  if (kIsMeta) {
    return tensor_d;
  }

  // Prepare arguments for CUTLASS sparse GEMM kernel.
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  LayoutInputA layout_a(tensor_a_stride);
  LayoutInputB layout_b(tensor_b_stride);
  LayoutOutput layout_d(tensor_d.stride(0));
  auto tensor_a_device_ref = cutlass::TensorRef<ElementInputA, LayoutInputA>(
      (ElementInputA*)tensor_a.data_ptr(), layout_a);
  auto tensor_b_device_ref = cutlass::TensorRef<ElementInputB, LayoutInputB>(
      (ElementInputB*)tensor_b.data_ptr(), layout_b);
  auto tensor_d_device_ref = cutlass::TensorRef<ElementOutput, LayoutOutput>(
      (ElementOutput*)tensor_d.data_ptr(), layout_d);
  auto tensor_e_reordered_device_ref =
      cutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>(
          (ElementInputE*)meta_reordered.data_ptr(),
          ReorderedLayoutInputE::packed({length_m, meta_ncols}));
  ElementComputeEpilogue alpha(1);
  ElementComputeEpilogue beta(0);
  constexpr int split_k_slices = 1;

  // Create a tuple of CUTLASS sparse GEMM kernel arguments.
  typename Gemm::Arguments arguments{
      problem_size,
      tensor_a_device_ref,
      tensor_b_device_ref,
      tensor_d_device_ref,
      tensor_d_device_ref,
      tensor_e_reordered_device_ref,
      {alpha, beta},
      split_k_slices};

  cutlass::Status status;

  // Create CUTLASS sparse GEMM kernel object.
  Gemm gemm_op;

  // Verify that sparse GEMM operation with given arguments can be
  // performed by CUTLASS.
  status = gemm_op.can_implement(arguments);
  CUTLASS_STATUS_CHECK(status);

  // Allocate workspace for CUTLASS sparse GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = torch::stable::new_empty(
      tensor_a, {(int64_t)workspace_size}, torch::headeronly::ScalarType::Byte);

  // Initialize CUTLASS sparse GEMM object.
  status = gemm_op.initialize(
      arguments, workspace.data_ptr(), xf_getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  // Perform sparse GEMM operation.
  status = gemm_op.run(xf_getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);

  XF_CUDA_KERNEL_LAUNCH_CHECK();

  return tensor_d;
}

template <bool kIsMeta>
torch::stable::Tensor _sparse24_gemm(
    const torch::stable::Tensor& tensor_a,
    const torch::stable::Tensor& tensor_b,
    const torch::stable::Tensor& mask_or_meta) {
  // No need to check that all tensors are on CUDA device, as this
  // is provided by dispatch.

  // For now, only CC 8.x devices are supported.
  if (!kIsMeta) {
    const auto dprops = xf_getCurrentDeviceProperties();
    const auto is_sm8x = dprops->major == 8;
    STD_TORCH_CHECK(
        is_sm8x,
        "torch._structured_sparse_linear: Supported only on GPUs with "
        "compute capability 8.x");
  }

  // Validate layouts of input tensors.
  STD_TORCH_CHECK(
      !xf_is_sparse(tensor_a),
      "torch._structured_sparse_linear: Expected tensor_a argument "
      "to be strided, but got layout ",
      xf_get_layout(tensor_a));
  STD_TORCH_CHECK(
      tensor_a.dim() == 2,
      "torch._structured_sparse_linear: Expected tensor_a argument "
      "to be 2D tensor, got ",
      tensor_a.dim(),
      " dims");
  const auto strides_a = tensor_a.strides();
  STD_TORCH_CHECK(
      (strides_a[0] == 1 || strides_a[1] == 1) && strides_a[0] != strides_a[1],
      "torch._structured_sparse_linear: Invalid strides for tensor_a "
      "argument: row stride = ",
      strides_a[0],
      ", column stride = ",
      strides_a[1]);
  STD_TORCH_CHECK(
      !xf_is_sparse(tensor_b),
      "torch._structured_sparse_linear: Expected tensor_b argument "
      "to be strided, but got layout ",
      xf_get_layout(tensor_b));
  STD_TORCH_CHECK(
      tensor_b.dim() == 2,
      "torch._structured_sparse_linear: Expected tensor_b argument "
      "to be 2D tensor, got ",
      tensor_b.dim(),
      " dims");
  const auto strides_b = tensor_b.strides();
  STD_TORCH_CHECK(
      (strides_b[0] == 1 || strides_b[1] == 1) && strides_b[0] != strides_b[1],
      "torch._structured_sparse_linear: Invalid strides for tensor_b "
      "argument: row stride = ",
      strides_b[0],
      ", column stride = ",
      strides_b[1]);

  // Determine layout (row-major or column-major) of input tensors.
  auto tensor_a_row_major = strides_a[1] == 1;
  auto tensor_a_stride = tensor_a_row_major ? strides_a[0] : strides_a[1];
  auto tensor_b_row_major = strides_b[1] == 1;
  auto tensor_b_stride = tensor_b_row_major ? strides_b[0] : strides_b[1];

  // Call wrapper function for CUTLASS sparse GEMM, dispatching on
  // the input datatype, and then on input tensors layouts.
  // According to the input tensors datatypes and layouts,
  // correspnding template arguments are supplied for instantiating
  // the wrapper function.  The tile sizes template arguments are
  // selected according to the CUTLASS profiler results, for number
  // of runs.
  torch::stable::Tensor result;
  auto runGemm = [&](auto dtype) {
    using ElementInputA = decltype(dtype);
    using ElementInputB = decltype(dtype);
    using ElementOutput = decltype(dtype);

    using ElementAccumulator = float;
    using ElementComputeEpilogue = float;
    using ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;
    if (tensor_a_row_major && tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::RowMajor,
          cutlass::layout::RowMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    } else if (tensor_a_row_major && !tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::RowMajor,
          cutlass::layout::ColumnMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    } else if (!tensor_a_row_major && tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::ColumnMajor,
          cutlass::layout::RowMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    } else if (!tensor_a_row_major && !tensor_b_row_major) {
      result = two_four_sgemm_cutlass<
          kIsMeta,
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ElementComputeEpilogue,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          EpilogueOp,
          cutlass::layout::ColumnMajor,
          cutlass::layout::ColumnMajor>(
          tensor_a, tensor_a_stride, tensor_b, tensor_b_stride, mask_or_meta);
    }
  };
  if (tensor_a.scalar_type() == torch::headeronly::ScalarType::Half) {
    runGemm(cutlass::half_t());
  } else if (
      tensor_a.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    runGemm(cutlass::bfloat16_t());
  } else {
    STD_TORCH_CHECK(false, "Unsupported Sparse24 GEMM")
  }
  return result;
}
// END PyTorch copy-pasted code
} // namespace

STABLE_TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl("_sparse24_gemm", XF_BOXED_FN(_sparse24_gemm<false>));
}

STABLE_TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl("_sparse24_gemm", XF_BOXED_FN(_sparse24_gemm<true>));
}
