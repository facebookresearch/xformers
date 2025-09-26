#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cute/algorithm/functional.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
#include "cutlass/arch/wmma.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/device/transform_universal_adapter.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"

#include <tuple>
#include <type_traits>

// CUTLASS does not mix well with Windows
#ifndef _WIN32
#if __CUDACC_VER_MAJOR__ > 12 || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ > 0)

#ifdef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
namespace {
#define CUTLASS_STATUS_CHECK(status)              \
  {                                               \
    TORCH_CHECK(                                  \
        status == cutlass::Status::kSuccess,      \
        "Got CUTLASS error: ",                    \
        cutlass::cutlassGetStatusString(status)); \
  }

using namespace at;

template <typename T>
struct identity {
  CUTLASS_HOST_DEVICE
  T operator()(T lhs) const {
    return lhs;
  }
};

template <typename ElementA>
struct SparseRowwiseKernel;

template <>
struct SparseRowwiseKernel<cutlass::float_e4m3_t> {
  static constexpr auto kElementOutAt = at::ScalarType::BFloat16;
  static constexpr auto kElementAAt = at::ScalarType::Float8_e4m3fn;

  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementOut = cutlass::bfloat16_t;
  using ElementAccumulator = float;

  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;

  // Epilogue visitor tree
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using AScale =
      cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float>;
  using BScale =
      cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float>;
  using Multiply = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using Cast = cutlass::epilogue::fusion::Sm90Compute<
      identity,
      ElementOut,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      Cast,
      cutlass::epilogue::fusion::Sm90EVT<
          Multiply,
          BScale,
          cutlass::epilogue::fusion::Sm90EVT<Multiply, AScale, Accum>>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassSparseTensorOp,
          TileShape,
          cute::Shape<cute::_2, cute::_1, cute::_1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          float,
          ElementOut,
          cutlass::layout::RowMajor,
          8,
          ElementOut,
          cutlass::layout::RowMajor,
          8,
          cutlass::epilogue::TmaWarpSpecializedCooperative,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassSparseTensorOp,
          ElementA,
          cutlass::layout::RowMajor,
          32,
          ElementB,
          cutlass::layout::ColumnMajor,
          16,
          ElementAccumulator,
          cute::Shape<cute::_128, cute::_256, cute::_128>,
          cute::Shape<cute::_2, cute::_1, cute::_1>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>::
          CollectiveOp;

  // Gemm operator
  // cutlass3x_sm90_sptensorop_s64x256x64spgemm_e4m3_e4m3_f32_bf16_bf16_128x256x128_2x1x1_0_tnt_align32_warpspecialized_cooperative_fp8_fastaccum_epi_tma
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using ElementE = CollectiveMainloop::ElementE;
};

template <>
struct SparseRowwiseKernel<cutlass::bfloat16_t> {
  static constexpr auto kElementOutAt = at::ScalarType::BFloat16;
  static constexpr auto kElementAAt = at::ScalarType::BFloat16;

  using ElementA = cutlass::bfloat16_t;
  using ElementB = cutlass::bfloat16_t;
  using ElementOut = cutlass::bfloat16_t;

  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;

  // Epilogue visitor tree
  using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
  using AScale =
      cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float>;
  using BScale =
      cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float>;
  using Multiply = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies,
      float,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using Cast = cutlass::epilogue::fusion::Sm90Compute<
      identity,
      ElementOut,
      float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      Cast,
      cutlass::epilogue::fusion::Sm90EVT<
          Multiply,
          BScale,
          cutlass::epilogue::fusion::Sm90EVT<Multiply, AScale, Accum>>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassSparseTensorOp,
          TileShape,
          cute::Shape<cute::_2, cute::_1, cute::_1>,
          cutlass::epilogue::collective::EpilogueTileAuto,
          float,
          float,
          ElementOut,
          cutlass::layout::RowMajor,
          8,
          ElementOut,
          cutlass::layout::RowMajor,
          8,
          cutlass::epilogue::TmaWarpSpecializedCooperative,
          EpilogueEVT>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          cutlass::arch::Sm90,
          cutlass::arch::OpClassSparseTensorOp,
          ElementA,
          cutlass::layout::RowMajor,
          16,
          ElementB,
          cutlass::layout::ColumnMajor,
          16,
          float,
          cute::Shape<cute::_128, cute::_128, cute::_64>,
          cute::Shape<cute::_2, cute::_1, cute::_1>,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  // Gemm operator
  // cutlass3x_sm90_sptensorop_s64x128x32spgemm_bf16_bf16_f32_void_f32_128x128x64_2x1x1_0_ttn_align16_warpspecialized_cooperative_epi_tma
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using ElementE = CollectiveMainloop::ElementE;
};

template <bool kIsMeta>
Tensor _sparse24_fp8_sm90_cutlass_gemm(
    const Tensor& tensor_a,
    const Tensor& tensor_e, // metadata for `A`
    const Tensor& tensor_b,
    // *,
    std::optional<at::Tensor> a_scale,
    std::optional<at::Tensor> b_scale,
    int64_t swizzle_size,
    std::string swizzle_axis,
    int64_t sm_count) {
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!kIsMeta) {
    device_guard.emplace(tensor_a.device());
  }

  using K = SparseRowwiseKernel<cutlass::float_e4m3_t>;

  // For now, only CC 9.x devices are supported.
  if (!kIsMeta) {
    const auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        dprops && dprops->major == 9,
        "_sparse24_gemm_fp8_sm90: Supported only on GPUs with "
        "compute capability 9.x");
  }

  // Validate layouts of input tensors.
  TORCH_CHECK(tensor_a.device() == tensor_b.device());
  TORCH_CHECK(tensor_a.device() == tensor_e.device());
  TORCH_CHECK(tensor_a.dim() == 2);
  TORCH_CHECK(tensor_b.dim() == 2);
  TORCH_CHECK(tensor_a.scalar_type() == tensor_b.scalar_type());
  TORCH_CHECK(tensor_a.scalar_type() == K::kElementAAt);
  TORCH_CHECK(tensor_b.stride(0) == 1, "B must be Row-Major");
  TORCH_CHECK(tensor_a.is_contiguous());
  TORCH_CHECK(tensor_b.t().is_contiguous());
  int64_t a_rows = tensor_a.size(0);
  if (a_scale.has_value()) {
    TORCH_CHECK(a_scale->is_contiguous());
    TORCH_CHECK(a_scale->scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(a_scale->device() == tensor_a.device());
    TORCH_CHECK(a_scale->dim() == 2);
    TORCH_CHECK(a_scale->size(0) == a_rows);
    TORCH_CHECK(a_scale->size(1) == 1);
  }
  if (b_scale.has_value()) {
    TORCH_CHECK(b_scale->is_contiguous());
    TORCH_CHECK(b_scale->scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(b_scale->device() == tensor_b.device());
    TORCH_CHECK(b_scale->dim() == 2);
    TORCH_CHECK(b_scale->size(0) == 1);
    TORCH_CHECK(b_scale->size(1) == tensor_b.size(1));
  }

  typename K::GemmKernel::Arguments args;
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = cute::make_shape(
      int(a_rows), int(tensor_b.size(1)), int(tensor_b.size(0)), 1);
  Tensor out = tensor_a.new_empty(
      {cute::get<0>(args.problem_shape), cute::get<1>(args.problem_shape)},
      at::TensorOptions().dtype(K::kElementOutAt));

  args.mainloop.ptr_A =
      reinterpret_cast<K::ElementA const*>(tensor_a.data_ptr());
  args.mainloop.ptr_B = static_cast<K::ElementB const*>(tensor_b.data_ptr());
  args.mainloop.ptr_E =
      reinterpret_cast<K::ElementE const*>(tensor_e.data_ptr());
  args.epilogue.ptr_C = nullptr;
  args.epilogue.ptr_D = static_cast<K::ElementOut*>(out.data_ptr());

  float const* a_scale_ptr =
      (float const*)(a_scale.has_value() ? a_scale->data_ptr() : nullptr);
  float const* b_scale_ptr =
      (float const*)(b_scale.has_value() ? b_scale->data_ptr() : nullptr);
  float default_scale = 1.0f; // used if ptr is nullptr
  auto& cast_op = args.epilogue.thread;
  auto& mulB_op = cast_op.op_0;
  mulB_op.op_0 = {b_scale_ptr, default_scale};
  auto& mulA_op = mulB_op.op_1;
  mulA_op.op_0 = {a_scale_ptr, default_scale};

  args.mainloop.layout_a =
      K::CollectiveMainloop::SparseConfig::fill_layoutA(args.problem_shape);
  args.mainloop.layout_e =
      K::CollectiveMainloop::SparseConfig::fill_layoutE(args.problem_shape);
  args.mainloop.dB = cute::make_int_tuple_from<typename K::GemmKernel::StrideB>(
      tensor_b.stride(1), 0);
  args.epilogue.dC = cute::make_int_tuple_from<typename K::GemmKernel::StrideC>(
      out.stride(0), 0);
  args.epilogue.dD = cute::make_int_tuple_from<typename K::GemmKernel::StrideD>(
      out.stride(0), 0);

  /* Query device SM count to pass onto the kernel as an argument, where needed
   */
  args.hw_info.device_id = tensor_a.device().index();
  args.hw_info.sm_count = sm_count;
  args.scheduler.max_swizzle_size = swizzle_size;
  using Enum_t = decltype(args.scheduler.raster_order);
  if (swizzle_axis == "n") {
    args.scheduler.raster_order = Enum_t::AlongN;
  } else {
    TORCH_CHECK(
        swizzle_axis == "m",
        "Invalid value for swizzle_axis ('",
        swizzle_axis,
        "')");
    args.scheduler.raster_order = Enum_t::AlongM;
  }

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<K::GemmKernel>;
  int64_t device_op_workspace_size = Gemm::get_workspace_size(args);
  Tensor workspace = tensor_a.new_empty(
      {device_op_workspace_size},
      at::TensorOptions().dtype(at::ScalarType::Byte));

  Gemm gemm;
  // Check the problem size is supported or not
  CUTLASS_STATUS_CHECK(gemm.can_implement(args));

  auto status = gemm.run(
      args, (void*)workspace.data_ptr(), at::cuda::getCurrentCUDAStream());
  CUTLASS_STATUS_CHECK(status);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

template <bool kIsMeta, typename ElementT>
std::tuple<Tensor, Tensor> _sparse24_sm90_cutlass_compress_t(Tensor a) {
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!kIsMeta) {
    device_guard.emplace(a.device());
  }

  using K = SparseRowwiseKernel<ElementT>;
  TORCH_CHECK(a.scalar_type() == K::kElementAAt);
  TORCH_CHECK(a.is_contiguous());

  // Offline compressor kernel
  using LayoutA = cutlass::layout::RowMajor;
  using ProblemShape = cute::Shape<int, int, int, int>;
  using SparseConfig = typename K::CollectiveMainloop::SparseConfig;
  using CompressorUtility =
      cutlass::transform::kernel::StructuredSparseCompressorUtility<
          ProblemShape,
          typename K::ElementA,
          LayoutA,
          SparseConfig>;

  using CompressorKernel =
      cutlass::transform::kernel::StructuredSparseCompressor<
          ProblemShape,
          typename K::ElementA,
          LayoutA,
          SparseConfig,
          cutlass::arch::Sm90>;

  using Compressor =
      cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

  auto problem_shape =
      cute::make_shape(int(a.size(0)), 8192, int(a.size(1)), 1);
  auto [M, N, k, L] = problem_shape;
  auto stride_A = cutlass::make_cute_packed_stride(
      cutlass::gemm::TagToStrideA_t<LayoutA>{}, cute::make_shape(M, k, L));
  CompressorUtility compressor_utility(problem_shape, stride_A);

  int ME = compressor_utility.get_metadata_m_physical();
  int KE = compressor_utility.get_metadata_k_physical();
  int KC = compressor_utility.get_tensorA_k_physical();

  auto a_compressed = a.new_empty({M, KC * L});
  auto e = a.new_empty({ME * KE * L}, at::TensorOptions().dtype(at::kByte));

  if (kIsMeta) {
    return std::make_tuple(a_compressed, e);
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = a.device().index();
  hw_info.sm_count = 128;
  typename Compressor::Arguments arguments{
      problem_shape,
      {(typename K::ElementA const*)a.data_ptr(),
       stride_A,
       (typename K::ElementA*)a_compressed.data_ptr(),
       (typename K::ElementE*)e.data_ptr()},
      {hw_info}};

  Compressor compressor_op;
  int64_t workspace_size = Compressor::get_workspace_size(arguments);
  Tensor workspace = a.new_empty(
      {workspace_size}, at::TensorOptions().dtype(at::ScalarType::Byte));

  CUTLASS_STATUS_CHECK(compressor_op.can_implement(arguments));
  CUTLASS_STATUS_CHECK(
      compressor_op.initialize(arguments, workspace.data_ptr()));
  CUTLASS_STATUS_CHECK(compressor_op.run());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return std::make_tuple(a_compressed, e);
}

template <bool kIsMeta>
std::tuple<Tensor, Tensor> _sparse24_sm90_cutlass_compress(Tensor a) {
  if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    return _sparse24_sm90_cutlass_compress_t<kIsMeta, cutlass::float_e4m3_t>(a);
  }
  if (a.scalar_type() == at::ScalarType::BFloat16) {
    return _sparse24_sm90_cutlass_compress_t<kIsMeta, cutlass::bfloat16_t>(a);
  }
  TORCH_CHECK(false, "Unsupported dtype for operand");
}
} // namespace

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_sparse24_fp8_sm90_cutlass_gemm(Tensor a, Tensor a_mdata, Tensor b, *, Tensor? a_scale = None, Tensor? b_scale = None, int swizzle_size=8, str swizzle_axis='n', int sm_count=128) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_sparse24_sm90_cutlass_compress(Tensor a) -> (Tensor, Tensor)"));
}
TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_sparse24_fp8_sm90_cutlass_gemm"),
      TORCH_FN(_sparse24_fp8_sm90_cutlass_gemm<false>));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_sparse24_sm90_cutlass_compress"),
      TORCH_FN(_sparse24_sm90_cutlass_compress<false>));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_sparse24_fp8_sm90_cutlass_gemm"),
      TORCH_FN(_sparse24_fp8_sm90_cutlass_gemm<true>));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_sparse24_sm90_cutlass_compress"),
      TORCH_FN(_sparse24_sm90_cutlass_compress<true>));
}
#endif
#endif
#endif
