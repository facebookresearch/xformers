/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <45_dual_gemm/device/dual_gemm.h>
#include "epilogue_lhs_activation_and_mul.h"

namespace {

template <typename T>
using SiLu = cutlass::epilogue::thread::SiLu<T>;

template <typename scalar_t, template <typename> typename ActivationFn>
std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_lhs_activation_and_mul_(
    const at::Tensor& x,
    const at::Tensor& w0,
    const std::optional<at::Tensor>& b0,
    const at::Tensor& w1,
    const std::optional<at::Tensor>& b1) {
  TORCH_CHECK(x.dim() == 2);
  TORCH_CHECK(w0.dim() == 2);
  TORCH_CHECK(w1.dim() == 2);

  TORCH_CHECK(x.stride(-1) == 1);
  TORCH_CHECK(w0.stride(-1) == 1);
  TORCH_CHECK(w1.stride(-1) == 1);

  at::cuda::CUDAGuard device_guard(x.device());

  int64_t B = x.size(0);
  int64_t I = x.size(1);
  int64_t H = w0.size(0);

  at::Tensor d0 = at::empty({B, H}, x.options());
  at::Tensor d1 = at::empty({B, H}, x.options());
  at::Tensor d2 = at::empty({B, H}, x.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using EpilogueOutputOp2 = EpilogueLHSActivationAndMul<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
      ActivationFn,
      ElementOutput,
      ElementCompute>;

  const ElementCompute alpha0 = ElementCompute(1);
  const ElementCompute beta0 =
      b0.has_value() ? ElementCompute(1) : ElementCompute(0);
  const ElementCompute alpha1 = ElementCompute(1);
  const ElementCompute beta1 =
      b1.has_value() ? ElementCompute(1) : ElementCompute(0);

  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

  // Optionally, we might not need intermediate GEMM outputs
  constexpr bool kStoreD0 = true;
  constexpr bool kStoreD1 = true;
  using ArchTag = cutlass::arch::Sm80;

  using DualGemm = cutlass::gemm::device::DualGemm<
      scalar_t,
      cutlass::layout::RowMajor,
      scalar_t,
      cutlass::layout::ColumnMajor,
      cutlass::layout::ColumnMajor,
      ElementOutput,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp01,
      EpilogueOutputOp01,
      EpilogueOutputOp2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>,
      kStages,
      kStoreD0,
      kStoreD1,
      kSplitKSerial>;
  {
    cudaDeviceProp* p = at::cuda::getDeviceProperties(x.device().index());
    TORCH_CHECK(
        p->major * 10 + p->minor >= ArchTag::kMinComputeCapability,
        "Only A100+ GPUs are supported");
  }

  int split_k_slices = DualGemm::kSplitKSerial ? 2 : 1;
  using RefA = typename cutlass::
      TensorRef<typename DualGemm::ElementA, typename DualGemm::LayoutA>;
  using RefB0 = typename cutlass::
      TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB0>;
  using RefB1 = typename cutlass::
      TensorRef<typename DualGemm::ElementB, typename DualGemm::LayoutB1>;
  using RefC = typename cutlass::
      TensorRef<typename DualGemm::ElementC, typename DualGemm::LayoutC>;
  RefC ref_b0, ref_b1;
  if (b0.has_value()) {
    ref_b0 =
        RefC{(scalar_t*)b0->data_ptr(), typename DualGemm::LayoutC::Stride(0)};
  }
  if (b1.has_value()) {
    ref_b1 =
        RefC{(scalar_t*)b1->data_ptr(), typename DualGemm::LayoutC::Stride(0)};
  }
  typename DualGemm::Arguments arguments{
      cutlass::gemm::DualGemmMode::kGemm,
      problem_size,
      RefA{
          (scalar_t*)x.data_ptr(),
          typename DualGemm::LayoutA::Stride(x.stride(0))},
      RefB0{
          (scalar_t*)w0.data_ptr(),
          typename DualGemm::LayoutB0::Stride(w0.stride(0))},
      ref_b0,
      RefC{
          (scalar_t*)d0.data_ptr(),
          typename DualGemm::LayoutC::Stride(d0.stride(0))},
      RefB1{
          (scalar_t*)w1.data_ptr(),
          typename DualGemm::LayoutB1::Stride(w1.stride(0))},
      ref_b1,
      RefC{
          (scalar_t*)d1.data_ptr(),
          typename DualGemm::LayoutC::Stride(d1.stride(0))},
      RefC{
          (scalar_t*)d2.data_ptr(),
          typename DualGemm::LayoutC::Stride(d2.stride(0))},
      typename DualGemm::EpilogueOutputOp0::Params{alpha0, beta0},
      typename DualGemm::EpilogueOutputOp1::Params{alpha1, beta1},
      typename DualGemm::EpilogueOutputOp2::Params{},
      split_k_slices};

  DualGemm dual_gemm;
  at::Tensor workspace = at::empty(
      {int64_t(dual_gemm.get_workspace_size(arguments))},
      x.options().dtype(at::ScalarType::Byte));
  cutlass::Status status = dual_gemm.can_implement(arguments);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "`dual_gemm_lhs_activation_and_mul` does not support this input: ",
      cutlass::cutlassGetStatusString(status));

  status = dual_gemm.initialize(arguments, (uint8_t*)workspace.data_ptr());
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel initialize failed");
  status = dual_gemm(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "kernel run failed");

  return std::make_tuple(d0, d1, d2);
}

template <template <typename> typename ActivationFn>
std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_lhs_activation_and_mul(
    const at::Tensor& x,
    const at::Tensor& w0,
    const std::optional<at::Tensor>& b0,
    const at::Tensor& w1,
    const std::optional<at::Tensor>& b1) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(x.dim() == 2);
  TORCH_CHECK(w0.dim() == 2);
  TORCH_CHECK(w1.dim() == 2);

#define FWD_PARAMS x, w0, b0, w1, b1
  if (x.scalar_type() == at::ScalarType::Half) {
    return dual_gemm_lhs_activation_and_mul_<cutlass::half_t, ActivationFn>(
        FWD_PARAMS);
  } else {
    TORCH_CHECK(
        x.scalar_type() == at::ScalarType::BFloat16, "Only supports bf16/f16");
    return dual_gemm_lhs_activation_and_mul_<cutlass::bfloat16_t, ActivationFn>(
        FWD_PARAMS);
  }
}

template <template <typename> typename ActivationFn>
std::tuple<at::Tensor, at::Tensor, at::Tensor>
dual_gemm_lhs_activation_and_mul_autocast(
    const at::Tensor& x,
    const at::Tensor& w0,
    const std::optional<at::Tensor>& b0,
    const at::Tensor& w1,
    const std::optional<at::Tensor>& b1) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_dtype(at::kCUDA);
  return dual_gemm_lhs_activation_and_mul<ActivationFn>(
      at::autocast::cached_cast(exec_type, x),
      at::autocast::cached_cast(exec_type, w0),
      at::autocast::cached_cast(exec_type, b0),
      at::autocast::cached_cast(exec_type, w1),
      at::autocast::cached_cast(exec_type, b1));
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::dual_gemm_silu_identity_mul"),
      TORCH_FN(dual_gemm_lhs_activation_and_mul<SiLu>));
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::dual_gemm_silu_identity_mul"),
      TORCH_FN(dual_gemm_lhs_activation_and_mul_autocast<SiLu>));
}
