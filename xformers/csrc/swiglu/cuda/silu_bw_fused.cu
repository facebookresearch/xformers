/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <ATen/native/cuda/Loops.cuh>

namespace {
/*
Computes the following:

def silu_bw_fused(x1, x2, dx4):
    x3 = F.silu(x1)
    dx3 = dx4 * x2
    dx2 = dx4 * x3
    x4 = x2 * x3  # checkpointing
    # silu bw
    sigm = 1 / (1 + torch.exp(-x1.float()))
    dx1 = (dx3.float() * sigm * (1 + x1.float() * (1 - sigm))).to(x1.dtype)
    return dx1, dx2, x4
*/

template <bool kIsMeta = false>
std::tuple<at::Tensor, at::Tensor> silu_bw_fused(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4) {
  // TODO: Check all params. This would take a lot of lines of code...
  TORCH_CHECK(x2.dim() == 2);
  TORCH_CHECK(dx4.dim() == 2);
  TORCH_CHECK(x2.sym_size(0) == dx4.sym_size(0));
  TORCH_CHECK(x2.sym_size(1) == dx4.sym_size(1));

  at::SymInt B = x2.sym_size(0);
  at::SymInt H = x2.sym_size(1);
  at::Tensor dx1dx2 = at::empty_symint({B, 2, H}, x2.options());
  at::Tensor x4 = at::empty_symint({B, H}, x2.options());

  // Check if the function is in meta mode
  if (kIsMeta) {
    // Meta mode logic: return tensors with appropriate shapes
    // Infer the shapes based on input tensors and return empty tensors with
    // those shapes
    return std::make_tuple(dx1dx2, x4);
  }

  // Regular mode logic: perform actual computations
  at::Tensor dx1 = dx1dx2.select(1, 0);
  at::Tensor dx2 = dx1dx2.select(1, 1);

  auto iter = at::TensorIteratorConfig()
                  .add_output(dx1)
                  .add_output(dx2)
                  .add_output(x4)
                  .add_input(x1)
                  .add_input(x2)
                  .add_input(dx4)
                  .check_all_same_dtype(false)
                  .promote_inputs_to_common_dtype(false)
                  .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      x2.scalar_type(),
      "silu_bw_fused",
      ([&] {
        using acc_t = typename at::AccumulateType<scalar_t, true>::type;
        at::native::gpu_kernel_multiple_outputs(
            iter,
            [=] GPU_LAMBDA(scalar_t x1_, scalar_t x2_, scalar_t dx4_)
                -> thrust::tuple<scalar_t, scalar_t, scalar_t> {
              acc_t sigm = acc_t(1) / (acc_t(1) + std::exp(-acc_t(x1_)));
              acc_t x3_ = sigm * x1_;
              acc_t dx3_ = acc_t(dx4_) * acc_t(x2_);
              acc_t dx2_ = acc_t(dx4_) * acc_t(x3_);
              acc_t dx1_ =
                  (dx3_ * sigm * (acc_t(1) + acc_t(x1_) * (acc_t(1) - sigm)));
              acc_t x4_ = x3_ * x2_;
              return thrust::tuple<scalar_t, scalar_t, scalar_t>{
                  dx1_, dx2_, x4_};
            });
      }));
  return std::make_tuple(dx1dx2, x4);
}

std::tuple<at::Tensor, at::Tensor> silu_bw_fused_autocast(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_gpu_dtype();
  return silu_bw_fused(
      at::autocast::cached_cast(exec_type, x1),
      at::autocast::cached_cast(exec_type, x2),
      at::autocast::cached_cast(exec_type, dx4));
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::silu_bw_fused"),
      TORCH_FN(silu_bw_fused<false>));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::silu_bw_fused"),
      TORCH_FN(silu_bw_fused<true>));
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::silu_bw_fused"),
      TORCH_FN(silu_bw_fused_autocast));
}
