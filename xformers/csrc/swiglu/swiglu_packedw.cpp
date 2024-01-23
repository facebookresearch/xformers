/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
// clang-format off
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/autocast_mode.h>
#include <torch/csrc/api/include/torch/nn/modules/linear.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
// clang-format on

namespace {
// Kernels implemented in `cuda/`
std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_silu_identity_mul(
    const at::Tensor& x,
    const at::Tensor& w0,
    const c10::optional<at::Tensor>& b0,
    const at::Tensor& w1,
    const c10::optional<at::Tensor>& b1) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("xformers::dual_gemm_silu_identity_mul", "")
          .typed<decltype(dual_gemm_silu_identity_mul)>();
  return op.call(x, w0, b0, w1, b1);
}
std::tuple<at::Tensor, at::Tensor> silu_bw_fused(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("xformers::silu_bw_fused", "")
                       .typed<decltype(silu_bw_fused)>();
  return op.call(x1, x2, dx4);
}
std::tuple<at::Tensor, at::Tensor> gemm_fused_operand_sum(
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& out_mm,
    at::Tensor& out_sum) {
  static auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("xformers::gemm_fused_operand_sum", "")
          .typed<decltype(gemm_fused_operand_sum)>();
  return op.call(a, b, out_mm, out_sum);
}

bool shapesMatch(at::Tensor x, std::vector<int64_t> expectedShape) {
  if (x.dim() != int64_t(expectedShape.size())) {
    return false;
  }
  for (size_t i = 0; i < expectedShape.size(); ++i) {
    if (expectedShape[i] != -1 && x.size(i) != expectedShape[i]) {
      return false;
    }
  }
  return true;
}

std::string shapeToStr(c10::IntArrayRef shape) {
  std::stringstream oss;
  oss << "[" << shape[0];
  for (size_t i = 1; i < shape.size(); ++i) {
    oss << ", " << shape[i];
  }
  oss << "]";
  return oss.str();
}

#define TORCH_INTERNAL_ASSERT_SHAPE(X, ...) \
  TORCH_INTERNAL_ASSERT(                    \
      shapesMatch(X, {__VA_ARGS__}),        \
      "%s: shape is %s but expected %s",    \
      #X,                                   \
      shapeToStr(X.sizes()).c_str(),        \
      shapeToStr({__VA_ARGS__}).c_str());

class SwiGLUPackedWeights
    : public torch::autograd::Function<SwiGLUPackedWeights> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& x,
      const at::Tensor& w1w2,
      const c10::optional<at::Tensor>& b1b2,
      const at::Tensor w3,
      const c10::optional<at::Tensor>& b3) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto w1 = w1w2[0];
    auto w2 = w1w2[1];
    c10::optional<at::Tensor> b1, b2;
    if (b1b2.has_value()) {
      b1 = b1b2.value()[0];
      b2 = b1b2.value()[1];
    }
    at::Tensor x1, x2, x4;
    std::tie(x1, x2, x4) = dual_gemm_silu_identity_mul(x, w1, b1, w2, b2);
    auto x5 = torch::nn::functional::linear(
        x4, w3, b3.has_value() ? b3.value() : at::Tensor());

    if (ctx != nullptr) {
      ctx->save_for_backward({x, w1w2, w3, x1, x2});
      ctx->saved_data["has_b1b2"] = b1b2.has_value();
      ctx->saved_data["has_b3"] = b3.has_value();
    }
    return x5;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    at::AutoDispatchBelowADInplaceOrView g;

    // Unpack variables
    auto dx5 = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto w1w2 = saved[1];
    auto w3 = saved[2];
    auto x1 = saved[3];
    auto x2 = saved[4];
    bool has_b1b2 = ctx->saved_data["has_b1b2"].toBool();
    bool has_b3 = ctx->saved_data["has_b3"].toBool();
    int64_t B = x.size(0);
    int64_t H = x2.size(1);
    int64_t I = x.size(1);
    int64_t O = dx5.size(1);
    TORCH_INTERNAL_ASSERT_SHAPE(x1, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(x2, B, H);
    TORCH_INTERNAL_ASSERT_SHAPE(dx5, B, O);
    TORCH_INTERNAL_ASSERT_SHAPE(w1w2, 2, H, I);
    TORCH_INTERNAL_ASSERT_SHAPE(w3, O, H);

    // Compute BW
    at::Tensor dx1dx2, x4;
    TORCH_INTERNAL_ASSERT(dx5.size(1) == w3.size(0));
    auto dx4 = torch::mm(dx5, w3);
    std::tie(dx1dx2, x4) = silu_bw_fused(x1, x2, dx4);
    TORCH_INTERNAL_ASSERT_SHAPE(dx1dx2, B, 2, H);
    TORCH_INTERNAL_ASSERT_SHAPE(x4, B, H);
    x1.reset();
    x2.reset();
    dx4.reset();

    at::Tensor db3, dw3;

    if (has_b3) {
      db3 = torch::empty({O}, w3.options());
      dw3 = torch::empty({O, H}, w3.options());
      TORCH_INTERNAL_ASSERT(dx5.size(0) == x4.size(0));
      gemm_fused_operand_sum(dx5.transpose(-2, -1), x4, dw3, db3);
    } else {
      dw3 = torch::mm(dx5.transpose(-2, -1), x4);
    }
    x4.reset();
    dx5.reset();

    TORCH_INTERNAL_ASSERT(dx1dx2.is_contiguous());
    TORCH_INTERNAL_ASSERT(w1w2.is_contiguous());

    w1w2 = w1w2.view({2 * H, I});
    dx1dx2 = dx1dx2.view({B, 2 * H});
    auto dx = torch::mm(dx1dx2, w1w2);

    // backward of linear1 + linear2 - packed
    at::Tensor dw1dw2, db1db2;
    if (has_b1b2) {
      dw1dw2 = torch::empty({2 * H, I}, w1w2.options());
      db1db2 = torch::empty({2 * H}, w1w2.options());
      gemm_fused_operand_sum(dx1dx2.transpose(-2, -1), x, dw1dw2, db1db2);
      db1db2 = db1db2.view({2, H});
    } else {
      dw1dw2 = torch::mm(dx1dx2.transpose(-2, -1), x);
    }

    return {dx, dw1dw2.view({2, H, I}), db1db2, dw3, db3};
  }
};

at::Tensor swiglu_packedw_autograd(
    const at::Tensor& x,
    const at::Tensor& w1w2,
    const c10::optional<at::Tensor> b1b2,
    const at::Tensor w3,
    const c10::optional<at::Tensor> b3) {
  return SwiGLUPackedWeights::apply(x, w1w2, b1b2, w3, b3);
}

at::Tensor swiglu_packedw_autocast(
    const at::Tensor& x,
    const at::Tensor& w1w2,
    const c10::optional<at::Tensor> b1b2,
    const at::Tensor w3,
    const c10::optional<at::Tensor> b3) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_gpu_dtype();
  return SwiGLUPackedWeights::apply(
      at::autocast::cached_cast(exec_type, x),
      at::autocast::cached_cast(exec_type, w1w2),
      at::autocast::cached_cast(exec_type, b1b2),
      at::autocast::cached_cast(exec_type, w3),
      at::autocast::cached_cast(exec_type, b3));
}

at::Tensor swiglu_packedw_cuda(
    const at::Tensor& x,
    const at::Tensor& w1w2,
    const c10::optional<at::Tensor> b1b2,
    const at::Tensor w3,
    const c10::optional<at::Tensor> b3) {
  if (x.requires_grad()) {
    return SwiGLUPackedWeights::apply(x, w1w2, b1b2, w3, b3);
  } else {
    return SwiGLUPackedWeights::forward(
        /* ctx */ nullptr, x, w1w2, b1b2, w3, b3);
  }
}
} // namespace

TORCH_LIBRARY(xformers, m) {
  m.def(
      "swiglu_packedw(Tensor x, Tensor w1w2, Tensor? b1b2, Tensor w3, Tensor? b3) -> Tensor");
}

TORCH_LIBRARY_IMPL(xformers, Autograd, m) {
  m.impl("swiglu_packedw", swiglu_packedw_autograd);
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl("swiglu_packedw", swiglu_packedw_autocast);
}

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl("swiglu_packedw", swiglu_packedw_cuda);
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl("swiglu_packedw", swiglu_packedw_cuda);
}
