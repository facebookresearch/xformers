#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/autocast_mode.h>
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/api/include/torch/nn/modules/linear.h>

namespace {
// Kernels implemented in `cuda/`
std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_silu_identity_mul(
    const at::Tensor& x,
    const at::Tensor& w0,
    const at::Tensor& b0,
    const at::Tensor& w1,
    const at::Tensor& b1
) {
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("xformers::dual_gemm_silu_identity_mul", "")
    .typed<decltype(dual_gemm_silu_identity_mul)>();
  return op.call(x, w0, b0, w1, b1);
}
std::tuple<at::Tensor, at::Tensor> silu_bw_fused(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const at::Tensor& dx4
) {
  static auto op = c10::Dispatcher::singleton()
    .findSchemaOrThrow("xformers::silu_bw_fused", "")
    .typed<decltype(silu_bw_fused)>();
  return op.call(x1, x2, dx4);
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
  TORCH_INTERNAL_ASSERT(\
    shapesMatch(X, {__VA_ARGS__}),\
    "%s: shape is %s but expected %s", \
    #X, shapeToStr(X.sizes()).c_str(), shapeToStr({__VA_ARGS__}).c_str() \
  );

class SwiGLUPackedWeights : public torch::autograd::Function<SwiGLUPackedWeights> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext *ctx, const at::Tensor& x, const at::Tensor& w1w2, const at::Tensor& b1b2, const at::Tensor w3, const at::Tensor b3) {
    at::AutoDispatchBelowADInplaceOrView g;
    auto w1 = w1w2[0];
    auto w2 = w1w2[1];
    auto b1 = b1b2[0];
    auto b2 = b1b2[1];
    at::Tensor x1, x2, x4;
    std::tie(x1, x2, x4) = dual_gemm_silu_identity_mul(x, w1, b1, w2, b2);
    auto x5 = torch::nn::functional::linear(x4, w3, b3);

    ctx->save_for_backward({x, w1w2, w3, x1, x2});
    return x5;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
    at::AutoDispatchBelowADInplaceOrView g;

    // Unpack variables
    auto dx5 = grad_outputs[0];
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    auto w1w2 = saved[1];
    auto w3 = saved[2];
    auto x1 = saved[3];
    auto x2 = saved[4];
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

    auto db3 = dx5.sum(0);
    TORCH_INTERNAL_ASSERT(dx5.size(0) == x4.size(0));
    auto dw3 = torch::mm(dx5.transpose(-2, -1), x4);
    TORCH_INTERNAL_ASSERT_SHAPE(db3, O);
    TORCH_INTERNAL_ASSERT_SHAPE(dw3, O, H);
    x4.reset();
    dx5.reset();

    TORCH_INTERNAL_ASSERT(dx1dx2.is_contiguous());
    TORCH_INTERNAL_ASSERT(w1w2.is_contiguous());

    w1w2 = w1w2.view({2 * H, I});
    dx1dx2 = dx1dx2.view({B, 2 * H});
    auto dx = torch::mm(dx1dx2, w1w2);

    // backward of linear1 + linear2 - packed
    auto dw1dw2 = torch::mm(dx1dx2.transpose(-2, -1), x);
    auto db1db2 = dx1dx2.sum(0);

    auto p = db1db2.view({2, H});
    return {dx, dw1dw2.view({2, H, I}), db1db2.view({2, H}), dw3, db3};
  }
};

at::Tensor swiglu_packedw_autograd(const at::Tensor& x, const at::Tensor& w1w2, const at::Tensor& b1b2, const at::Tensor w3, const at::Tensor b3) {
  return SwiGLUPackedWeights::apply(x, w1w2, b1b2, w3, b3);
}

at::Tensor swiglu_packedw_autocast(const at::Tensor& x, const at::Tensor& w1w2, const at::Tensor& b1b2, const at::Tensor w3, const at::Tensor b3) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_gpu_dtype();
  return SwiGLUPackedWeights::apply(
    at::autocast::cached_cast(exec_type, x),
    at::autocast::cached_cast(exec_type, w1w2),
    at::autocast::cached_cast(exec_type, b1b2),
    at::autocast::cached_cast(exec_type, w3),
    at::autocast::cached_cast(exec_type, b3)
  );
}
}

TORCH_LIBRARY(xformers, m) {
  m.def("swiglu_packedw(Tensor x, Tensor w1w2, Tensor b1b2, Tensor w3, Tensor b3) -> Tensor");
}

TORCH_LIBRARY_IMPL(xformers, Autograd, m) {
  m.impl("swiglu_packedw", swiglu_packedw_autograd);
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl("swiglu_packedw", swiglu_packedw_autocast);
}
