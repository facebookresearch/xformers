/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "../matmul.h"
#include <ATen/ATen.h>
#include <torch/autograd.h>
#include <torch/types.h>

namespace {

class MatmulWithMask : public torch::autograd::Function<MatmulWithMask> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& a,
      const torch::autograd::Variable& b,
      const torch::autograd::Variable& mask) {
    // optimization: only need to save the mask if it's dense
    if (mask.is_sparse())
      ctx->save_for_backward({a, b});
    else
      ctx->save_for_backward({a, b, mask});
    at::AutoNonVariableTypeMode g;
    auto result = matmul_with_mask(a, b, mask);
    return {result};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_output) {
    // Use data saved in forward
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];

    auto grad_o = grad_output[0];
    if (saved.size() == 3) {
      // mask is dense, need to mask manually
      auto mask = saved[2];
      grad_o = grad_o.masked_fill(mask.logical_not(), 0.);
    }
    // TODO: compute grad only if they require grad
    auto grad_a = grad_o.bmm(b.transpose(-2, -1));
    auto grad_b = grad_o.transpose(-2, -1).bmm(a).transpose(-2, -1);
    return {grad_a, grad_b, torch::autograd::Variable()};
  }
};

at::Tensor matmul_with_mask_autograd(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask) {
  return MatmulWithMask::apply(a, b, mask)[0];
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, Autograd, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::matmul_with_mask"),
      TORCH_FN(matmul_with_mask_autograd));
}
