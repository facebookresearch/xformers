/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <torch/types.h>

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor> dual_gemm_silu_identity_mul_META(
    const at::Tensor& x,
    const at::Tensor& w0,
    const std::optional<at::Tensor>& /*b0*/,
    const at::Tensor& w1,
    const std::optional<at::Tensor>& /*b1*/) {
  TORCH_CHECK(x.sym_stride(-1) == 1);
  TORCH_CHECK(w0.sym_stride(-1) == 1);
  TORCH_CHECK(w1.sym_stride(-1) == 1);

  at::SymInt B = x.sym_size(0);
  at::SymInt I = x.sym_size(1);
  at::SymInt H = w0.sym_size(0);

  at::Tensor d0 = at::empty_symint({B, H}, x.options());
  at::Tensor d1 = at::empty_symint({B, H}, x.options());
  at::Tensor d2 = at::empty_symint({B, H}, x.options());

  return std::make_tuple(d0, d1, d2);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::dual_gemm_silu_identity_mul(Tensor x, Tensor w1, Tensor? b1, Tensor w2, Tensor? b2) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::silu_bw_fused(Tensor x1, Tensor x2, Tensor dx4) -> (Tensor, Tensor)"));
  m.def(
      TORCH_SELECTIVE_SCHEMA(
          "xformers::gemm_fused_operand_sum(Tensor a, Tensor b) -> (Tensor, Tensor)"),
      {at::Tag::needs_fixed_stride_order});
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::dual_gemm_silu_identity_mul"),
      TORCH_FN(dual_gemm_silu_identity_mul_META));
}
