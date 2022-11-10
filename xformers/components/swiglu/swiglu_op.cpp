#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::dual_gemm_silu_identity_mul(Tensor x, Tensor w1, Tensor b1, Tensor w2, Tensor b2) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::silu_bw_fused(Tensor x1, Tensor x2, Tensor dx4) -> (Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::gemm_fused_operand_sum(Tensor a, Tensor b, Tensor out_mm, Tensor out_sum) -> (Tensor, Tensor)"));
}
