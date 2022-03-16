#include "matmul.h"
#include <torch/types.h>
#include <limits>

/*
at::Tensor matmul_with_mask(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("xformers::matmul_with_mask", "")
                       .typed<decltype(matmul_with_mask)>();
  auto result = op.call(a, b, mask);
  return result;
}
*/

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::efficient_attention(Tensor query, Tensor key, Tensor value) -> Tensor"));
}
