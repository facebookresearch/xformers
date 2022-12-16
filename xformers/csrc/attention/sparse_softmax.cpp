#include <ATen/ATen.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sparse_softmax_sputnik(int m, int n, Tensor row_indices, Tensor values, Tensor row_offsets, Tensor column_indices) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sparse_softmax_backward_sputnik(int m, int n, Tensor row_indices, Tensor values, Tensor gradient, Tensor row_offsets, Tensor column_indices) -> Tensor"));
}
