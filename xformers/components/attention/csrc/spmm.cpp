#include <ATen/ATen.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::spmm_sputnik(Tensor b, Tensor row_indices, Tensor values, Tensor row_offsets, Tensor column_indices, int m) -> Tensor"));
}
