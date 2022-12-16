#include <ATen/ATen.h>
#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::sddmm_sputnik(Tensor a, Tensor b, Tensor row_indices, Tensor row_offsets, Tensor column_indices) -> Tensor"));
}
