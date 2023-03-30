#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::scaled_index_addF(Tensor output, Tensor? input, Tensor source, Tensor index, Tensor? source_scaling, float alpha) -> Tensor"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::scaled_index_addB(Tensor grad_source, Tensor? grad_source_scaling, Tensor grad_output, Tensor source, Tensor index, Tensor? source_scaling, float alpha) -> (Tensor, Tensor?)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::index_select(Tensor output, Tensor source, Tensor index) -> Tensor"));
}
