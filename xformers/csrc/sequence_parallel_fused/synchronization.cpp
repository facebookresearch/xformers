#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::write_values(Tensor(a!)[] ptrs, Scalar values, Stream stream) -> ()"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::wait_values(Tensor[] ptrs, Scalar value, Stream stream, Scalar timeout_s) -> ()"));
}
