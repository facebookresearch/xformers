#include <torch/types.h>

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::cuda_memset_32b_async(Tensor buffer, Scalar value, Stream stream) -> ()"));
}
