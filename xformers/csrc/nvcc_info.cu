#include <torch/types.h>

namespace {
std::tuple<int64_t, int64_t, int64_t> nvcc_build_version() {
  return std::make_tuple(
      __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
}
} // namespace

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::_nvcc_build_version() -> (int, int, int)"));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::_nvcc_build_version"),
      TORCH_FN(nvcc_build_version));
}
