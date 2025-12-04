#include <torch/csrc/stable/library.h>

#include "pt_stable_utils.h"

namespace {
std::tuple<int64_t, int64_t, int64_t> nvcc_build_version() {
  return std::make_tuple(
      __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
}
} // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def("_nvcc_build_version() -> (int, int, int)");
  m.impl("_nvcc_build_version", XF_BOXED_FN(nvcc_build_version));
}
