#pragma once

#include <ATen/autocast_mode.h>
#include <torch/version.h>

namespace xformers {

// In PyTorch 2.4 (https://github.com/pytorch/pytorch/pull/124359) they renamed
// some functions and immediately marked the old ones as deprecated, causing a
// lot of log spew. For a while we need to support both old and new PyTorch.

inline at::ScalarType get_autocast_cuda_dtype() {
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 4)
  return at::autocast::get_autocast_dtype(at::kCUDA);
#else
  return at::autocast::get_autocast_gpu_dtype();
#endif
}

} // namespace xformers
