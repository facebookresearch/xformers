#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

namespace {

template <typename T>
T getCudaDriverSymbol(const char* name) {
  void* fn = nullptr;
#if CUDA_VERSION >= 12000
  enum cudaDriverEntryPointQueryResult queryResult;
  C10_CUDA_CHECK(
      cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault, &queryResult));
  TORCH_CHECK(
      queryResult == cudaDriverEntryPointSuccess,
      "Querying the ",
      name,
      " symbol from the CUDA driver failed with error ",
      queryResult);
#else // CUDA_VERSION < 12000
  C10_CUDA_CHECK(cudaGetDriverEntryPoint(name, &fn, cudaEnableDefault));
#endif // CUDA_VERSION
  TORCH_CHECK(
      fn != nullptr,
      "Querying the ",
      name,
      " symbol from the CUDA driver returned a null pointer");
  return reinterpret_cast<T>(fn);
}

void raiseCudaDriverError(CUresult result, const char* fnName) {
  static PFN_cuGetErrorName myCuGetErrorName =
      getCudaDriverSymbol<PFN_cuGetErrorName>("cuGetErrorName");
  static PFN_cuGetErrorString myCuGetErrorString =
      getCudaDriverSymbol<PFN_cuGetErrorString>("cuGetErrorString");

  const char* ptr;
  CUresult subResult = myCuGetErrorName(result, &ptr);
  std::string errorName = subResult == CUDA_SUCCESS ? ptr : "UNKNOWN";
  subResult = myCuGetErrorString(result, &ptr);
  std::string errorString = subResult == CUDA_SUCCESS ? ptr : "???";

  TORCH_CHECK(
      result == CUDA_SUCCESS,
      "Calling ",
      fnName,
      " from the CUDA driver failed with error ",
      errorName,
      " (code ",
      result,
      "): ",
      errorString);
}

void cudaMemcpy32bAsync(
    torch::Tensor buffer,
    torch::Scalar value,
    torch::Stream stream) {
  static PFN_cuMemsetD32Async myCuMemsetD32Async =
      getCudaDriverSymbol<PFN_cuMemsetD32Async>("cuMemsetD32Async");

  TORCH_CHECK(buffer.is_cuda());
  TORCH_CHECK(buffer.dtype() == torch::kInt32);
  TORCH_CHECK(buffer.is_non_overlapping_and_dense());
  TORCH_CHECK(value.isIntegral(/*includeBool=*/false));
  CUresult result = myCuMemsetD32Async(
      reinterpret_cast<CUdeviceptr>(buffer.data_ptr()),
      static_cast<unsigned int>(value.toInt()),
      buffer.numel(),
      c10::cuda::CUDAStream(stream).stream());
  if (result != CUDA_SUCCESS) {
    raiseCudaDriverError(result, "cuMemsetD32Async");
  }
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::cuda_memset_32b_async"),
      TORCH_FN(cudaMemcpy32bAsync));
}
