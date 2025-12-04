#include <deque>
#include <mutex>
#include <vector>

#include "pt_stable_utils.h"

namespace {

std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
  static bool init_flag [[maybe_unused]] = []() {
    int num_gpus;
    XF_CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    device_flags.resize(num_gpus);
    device_properties.resize(num_gpus);
    return true;
  }();
}

void initDeviceProperty(torch::stable::accelerator::DeviceIndex device_index) {
  cudaDeviceProp device_prop{};
  XF_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // namespace

cudaDeviceProp* xf_getCurrentDeviceProperties() {
  initCUDAContextVectors();
  torch::stable::accelerator::DeviceIndex device =
      torch::stable::accelerator::getCurrentDeviceIndex();
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}
