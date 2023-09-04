#pragma once

#include <torch/torch.h>

#include <ck/utility/data_type.hpp>

template <typename scalar_t>
struct MaxVectorSizeForType {
  static constexpr int value = 4;
};

template <>
struct MaxVectorSizeForType<ck::half_t> {
  static constexpr int value = 8;
};

template <>
struct MaxVectorSizeForType<ck::bhalf_t> {
  static constexpr int value = 8;
};

struct SimpleDeviceMem {
  SimpleDeviceMem() = delete;
  SimpleDeviceMem(std::size_t mem_size) {
    auto options = torch::TensorOptions();
    mem = at::empty(
        mem_size, options.dtype(at::ScalarType::Byte).device(torch::kCUDA));
  }
  void* GetDeviceBuffer() {
    return mem.data_ptr();
  }
  ~SimpleDeviceMem() {}

  at::Tensor mem;
};

// useful aliasing for making the codes easy
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;
