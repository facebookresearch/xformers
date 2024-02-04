/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <map>
#include <mutex>

#include <c10/hip/HIPCachingAllocator.h>
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
  SimpleDeviceMem(size_t sizeInBytes) {
    pData_ = c10::hip::HIPCachingAllocator::raw_alloc(sizeInBytes);
  }
  void* GetDeviceBuffer() {
    return pData_;
  }
  ~SimpleDeviceMem() {
    c10::cuda::HIPCachingAllocator::raw_delete(pData_);
  }

  void* pData_;
};

// useful aliasing for making the codes easy
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F32 = float;
