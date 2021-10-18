// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_SPUTNIK_LOAD_STORE_H_
#define THIRD_PARTY_SPUTNIK_LOAD_STORE_H_

/**
 * @file @brief Defines utilities for loading and storing data.
 */

#include <cstring>
#include "sputnik/cuda_utils.h"

namespace sputnik {

template <class To, class From>
__device__ __forceinline__ To BitCast(const From& src) noexcept {
  To dst;
  std::memcpy(&dst, &src, sizeof(To));
  return dst;
}

template <typename T>
__device__ __forceinline__ void Store(const T& value, T* ptr) {
  *ptr = value;
}

__device__ __forceinline__ void Store(const half8& value, half8* ptr) {
  *reinterpret_cast<float4*>(ptr) = BitCast<float4>(value);
}

__device__ __forceinline__ void Store(const half4& value, half4* ptr) {
  *reinterpret_cast<float2*>(ptr) = BitCast<float2>(value);
}

__device__ __forceinline__ void Store(const short8& value, short8* ptr) {
  *reinterpret_cast<int4*>(ptr) = BitCast<int4>(value);
}

__device__ __forceinline__ void Store(const short4& value, short4* ptr) {
  *reinterpret_cast<int2*>(ptr) = BitCast<int2>(value);
}

template <typename T>
__device__ __forceinline__ T Load(const T* address) {
  return __ldg(address);
}

__device__ __forceinline__ half4 Load(const half4* address) {
  float2 x = __ldg(reinterpret_cast<const float2*>(address));
  return BitCast<half4>(x);
}

__device__ __forceinline__ half8 Load(const half8* address) {
  float4 x = __ldg(reinterpret_cast<const float4*>(address));
  return BitCast<half8>(x);
}

__device__ __forceinline__ short4 Load(const short4* address) {
  int2 x = __ldg(reinterpret_cast<const int2*>(address));
  return BitCast<short4>(x);
}

__device__ __forceinline__ short8 Load(const short8* address) {
  int4 x = __ldg(reinterpret_cast<const int4*>(address));
  return BitCast<short8>(x);
}

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_LOAD_STORE_H_
