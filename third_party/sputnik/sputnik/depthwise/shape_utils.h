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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_SHAPE_UTILS_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_SHAPE_UTILS_H_

#include <utility>

#include "sputnik/common.h"

namespace sputnik {

template <int kKernelSize, int kPadding, int kStride>
struct InputShape {
  // Calculate the number of inputs needed to produce an output
  // of dimension `x`, ignoring any implicit padding.
  static constexpr __host__ __device__ __forceinline__ int NoPad(int x) {
    return (x - 1) * kStride + kKernelSize;
  }
};

template <int kKernelSize, int kPadding, int kStride>
struct OutputShape {
  static constexpr __host__ __device__ __forceinline__ int Get(int x) {
    return (x - kKernelSize + 2 * kPadding) / kStride + 1;
  }
};

template <int kBlockDimX, int kBlockDimY>
struct ThreadBlock {
  struct ThreadIdx {
    int x, y;
  };

  template <int kNewBlockDimX>
  static __device__ __forceinline__ ThreadIdx Reshape() {
    // NOTE: We need the dimensions to be powers of two to do this
    // with bitwise ops instead of mod/div. We only need to support
    // a couple of cases.
    static_assert(kNewBlockDimX == 1 || kNewBlockDimX == 2 ||
                  kNewBlockDimX == 4);
    const int kThreadIdx = threadIdx.x + threadIdx.y * kBlockDimX;

    const int kBits = Log2(kNewBlockDimX);
    ThreadIdx thread_idx;
    thread_idx.x = kThreadIdx & kBits;
    thread_idx.y = kThreadIdx >> kBits;
    return thread_idx;
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_SHAPE_UTILS_H_
