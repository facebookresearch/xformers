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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_WIDTH_PADDING_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_WIDTH_PADDING_H_

#include "sputnik/common.h"
#include "sputnik/depthwise/shape_utils.h"

namespace sputnik {

// TODO(tgale): We can reduce the width padding in cases where
// there is guaranteed alignment. For example, 3x3 depthwise
// conv with padding of 1 all base addresses should be 2-aligned
// (assuming input alignment). For this case we'd pad 2 or 0, so
// with a 32x32 output tile we'd only need to load an 34x36
// region of the input rather than a 34x40 region of the input.
template <typename MemOp, int kKernelSize, int kPadding, int kStride,
          int kBlockItemsX>
struct WidthPadding {
  //
  /// Static members.
  //

  // Number of elements in each memory operation.
  static constexpr int kVectorWidth = sizeof(MemOp) / sizeof(float);

  // Helper to calculate the input shape from the output tile size.
  typedef InputShape<kKernelSize, kPadding, kStride> InputShape;

  static __host__ __device__ __forceinline__ constexpr int Get() {
    return RoundUpTo(InputShape::NoPad(kBlockItemsX) + kVectorWidth - 1,
		     kVectorWidth) - kBlockItemsX;
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_WIDTH_PADDING_H_
