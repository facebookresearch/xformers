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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_OUTPUT_TILE_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_OUTPUT_TILE_H_

#include "sputnik/common.h"

namespace sputnik {

template <typename MemOp, int kBlockItemsX, int kBlockItemsY, int kThreadItemsX,
          int kThreadItemsY>
struct OutputTile {
  //
  /// Static members.
  //

  // The number of elements in each vector memory operation.
  static constexpr int kVectorWidth = sizeof(MemOp) / sizeof(float);

  //
  /// Member variables.
  //

  const int kH, kW;
  const float* output_fragment;
  float* out;

  __device__ __forceinline__
  OutputTile(int h, int w, const float* __restrict__ output_fragment_,
             int image_offset, float* __restrict__ out_)
      : kH(h),
        kW(w),
        output_fragment(output_fragment_),
        out(out_ + image_offset) {}

  __device__ __forceinline__ void Store() {
    const int kOutputOffsetH =
        blockIdx.y * kBlockItemsY + threadIdx.y * kThreadItemsY;
    const int kOutputOffsetW =
        blockIdx.x * kBlockItemsX + threadIdx.x * kThreadItemsX;

    // Set the output predicates.
    constexpr int kNumPredicatesH =
        kThreadItemsY / Min(kVectorWidth, kThreadItemsY);
    constexpr int kNumPredicatesW =
        kThreadItemsX / Min(kVectorWidth, kThreadItemsX);
    bool predicates_h[kNumPredicatesH];
    bool predicates_w[kNumPredicatesW];
#pragma unroll
    for (int y_item_idx = 0; y_item_idx < kNumPredicatesH; ++y_item_idx) {
      if (kOutputOffsetH + y_item_idx * kVectorWidth >= kH) {
        predicates_h[y_item_idx] = false;
      } else {
        predicates_h[y_item_idx] = true;
      }
    }

#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kNumPredicatesW; ++x_item_idx) {
      if (kOutputOffsetW + x_item_idx * kVectorWidth >= kW) {
        predicates_w[x_item_idx] = false;
      } else {
        predicates_w[x_item_idx] = true;
      }
    }

    // Write the outputs.
    const MemOp* vector_fragment = OffsetCast<MemOp>(output_fragment, 0);
    const int kOutputOffset = kOutputOffsetH * kW + kOutputOffsetW;
    MemOp* vector_out = OffsetCast<MemOp>(out, kOutputOffset * sizeof(float));
    constexpr int kNumStoresX = kThreadItemsX / kVectorWidth;
#pragma unroll
    for (int y_item_idx = 0; y_item_idx < kThreadItemsY; ++y_item_idx) {
      const int kPredicateIdxY = y_item_idx / kVectorWidth;
      MemOp* inner_vector_out = vector_out;
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kNumStoresX; ++x_item_idx) {
        if (predicates_h[kPredicateIdxY] && predicates_w[x_item_idx]) {
          const int kFragmentIdx = x_item_idx + y_item_idx * kNumStoresX;
          sputnik::Store(vector_fragment[kFragmentIdx], inner_vector_out);
          ++inner_vector_out;
        }
      }
      vector_out += kW / kVectorWidth;
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_OUTPUT_TILE_H_
