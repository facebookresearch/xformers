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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_COMPUTER_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_COMPUTER_H_

#include "sputnik/depthwise/width_padding.h"

namespace sputnik {

template <typename MemOp, int kKernelSize, int kPadding, int kStride,
          int kBlockItemsX, int kBlockItemsY, int kThreadItemsX,
          int kThreadItemsY>
struct Computer {
  //
  /// Static members.
  //

  typedef WidthPadding<MemOp, kKernelSize, kPadding, kStride, kBlockItemsX>
      WidthPadding;

  typedef InputShape<kKernelSize, kPadding, kStride> InputShape;

  // The width padding to enable vector memory instructions.
  static constexpr int kWidthPadding = WidthPadding::Get();

  // The width of the input tile with padding.
  static constexpr int kInputTileX = kBlockItemsX + kWidthPadding;

  // The number of elements in each vector memory operation.
  static constexpr int kVectorWidth = sizeof(MemOp) / sizeof(float);

  //
  /// Member variables.
  //

  const int kH, kW;
  const float* image_tile;
  const float* filter_fragment;
  float* output_fragment;

  __device__ __forceinline__
  Computer(int h, int w, const float* __restrict__ image_tile_,
           const float* __restrict__ filter_fragment_,
           float* __restrict__ output_fragment_)
      : kH(h),
        kW(w),
        image_tile(image_tile_),
        filter_fragment(filter_fragment_),
        output_fragment(output_fragment_) {}

  __device__ __forceinline__ void Compute() {
    const int kTileOffsetW = threadIdx.x * kThreadItemsX * kStride;
    const int kTileOffsetH = threadIdx.y * kThreadItemsY * kStride;
    image_tile += kTileOffsetW + kTileOffsetH * kInputTileX;

    // Load all of the pixels to registers.
    //
    // TODO(tgale): There are a variable number of bank conflicts
    // here depending on the configuration. Straighten this out.
    constexpr int kWindowX = InputShape::NoPad(kThreadItemsX);
    constexpr int kWindowY = InputShape::NoPad(kThreadItemsY);
    float pixels[kWindowX * kWindowY];
#pragma unroll
    for (int y_item_idx = 0; y_item_idx < kWindowY; ++y_item_idx) {
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kWindowX; ++x_item_idx) {
        const int kPixelIdx = y_item_idx * kInputTileX + x_item_idx;
        pixels[y_item_idx * kWindowX + x_item_idx] = image_tile[kPixelIdx];
      }
    }

    // TODO(tgale): This almost certainly doesn't generalize beyond
    // 3x3 filters with 'same' padding or beyond stride 1.
    //
    // TODO(tgale): For problems where the h-tile is smaller than the
    // vector width, we'll need to replace kVectorWidth with
    // kThreadItemsY.
    constexpr int kVectorWidthH = Min(kVectorWidth, kThreadItemsY);
    const int kInputOffsetH =
        blockIdx.y * kBlockItemsY * kStride - kPadding + kTileOffsetH;
    bool top_predicate = kInputOffsetH < 0 ? false : true;
    constexpr int kBottomPredicates = DivUp(kThreadItemsY, kVectorWidthH);
    bool bottom_predicates[kBottomPredicates];

    const int kBottomInputH =
        kInputOffsetH + (kKernelSize - 1) + (kVectorWidthH - 1) * kStride;
#pragma unroll
    for (int y_item_idx = 0; y_item_idx < kBottomPredicates; ++y_item_idx) {
      if (kBottomInputH + y_item_idx * kVectorWidthH * kStride >= kH) {
        bottom_predicates[y_item_idx] = false;
      } else {
        bottom_predicates[y_item_idx] = true;
      }
    }

    const int kInputOffsetW =
        blockIdx.x * kBlockItemsX * kStride - kPadding + kTileOffsetW;
    bool left_predicate = kInputOffsetW < 0 ? false : true;
    constexpr int kRightPredicates = DivUp(kThreadItemsX, kVectorWidth);
    bool right_predicates[kRightPredicates];

    const int kRightInputW =
        kInputOffsetW + (kKernelSize - 1) + (kVectorWidth - 1) * kStride;
#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kRightPredicates; ++x_item_idx) {
      if (kRightInputW + x_item_idx * kVectorWidth * kStride >= kW) {
        right_predicates[x_item_idx] = false;
      } else {
        right_predicates[x_item_idx] = true;
      }
    }

    // Compute the results.
#pragma unroll
    for (int oy_idx = 0; oy_idx < kThreadItemsY; ++oy_idx) {
#pragma unroll
      for (int ox_idx = 0; ox_idx < kThreadItemsX; ++ox_idx) {
#pragma unroll
        for (int fx_idx = 0; fx_idx < kKernelSize; ++fx_idx) {
#pragma unroll
          for (int fy_idx = 0; fy_idx < kKernelSize; ++fy_idx) {
            // Find the weight.
            const int kWeightIdx = fy_idx * kKernelSize + fx_idx;
            float weight = filter_fragment[kWeightIdx];

            // Find the pixel.
            const int kPixelIdxX = fx_idx + ox_idx * kStride;
            const int kPixelIdxY = fy_idx + oy_idx * kStride;
            const int kPixelIdx = kPixelIdxY * kWindowX + kPixelIdxX;
            float pixel = pixels[kPixelIdx];

            const int kPredicateIdxY = oy_idx / kVectorWidthH;
            const int kPredicateIdxX = ox_idx / kVectorWidth;
            if ((kPixelIdxY != 0 || top_predicate) &&
                (fy_idx != (kKernelSize - 1) ||
                 (oy_idx + 1) % kVectorWidthH != 0 ||
                 bottom_predicates[kPredicateIdxY]) &&
                (kPixelIdxX != 0 || left_predicate) &&
                (fx_idx != (kKernelSize - 1) ||
                 (ox_idx + 1) % kVectorWidth != 0 ||
                 right_predicates[kPredicateIdxX])) {
              // Compute the product and sum.
              output_fragment[oy_idx * kThreadItemsX + ox_idx] +=
                  weight * pixel;
            }
          }
        }
      }
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_COMPUTER_H_
