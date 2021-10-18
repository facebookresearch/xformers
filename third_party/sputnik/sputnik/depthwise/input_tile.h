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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_INPUT_TILE_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_INPUT_TILE_H_

#include <cstdint>

#include "sputnik/common.h"
#include "sputnik/depthwise/width_padding.h"
#include "sputnik/load_store.h"
#include "sputnik/tiling_utils.h"

namespace sputnik {

template <typename MemOp, int kKernelSize, int kPadding, int kStride,
          int kBlockItemsX, int kBlockItemsY, int kBlockDimX, int kBlockDimY>
struct InputTile {
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

  // The height of the input tile with padding.
  static constexpr int kInputTileY = InputShape::NoPad(kBlockItemsY);

  // The number of elements in each vector memory operation.
  static constexpr int kVectorWidth = sizeof(MemOp) / sizeof(float);

  //
  /// Member variables.
  //

  const int kW, kBaseInputOffset;
  const float* in;
  float* in_tile;

  __device__ __forceinline__ InputTile(int w, int input_offset,
                                       const float* __restrict__ in_,
                                       int tile_offset,
                                       float* __restrict__ in_tile_)
      : kW(w), kBaseInputOffset(input_offset), in(in_) {
    in_tile = in_tile_ + tile_offset;
  }

  __device__ __forceinline__ void Load() {
    // Step 1: Load the regular sized region of the input region.
    // I.e., the region that doesn't take into account the halo
    // size and the width padding from ROMA.

    // Align the base offset.
    constexpr uint32_t kAlignmentMask = ~(kVectorWidth - 1);
    const int kAlignedBaseInputOffset = kBaseInputOffset & kAlignmentMask;
    const int kInputOffset = (kAlignedBaseInputOffset + threadIdx.y * kW +
                              threadIdx.x * kVectorWidth) *
                             sizeof(float);

    const int kTileOffset =
        (threadIdx.y * kInputTileX + threadIdx.x * kVectorWidth) *
        sizeof(float);

    // Vector versions of the input and shared memory pointers.
    const MemOp* vector_in = OffsetCast<const MemOp>(in, kInputOffset);
    MemOp* vector_in_tile = OffsetCast<MemOp>(in_tile, kTileOffset);

    constexpr int kStepsY = kInputTileY / kBlockDimY;
    constexpr int kStepsX = kInputTileX / (kBlockDimX * kVectorWidth);

    // TODO(tgale): Need to add predicates in both dimensions. Only
    // need to check every kVectorWidth values in each dimension.
    //
    // Each thread should set kStepsX predicates prior to starting
    // this loop. We can reuse these in step 2.
    //
    // Each thread should set a height predicate every kVectorWidth
    // (or greater) values. If this is set after exiting loop, we
    // can skip step 2 and 4.
#pragma unroll
    for (int y_item_idx = 0; y_item_idx < kStepsY; ++y_item_idx) {
      const MemOp* inner_vector_in = vector_in;
      MemOp* inner_vector_in_tile = vector_in_tile;
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kStepsX; ++x_item_idx) {
        Store(sputnik::Load(inner_vector_in), inner_vector_in_tile);
        inner_vector_in_tile += kBlockDimX;
        inner_vector_in += kBlockDimX;
      }
      vector_in += kBlockDimY / kVectorWidth * kW;
      vector_in_tile += kBlockDimY * kInputTileX / kVectorWidth;
    }

    // Step 2: Load the y-dimension residue region.
    constexpr int kResidueY = kInputTileY % kBlockDimY;

    if (threadIdx.y < kResidueY) {
      const MemOp* inner_vector_in = vector_in;
      MemOp* inner_vector_in_tile = vector_in_tile;
#pragma unroll
      for (int x_item_idx = 0; x_item_idx < kStepsX; ++x_item_idx) {
        Store(sputnik::Load(inner_vector_in), inner_vector_in_tile);
        inner_vector_in_tile += kBlockDimX;
        inner_vector_in += kBlockDimX;
      }
    }

    // Step 3: Load the x-dimension residue region.
    //
    // Each thread should set 1 width predicate here and set
    // a new height predicate every 4 (or greater)
    typedef ThreadBlock<kBlockDimX, kBlockDimY> ThreadBlock;
    constexpr int kResidueX = kInputTileX % (kBlockDimX * kVectorWidth);
    constexpr int kHaloBlockDimX = kResidueX / kVectorWidth;
    const auto kThreadIdx = ThreadBlock::template Reshape<kHaloBlockDimX>();

    // Offset the pointers to the halo region.
    constexpr int kResidueOffset = kInputTileX - kResidueX;
    const int kBaseHaloOffset = kAlignedBaseInputOffset + kResidueOffset;
    const int kHaloOffset =
        (kBaseHaloOffset + kThreadIdx.y * kW + kThreadIdx.x * kVectorWidth) *
        sizeof(float);
    const int kHaloTileOffset = (kResidueOffset + kThreadIdx.y * kInputTileX +
                                 kThreadIdx.x * kVectorWidth) *
                                sizeof(float);
    vector_in = OffsetCast<const MemOp>(in, kHaloOffset);
    vector_in_tile = OffsetCast<MemOp>(in_tile, kHaloTileOffset);

    constexpr int kHaloBlockDimY = kBlockDimY * kBlockDimX / kHaloBlockDimX;
    constexpr int kHaloStepsY = kInputTileY / kHaloBlockDimY;

#pragma unroll
    for (int y_item_idx = 0; y_item_idx < kHaloStepsY; ++y_item_idx) {
      Store(sputnik::Load(vector_in), vector_in_tile);
      vector_in += kHaloBlockDimY / kVectorWidth * kW;
      vector_in_tile += kHaloBlockDimY * kInputTileX / kVectorWidth;
    }

    // Step 4: Load the x-dimension residue residue.
    constexpr int kResidueXResidueY = kInputTileY % kHaloBlockDimY;
    if (kResidueXResidueY > 0 && kThreadIdx.y < kResidueXResidueY) {
      Store(sputnik::Load(vector_in), vector_in_tile);
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_INPUT_TILE_H_
