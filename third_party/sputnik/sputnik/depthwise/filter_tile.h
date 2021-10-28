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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_FILTER_TILE_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_FILTER_TILE_H_

#include "sputnik/common.h"
#include "sputnik/load_store.h"
#include "sputnik/tiling_utils.h"

namespace sputnik {

template <int kKernelSize, int kBlockDimX>
struct FilterTile {
  //
  /// Static members.
  //

  // The number of weights in each filter.
  static constexpr int kNumWeights = kKernelSize * kKernelSize;

  //
  /// Member variables.
  //

  const float* filters;
  float* smem;
  float* filter_fragment;

  __device__ __forceinline__ FilterTile(int filter_offset,
                                        const float* __restrict__ filters_,
                                        float* __restrict__ smem_,
                                        float* __restrict__ filter_fragment_)
      : filters(filters_ + filter_offset),
        smem(smem_),
        filter_fragment(filter_fragment_) {}

  __device__ __forceinline__ void Load() {
    // TODO(tgale): This should always be true. We could get rid of
    // this an just have the compiler unroll a loop.
    const int thread_idx = threadIdx.x + threadIdx.y * kBlockDimX;
    if (thread_idx < kNumWeights) {
      Store(sputnik::Load(filters + thread_idx), smem + thread_idx);
    }
    __syncthreads();

    // Load all of the weights. We use 4-wide vector loads. This
    // loads a few more values than we need, but it doesn't matter
    // and the register allocator should just reuse those registers
    // right after we store to them.
    constexpr int kValuesPerLoad = 4;
    constexpr int kSharedFilterLoads =
        RoundUpTo(kNumWeights, kValuesPerLoad) / kValuesPerLoad;
    auto vector_smem = OffsetCast<const float4>(smem, 0);
    auto vector_filter_fragment = OffsetCast<float4>(filter_fragment, 0);
#pragma unroll
    for (int load_idx = 0; load_idx < kSharedFilterLoads; ++load_idx) {
      Store(vector_smem[load_idx], vector_filter_fragment);
      vector_filter_fragment++;
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_FILTER_TILE_H_
