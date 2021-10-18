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

#ifndef THIRD_PARTY_SPUTNIK_DEPTHWISE_DEPTHWISE_CONFIG_H_
#define THIRD_PARTY_SPUTNIK_DEPTHWISE_DEPTHWISE_CONFIG_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {

template <typename MemOp_, int kKernelSize_, int kPadding_, int kStride_,
          int kBlockItemsX_, int kBlockItemsY_, int kThreadItemsX_,
          int kThreadItemsY_>
struct DepthwiseConfig {
  static constexpr int kKernelSize = kKernelSize_;
  static constexpr int kPadding = kPadding_;
  static constexpr int kStride = kStride_;
  static constexpr int kBlockItemsX = kBlockItemsX_;
  static constexpr int kBlockItemsY = kBlockItemsY_;
  static constexpr int kThreadItemsX = kThreadItemsX_;
  static constexpr int kThreadItemsY = kThreadItemsY_;

  // Thread block configuration.
  static constexpr int kBlockDimX = kBlockItemsX / kThreadItemsX;
  static constexpr int kBlockDimY = kBlockItemsY / kThreadItemsY;

  // Memory operation type.
  typedef MemOp_ MemOp;
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_DEPTHWISE_DEPTHWISE_CONFIG_H_
