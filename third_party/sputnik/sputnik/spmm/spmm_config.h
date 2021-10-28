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

#ifndef THIRD_PARTY_SPUTNIK_SPMM_SPMM_CONFIG_H_
#define THIRD_PARTY_SPUTNIK_SPMM_SPMM_CONFIG_H_

/**
 * @file @brief Configuration helper for SpMM kernels.
 */

#include "sputnik/type_utils.h"
#include "sputnik/vector_utils.h"

namespace sputnik {

template <typename ScalarValue_,    // Scalar data type for all operands.
          typename SparseValue_,    // Vector data type for the sparse matrix.
          typename DenseValue_,     // Vector data type for the dense operands.
          int kBlockItemsY_,        // Tile size in the m-dimension.
          int kBlockItemsK_,        // Tile size in the k-dimension.
          int kBlockItemsX_,        // Tile size in the n-dimension.
          int kBlockWidth_,         // Threadblock width.
          int kResidueUnroll_ = 4,  // Number of unroll steps in the residue.
          int kPredicateLoads_ = true,  // Whether to predicate loads or not.
          bool kLaunchBounds_ = false,  // Whether or not to set launch bounds.
          int kMinOccupancy_ = 8>       // Minimum occupancy to target.
struct SpmmConfig {
  typedef ScalarValue_ ScalarValue;
  typedef SparseValue_ SparseValue;
  typedef DenseValue_ DenseValue;
  typedef typename Value2Index<SparseValue>::Index Index;
  typedef typename Value2Index<ScalarValue>::Index ScalarIndex;

  static constexpr int kBlockItemsY = kBlockItemsY_;
  static constexpr int kBlockItemsK = kBlockItemsK_;
  static constexpr int kBlockItemsX = kBlockItemsX_;
  static constexpr int kBlockWidth = kBlockWidth_;
  static constexpr int kResidueUnroll = kResidueUnroll_;
  static constexpr int kPredicateLoads = kPredicateLoads_;
  static constexpr bool kLaunchBounds = kLaunchBounds_;
  static constexpr int kMinOccupancy = kMinOccupancy_;
  static constexpr int kElementsPerScalar =
      TypeUtils<ScalarValue_>::kElementsPerScalar;

  // Sanity checks on the template arguments.
  static_assert((kBlockItemsY * kBlockWidth) % 32 == 0,
                "The thread-block size must be divisible by the warp size.");
  static_assert((kBlockItemsY * kBlockWidth) > 0,
                "The thread-block size must be nonzero.");
  static_assert(kBlockItemsK >= kBlockWidth,
                "k-dimension tile must be >= block width.");
  static_assert(kBlockItemsK % kBlockWidth == 0,
                "k-dimension tile size must be divisible by block width.");
  static_assert(kBlockItemsX >= kBlockWidth,
                "n-dimension tile size must be >= block width.");
  static_assert(kBlockItemsX % kBlockWidth == 0,
                "n-dimension tile size must be divisible by block width.");
  //
  /// Commonly used statically known values.
  //

  // The number of values in every load/store of type DenseValue.
  static constexpr int kValuesPerItemX =
      sizeof(DenseValue) / sizeof(ScalarValue);

  // The number of items in the n-dimension each thread is responsbile for.
  static constexpr int kThreadItemsX =
      kBlockItemsX / kBlockWidth / kValuesPerItemX;

  // The number of threads per threadblock.
  static constexpr int kThreadsPerBlock = kBlockItemsY * kBlockWidth;
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SPMM_SPMM_CONFIG_H_
