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

#ifndef THIRD_PARTY_SPUTNIK_SPMM_PREDICATE_UTILS_H_
#define THIRD_PARTY_SPUTNIK_SPMM_PREDICATE_UTILS_H_

/**
 * @file @brief Utilities for storing and computing boolean predicates.
 */

namespace sputnik {

/**
 * @brief A vector of boolean predicates packed into registers.
 *
 * TODO(tgale): The predicate usage in these kernels is currently a
 * bit wonky and these bit-packed predicates get compiled to simple
 * branches instead of predicated ptx. This code could also be
 * simplified by just using a dynamic loop, although this would add
 * and index decrement in the inner loop of our kernel. We could also
 * handle these loads (and others) with Duff's device. None of this
 * is likely to impact performance significantly, but it could get
 * rid of a few extra instructions and make the code cleaner.
 */
template <int kPredicates, int kPredicatesPerByte = 4>
class PredicateVector {
 public:
  static_assert(kPredicatesPerByte <= 8,
                "Can't pack more than 8 predicates into a byte.");

  //
  /// Static members.
  //

  // Use 32-bit unsigned integers to store predicates.
  typedef uint32_t PredicateStorage;

  // The number of bytes we need to store the predicates.
  static constexpr int kBytes_ =
      (kPredicates + kPredicatesPerByte - 1) / kPredicatesPerByte;

  // The number of words we need to store the predicates.
  static constexpr int kWords_ =
      (kBytes_ + sizeof(PredicateStorage) - 1) / sizeof(PredicateStorage);

  //
  /// Member variables.
  //

  // Register storage for the predicates.
  PredicateStorage predicates_[kWords_];

  /**
   * @brief Constructor. Initialize all predicate bits to 1.
   */
  __device__ __forceinline__ PredicateVector() {
#pragma unroll
    for (int i = 0; i < kWords_; ++i) {
      predicates_[i] = 0xffffffff;
    }
  }

  /**
   * @brief Set the bit at the specified location to zero.
   */
  __device__ __forceinline__ void DisableBit(int idx) {
    int word, bit;
    GetWordAndBitOffsets(idx, &word, &bit);
    // NOTE: It could be worth looking into using bit-field insert
    // inline assembly for these operations.
    predicates_[word] &= ~(1 << bit);
  }

  /**
   * @brief Get the bit at the specified location.
   */
  __device__ __forceinline__ bool GetBit(int idx) const {
    int word, bit;
    GetWordAndBitOffsets(idx, &word, &bit);
    // NOTE: It could be worth looking into using bit-field extract
    // inline assembly for these operations.
    return (predicates_[word] >> bit) & 1;
  }

 private:
  /**
   * @brief Convert an index to word and byte offsets for setting and
   * extracting the underlying predicate.
   */
  __device__ __forceinline__ void GetWordAndBitOffsets(int idx, int *word,
                                                       int *bit) const {
    // NOTE: Indices to this function should be statically known s.t.
    // the following indexing math can be evaluated during compilation.
    //
    // TODO(tgale): Figure out a way to force the compiler to enforce
    // that these are statically known. Using constexpr here causes the
    // compiler to complain, even though all inputs are statically known
    // indices of unrolled loops.
    const int kWordOffset =
        (idx / kPredicatesPerByte) / sizeof(PredicateStorage);
    const int kByteOffset =
        (idx / kPredicatesPerByte) % sizeof(PredicateStorage);
    const int kBitOffset =
        (idx % kPredicatesPerByte) % sizeof(PredicateStorage);

    // TODO(tgale): Following cutlass, we store predicates in the first four
    // bits of each byte. It's not totally clear why we do this versus using
    // all the bits or spread out the predicates every-other bit.
    *word = kWordOffset;
    *bit = kByteOffset * 8 + kBitOffset;
  }
};

/**
 * @brief Setter for n-dimension load and store predicates.
 */
template <typename LoadType, int kBlockItemsX, int kBlockWidth>
struct PredicatesN {
  //
  /// Static members.
  //

  // The number of values in every load/store with of LoadType.
  static constexpr int kValuesPerItem_ = sizeof(LoadType) / sizeof(float);

  // The number of items in the n-dimension each thread is responsbile for.
  static constexpr int kThreadItemsX_ =
      kBlockItemsX / kBlockWidth / kValuesPerItem_;

  // The number of values we increment by after each load.
  static constexpr int increment_x_ = kBlockWidth * kValuesPerItem_;

  //
  /// Member functions.
  //

  // Shorthand for n-dim predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsX_> Predicates;

  // Default constructor.
  __device__ __forceinline__ PredicatesN() {}

  /**
   * @brief Set predicates for this threads loads in the n-dimension.
   *
   * When loading/storing along the n-dimension of the problem we need
   * to avoid going out of bounds if the problem dimensions don't divide
   * evenly by the tile dimensions. This function sets the appropriate
   * predicates to avoid out-of-bounds memory accesses.
   *
   * @param n_idx The column index marking the start of the 1-dimensional
   * tile that this thread collaborates to compute.
   * @param n The number of columns in the dense rhs and output matrices.
   * @param predicates Pointer to a vector of predicates that we'll store
   * the computed predicates in.
   */
  static __device__ __forceinline__ void Set(int n_idx, int n,
                                             Predicates *predicates) {
    int index = n_idx + threadIdx.x * kValuesPerItem_;

#pragma unroll
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
      if (index >= n) {
        predicates->DisableBit(x_item_idx);
      }
      index += increment_x_;
    }
  }
};

/**
 * @brief Setter for k-dimension load predicates on the final spmm
 * main-loop iteration.
 */
template <typename LoadType, int kBlockItemsK, int kBlockWidth>
struct PredicatesK {
  //
  /// Static members.
  //

  // The number of values in every load/store with of LoadType.
  static constexpr int kValuesPerItem_ = sizeof(LoadType) / sizeof(float);

  // The number of items in the n-dimension each thread is responsbile for.
  static constexpr int kThreadItemsK_ =
      kBlockItemsK / kBlockWidth / kValuesPerItem_;

  // The number of values we increment by after each load.
  static constexpr int increment_k_ = kBlockWidth * kValuesPerItem_;

  //
  /// Member functions.
  //

  // Shorthand for a predicate vector of the appropriate size.
  typedef PredicateVector<kThreadItemsK_> Predicates;

  // Default constructor.
  __device__ __forceinline__ PredicatesK() {}

  /**
   * @brief Set predicates for this threads loads in the k-dimension.
   *
   * When loading along the k-dimension of the problem we need to avoid
   * going out of bounds if the problem dimensions don't divide evenly
   * by the tile dimensions. This function sets the appropriate predicates
   * to avoid out-of-bounds memory accesses.
   *
   * @param residue The number of values left to load along the k-dimension
   * after we've computed the maximum number of full tiles possible.
   * @param predicates Pointer to a vector of predicates that we'll store
   * the computed predicates in.
   */
  static __device__ __forceinline__ void Set(int residue,
                                             Predicates *predicates) {
    int index = threadIdx.x * kValuesPerItem_;
#pragma unroll
    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; ++k_item_idx) {
      if (index >= residue) {
        predicates->DisableBit(k_item_idx);
      }
      index += increment_k_;
    }
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SPMM_PREDICATE_UTILS_H_
