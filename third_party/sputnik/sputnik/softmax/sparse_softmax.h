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

#ifndef THIRD_PARTY_SPUTNIK_SOFTMAX_SPARSE_SOFTMAX_H_
#define THIRD_PARTY_SPUTNIK_SOFTMAX_SPARSE_SOFTMAX_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {

/**
 * @brief Computes softmax function across the last dim of a sparse
 * matrix.
 */
cudaError_t SparseSoftmax(int m, int n, int nonzeros,
                          const float* __restrict__ values,
                          const int* __restrict__ row_indices,
                          const int* __restrict__ row_offsets,
                          const int* __restrict__ column_indices,
                          float* __restrict__ output_values,
                          cudaStream_t stream);

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SOFTMAX_SPARSE_SOFTMAX_H_
