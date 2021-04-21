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

#ifndef THIRD_PARTY_SPUTNIK_SDDMM_CUDA_SDDMM_H_
#define THIRD_PARTY_SPUTNIK_SDDMM_CUDA_SDDMM_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {

/**
 * @brief Compute a sampled dense-dense matrix product.
 */
cudaError_t CudaSddmm(int m, int k, int n, int nonzeros,
                      const int* __restrict__ row_indices,
                      const int* __restrict__ row_offsets,
                      const int* __restrict__ column_indices,
                      const float* __restrict__ lhs_matrix,
                      const float* __restrict__ rhs_matrix,
                      float* __restrict__ output_values,
                      cudaStream_t stream);

/**
 * @brief Compute a sampled dense-dense matrix product.
 */
template <typename LoadType, int kBlockItemsY, int kBlockItemsK,
          int kBlockItemsX, int kBlockWidth, int kPredicateK = true>
cudaError_t CudaSddmmEx(int m, int k, int n, int nonzeros,
                        const int* __restrict__ row_indices,
                        const int* __restrict__ row_offsets,
                        const int* __restrict__ column_indices,
                        const float* __restrict__ lhs_matrix,
                        const float* __restrict__ rhs_matrix,
                        float* __restrict__ output_values,
                        cudaStream_t stream);

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_SDDMM_CUDA_SDDMM_H_
