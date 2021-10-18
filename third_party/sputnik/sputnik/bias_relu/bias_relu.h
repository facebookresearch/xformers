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

#ifndef THIRD_PARTY_SPUTNIK_BIAS_RELU_BIAS_RELU_H_
#define THIRD_PARTY_SPUTNIK_BIAS_RELU_BIAS_RELU_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {

/**
 * @brief Add bias broadcast over channels and perform relu activation.
 *
 * For NCHW layout.
 */
cudaError_t BiasRelu(int n, int c, int d, const float* __restrict__ in,
                     const float* __restrict__ bias, float* __restrict__ out,
                     cudaStream_t stream);

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BIAS_RELU_BIAS_RELU_H_
