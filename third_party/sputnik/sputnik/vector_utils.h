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

#ifndef THIRD_PARTY_SPUTNIK_VECTOR_UTILS_H_
#define THIRD_PARTY_SPUTNIK_VECTOR_UTILS_H_

/**
 * @file @brief Utilities for working with CUDA vector data types.
 */

#include "sputnik/cuda_utils.h"
#include "sputnik/type_utils.h"

namespace sputnik {

/**
 * @brief Functor for computing FMAs & MULs on mixes of vector and scalar
 * data types.
 */
template <typename Value>
struct VectorCompute {
  typedef typename TypeUtils<Value>::Accumulator Accumulator;

  static __device__ __forceinline__ void FMA(float x1, Value x2,
                                             Accumulator *out);

  // Complementary index type to our load type.
  typedef typename Value2Index<Value>::Index Index;

  static __device__ __forceinline__ void Mul(int, Index x2, Index *out);

  static __device__ __forceinline__ void Dot(Value x1, Value x2,
                                             Accumulator *out);
};

template <>
struct VectorCompute<float> {
  static __device__ __forceinline__ void FMA(float x1, float x2, float *out) {
    out[0] += x1 * x2;
  }

  static __device__ __forceinline__ void Mul(int x1, int x2, int *out) {
    out[0] = x1 * x2;
  }

  static __device__ __forceinline__ void Dot(float x1, float x2, float *out) {
    out[0] += x1 * x2;
  }
};

template <>
struct VectorCompute<float2> {
  static __device__ __forceinline__ void FMA(float x1, float2 x2, float2 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
  }

  static __device__ __forceinline__ void Mul(int x1, int2 x2, int2 *out) {
    out[0].x = x1 * x2.x;
    out[0].y = x1 * x2.y;
  }

  static __device__ __forceinline__ void Dot(float2 x1, float2 x2, float *out) {
    out[0] += x1.x * x2.x;
    out[0] += x1.y * x2.y;
  }
};

template <>
struct VectorCompute<float4> {
  static __device__ __forceinline__ void FMA(float x1, float4 x2, float4 *out) {
    out[0].x += x1 * x2.x;
    out[0].y += x1 * x2.y;
    out[0].z += x1 * x2.z;
    out[0].w += x1 * x2.w;
  }

  static __device__ __forceinline__ void Mul(int x1, int4 x2, int4 *out) {
    out[0].x = x1 * x2.x;
    out[0].y = x1 * x2.y;
    out[0].z = x1 * x2.z;
    out[0].w = x1 * x2.w;
  }

  static __device__ __forceinline__ void Dot(float4 x1, float4 x2, float *out) {
    out[0] += x1.x * x2.x;
    out[0] += x1.y * x2.y;
    out[0] += x1.z * x2.z;
    out[0] += x1.w * x2.w;
  }
};

template <>
struct VectorCompute<half2> {
  static __device__ __forceinline__ void FMA(float x1, half2 x2, float2 *out) {
    float2 x2_f2 = __half22float2(x2);
    VectorCompute<float2>::FMA(x1, x2_f2, out);
  }

  static __device__ __forceinline__ void Mul(int x1, short2 x2, short2 *out) {
    out[0].x = static_cast<short>(x1 * x2.x);
    out[0].y = static_cast<short>(x1 * x2.y);
  }
};

template <>
struct VectorCompute<half4> {
  static __device__ __forceinline__ void FMA(float x1, half4 x2, float4 *out) {
    float2 x2x_f2 = __half22float2(x2.x);
    float2 x2y_f2 = __half22float2(x2.y);
    float4 x2_f4 = make_float4(x2x_f2.x, x2x_f2.y, x2y_f2.x, x2y_f2.y);
    VectorCompute<float4>::FMA(x1, x2_f4, out);
  }

  static __device__ __forceinline__ void Mul(int x1, short4 x2, short4 *out) {
    VectorCompute<half2>::Mul(x1, x2.x, &out[0].x);
    VectorCompute<half2>::Mul(x1, x2.y, &out[0].y);
  }
};

template <>
struct VectorCompute<half8> {
  static __device__ __forceinline__ void FMA(float x1, half8 x2, float4 *out) {
    half4 x2x_h4;
    x2x_h4.x = x2.x;
    x2x_h4.y = x2.y;
    VectorCompute<half4>::FMA(x1, x2x_h4, out);
    half4 x2y_h4;
    x2y_h4.x = x2.z;
    x2y_h4.y = x2.w;
    VectorCompute<half4>::FMA(x1, x2y_h4, out + 1);
  }

  static __device__ __forceinline__ void Mul(int x1, short8 x2, short8 *out) {
    VectorCompute<half2>::Mul(x1, x2.x, &out[0].x);
    VectorCompute<half2>::Mul(x1, x2.y, &out[0].y);
    VectorCompute<half2>::Mul(x1, x2.z, &out[0].z);
    VectorCompute<half2>::Mul(x1, x2.w, &out[0].w);
  }
};

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_VECTOR_UTILS_H_
