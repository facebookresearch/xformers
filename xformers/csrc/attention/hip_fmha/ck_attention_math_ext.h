/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/utility/data_type.hpp>
#include <ck/utility/type_convert.hpp>

namespace ck {
namespace math {
template <typename T>
inline __device__ T exp(T x) {
  return ck::type_convert<T>(__expf(ck::type_convert<float>(x)));
};

template <>
inline __device__ float exp<float>(float x) {
  return __expf(x);
};

template <>
inline __device__ double exp<double>(double x) {
  return exp(x);
};
} // namespace math
} // namespace ck
