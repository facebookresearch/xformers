/*
 * Copyright (c) 2023-2025, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

template <typename TA, typename TB, typename TC>
CK_TILE_DEVICE void inner_product(const TA& a, const TB& b, TC& c);

template <typename TA, typename TB, typename TC, typename TItem>
CK_TILE_DEVICE void inner_product_unrolled(const TA& a, const TB& b, TC& c) {
  static_assert(std::is_same_v<TA, TB>);
  constexpr int unroll_count = sizeof(TA) / sizeof(TItem);
  using item_array_t = TItem[unroll_count];
  auto av = *reinterpret_cast<const item_array_t*>(&a);
  auto bv = *reinterpret_cast<const item_array_t*>(&b);
#pragma unroll
  for (int i = 0; i < unroll_count; ++i) {
    inner_product(av[i], bv[i], c);
  }
}

template <>
CK_TILE_DEVICE void inner_product<float, float, float>(
    const float& a,
    const float& b,
    float& c) {
#if (defined(__gfx9__)) // for GPU code
  asm volatile(
      "\n \
            v_fmac_f32 %0, %1, %2 \n \
            "
      : "=v"(c)
      : "v"(a), "v"(b), "0"(c));
#else
  c += a * b;
#endif
}

template <>
CK_TILE_DEVICE void inner_product<fp32x2_t, fp32x2_t, float>(
    const fp32x2_t& a,
    const fp32x2_t& b,
    float& c) {
  inner_product_unrolled<fp32x2_t, fp32x2_t, float, fp32_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<fp32x4_t, fp32x4_t, float>(
    const fp32x4_t& a,
    const fp32x4_t& b,
    float& c) {
  inner_product_unrolled<fp32x4_t, fp32x4_t, float, fp32_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<bf16_t, bf16_t, float>(
    const bf16_t& a,
    const bf16_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
CK_TILE_DEVICE void inner_product<fp16_t, fp16_t, float>(
    const fp16_t& a,
    const fp16_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
CK_TILE_DEVICE void inner_product<fp16x2_t, fp16x2_t, float>(
    const fp16x2_t& a,
    const fp16x2_t& b,
    float& c) {
#if (defined(__gfx9__)) // for GPU code
  c = __builtin_amdgcn_fdot2(a, b, c, false);
#else
  inner_product_unrolled<fp16x2_t, fp16x2_t, float, fp16_t>(a, b, c);
#endif
}

template <>
CK_TILE_DEVICE void inner_product<fp16x4_t, fp16x4_t, float>(
    const fp16x4_t& a,
    const fp16x4_t& b,
    float& c) {
  inner_product_unrolled<fp16x4_t, fp16x4_t, float, fp16x2_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<fp16x8_t, fp16x8_t, float>(
    const fp16x8_t& a,
    const fp16x8_t& b,
    float& c) {
  inner_product_unrolled<fp16x8_t, fp16x8_t, float, fp16x2_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<bf16x2_t, bf16x2_t, float>(
    const bf16x2_t& a,
    const bf16x2_t& b,
    float& c) {
  inner_product_unrolled<bf16x2_t, bf16x2_t, float, bf16_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<bf16x4_t, bf16x4_t, float>(
    const bf16x4_t& a,
    const bf16x4_t& b,
    float& c) {
  inner_product_unrolled<bf16x4_t, bf16x4_t, float, bf16_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<int8_t, int8_t, int32_t>(
    const int8_t& a,
    const int8_t& b,
    int32_t& c) {
  c += type_convert<int32_t>(a) * type_convert<int32_t>(b);
}

template <>
CK_TILE_DEVICE void inner_product<int8x2_t, int8x2_t, int32_t>(
    const int8x2_t& a,
    const int8x2_t& b,
    int32_t& c) {
  inner_product_unrolled<int8x2_t, int8x2_t, int32_t, int8_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<int8x4_t, int8x4_t, int32_t>(
    const int8x4_t& a,
    const int8x4_t& b,
    int32_t& c) {
#if (defined(__gfx9__)) // for GPU code
  c = __builtin_amdgcn_sdot4(
      bit_cast<int32_t>(a), bit_cast<int32_t>(b), c, false);
#else
  inner_product_unrolled<int8x4_t, int8x4_t, int32_t, int8_t>(a, b, c);
#endif
}

template <>
CK_TILE_DEVICE void inner_product<int8x8_t, int8x8_t, int32_t>(
    const int8x8_t& a,
    const int8x8_t& b,
    int32_t& c) {
  inner_product_unrolled<int8x8_t, int8x8_t, int32_t, int8x4_t>(a, b, c);
}

template <>
CK_TILE_DEVICE void inner_product<int8x16_t, int8x16_t, int32_t>(
    const int8x16_t& a,
    const int8x16_t& b,
    int32_t& c) {
  inner_product_unrolled<int8x16_t, int8x16_t, int32_t, int8x4_t>(a, b, c);
}

// TBD: Packed I4

} // namespace ck_tile
