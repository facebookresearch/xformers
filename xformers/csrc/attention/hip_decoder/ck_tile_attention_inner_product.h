/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>

namespace ck_tile {

template <typename TA, typename TB, typename TC>
__device__ void inner_product(const TA& a, const TB& b, TC& c);

template <>
__device__ void inner_product<float, float, float>(
    const float& a,
    const float& b,
    float& c) {
#if CK_USE_AMD_V_MAC_INLINE_ASM && defined(CK_USE_AMD_V_MAC_F32)
  asm volatile(
      "\n \
            v_mac_f32 %0, %1, %2 \n \
            "
      : "=v"(c)
      : "v"(a), "v"(b), "0"(c));
#elif CK_USE_AMD_V_MAC_INLINE_ASM && defined(CK_USE_AMD_V_FMAC_F32)
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
__device__ void inner_product<fp32x2_t, fp32x2_t, float>(
    const fp32x2_t& a,
    const fp32x2_t& b,
    float& c) {
  inner_product(a[0], b[0], c);
  inner_product(a[1], b[1], c);
}

template <>
__device__ void inner_product<fp32x4_t, fp32x4_t, float>(
    const fp32x4_t& a,
    const fp32x4_t& b,
    float& c) {
  inner_product(a[0], b[0], c);
  inner_product(a[1], b[1], c);
  inner_product(a[2], b[2], c);
  inner_product(a[3], b[3], c);
}

template <>
__device__ void inner_product<bf16_t, bf16_t, float>(
    const bf16_t& a,
    const bf16_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
__device__ void inner_product<fp16_t, fp16_t, float>(
    const fp16_t& a,
    const fp16_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
__device__ void inner_product<fp16x2_t, fp16x2_t, float>(
    const fp16x2_t& a,
    const fp16x2_t& b,
    float& c) {
#if defined(CK_USE_AMD_V_DOT2_F32_F16)
#if CK_USE_AMD_V_DOT_INLINE_ASM
  // Use 3 x s_nop to avoid hazard (mi200 cdna2 isa page 47
  // https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf
  // ) s_nop with parameter 2 is equal to 3 x s_nop
  asm volatile(
      "\n \
            v_dot2_f32_f16 %0, %1, %2, %0\n \
            s_nop 2 \n \
            "
      : "=v"(c)
      : "v"(a), "v"(b), "0"(c));
#else
  c = __builtin_amdgcn_fdot2(a, b, c, false);
#endif
#else
  c += type_convert<float>(a[0]) * type_convert<float>(b[0]);
  c += type_convert<float>(a[1]) * type_convert<float>(b[1]);
#endif
}

template <>
__device__ void inner_product<fp16x4_t, fp16x4_t, float>(
    const fp16x4_t& a,
    const fp16x4_t& b,
    float& c) {
  c += type_convert<float>(a[0]) * type_convert<float>(b[0]);
  c += type_convert<float>(a[1]) * type_convert<float>(b[1]);
  c += type_convert<float>(a[2]) * type_convert<float>(b[2]);
  c += type_convert<float>(a[3]) * type_convert<float>(b[3]);
}

template <>
__device__ void inner_product<fp16x8_t, fp16x8_t, float>(
    const fp16x8_t& a,
    const fp16x8_t& b,
    float& c) {
  c += type_convert<float>(a[0]) * type_convert<float>(b[0]);
  c += type_convert<float>(a[1]) * type_convert<float>(b[1]);
  c += type_convert<float>(a[2]) * type_convert<float>(b[2]);
  c += type_convert<float>(a[3]) * type_convert<float>(b[3]);
  c += type_convert<float>(a[4]) * type_convert<float>(b[4]);
  c += type_convert<float>(a[5]) * type_convert<float>(b[5]);
  c += type_convert<float>(a[6]) * type_convert<float>(b[6]);
  c += type_convert<float>(a[7]) * type_convert<float>(b[7]);
}

template <>
__device__ void inner_product<bf16x2_t, bf16x2_t, float>(
    const bf16x2_t& a,
    const bf16x2_t& b,
    float& c) {
  c += type_convert<float>(a[0]) * type_convert<float>(b[0]);
  c += type_convert<float>(a[1]) * type_convert<float>(b[1]);
}

template <>
__device__ void inner_product<bf16x4_t, bf16x4_t, float>(
    const bf16x4_t& a,
    const bf16x4_t& b,
    float& c) {
  c += type_convert<float>(a[0]) * type_convert<float>(b[0]);
  c += type_convert<float>(a[1]) * type_convert<float>(b[1]);
  c += type_convert<float>(a[2]) * type_convert<float>(b[2]);
  c += type_convert<float>(a[3]) * type_convert<float>(b[3]);
}

template <>
__device__ void inner_product<int8_t, int8_t, int32_t>(
    const int8_t& a,
    const int8_t& b,
    int32_t& c) {
  c += type_convert<int32_t>(a) * type_convert<int32_t>(b);
}

template <>
__device__ void inner_product<int8x2_t, int8x2_t, int32_t>(
    const int8x2_t& a,
    const int8x2_t& b,
    int32_t& c) {
  c += type_convert<int32_t>(a[0]) * type_convert<int32_t>(b[0]);
  c += type_convert<int32_t>(a[1]) * type_convert<int32_t>(b[1]);
}

template <>
__device__ void inner_product<int8x4_t, int8x4_t, int32_t>(
    const int8x4_t& a,
    const int8x4_t& b,
    int32_t& c) {
#if defined(CK_USE_AMD_V_DOT4_I32_I8)
#if CK_USE_AMD_V_DOT_INLINE_ASM
  // Use 3 x s_nop to avoid hazard (mi200 cdna2 isa page 47
  // https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf
  // ) s_nop with parameter 2 is equal to 3 x s_nop
  asm volatile(
      "\n \
            v_dot4_i32_i8 %0, %1, %2, %0\n \
            s_nop 2 \n \
            "
      : "=v"(c)
      : "v"(bit_cast<int32_t>(a)), "v"(bit_cast<int32_t>(b)), "0"(c));
#else
  c = __builtin_amdgcn_sdot4(
      bit_cast<int32_t>(a), bit_cast<int32_t>(b), c, false);
#endif
#elif defined(CK_USE_AMD_V_DOT4_I32_I8_GFX11)
  c = __builtin_amdgcn_sudot4(
      true, bit_cast<int32_t>(a), true, bit_cast<int32_t>(b), c, false);
#else
  c += type_convert<int32_t>(a[0]) * type_convert<int32_t>(b[0]);
  c += type_convert<int32_t>(a[1]) * type_convert<int32_t>(b[1]);
  c += type_convert<int32_t>(a[2]) * type_convert<int32_t>(b[2]);
  c += type_convert<int32_t>(a[3]) * type_convert<int32_t>(b[3]);
#endif
}

template <>
__device__ void inner_product<int8x8_t, int8x8_t, int32_t>(
    const int8x8_t& a,
    const int8x8_t& b,
    int32_t& c) {
  c += type_convert<int32_t>(a[0]) * type_convert<int32_t>(b[0]);
  c += type_convert<int32_t>(a[1]) * type_convert<int32_t>(b[1]);
  c += type_convert<int32_t>(a[2]) * type_convert<int32_t>(b[2]);
  c += type_convert<int32_t>(a[3]) * type_convert<int32_t>(b[3]);
  c += type_convert<int32_t>(a[4]) * type_convert<int32_t>(b[4]);
  c += type_convert<int32_t>(a[5]) * type_convert<int32_t>(b[5]);
  c += type_convert<int32_t>(a[6]) * type_convert<int32_t>(b[6]);
  c += type_convert<int32_t>(a[7]) * type_convert<int32_t>(b[7]);
}

template <>
__device__ void inner_product<int8x16_t, int8x16_t, int32_t>(
    const int8x16_t& a,
    const int8x16_t& b,
    int32_t& c) {
  c += type_convert<int32_t>(a[0]) * type_convert<int32_t>(b[0]);
  c += type_convert<int32_t>(a[1]) * type_convert<int32_t>(b[1]);
  c += type_convert<int32_t>(a[2]) * type_convert<int32_t>(b[2]);
  c += type_convert<int32_t>(a[3]) * type_convert<int32_t>(b[3]);
  c += type_convert<int32_t>(a[4]) * type_convert<int32_t>(b[4]);
  c += type_convert<int32_t>(a[5]) * type_convert<int32_t>(b[5]);
  c += type_convert<int32_t>(a[6]) * type_convert<int32_t>(b[6]);
  c += type_convert<int32_t>(a[7]) * type_convert<int32_t>(b[7]);
  c += type_convert<int32_t>(a[8]) * type_convert<int32_t>(b[8]);
  c += type_convert<int32_t>(a[9]) * type_convert<int32_t>(b[9]);
  c += type_convert<int32_t>(a[10]) * type_convert<int32_t>(b[10]);
  c += type_convert<int32_t>(a[11]) * type_convert<int32_t>(b[11]);
  c += type_convert<int32_t>(a[12]) * type_convert<int32_t>(b[12]);
  c += type_convert<int32_t>(a[13]) * type_convert<int32_t>(b[13]);
  c += type_convert<int32_t>(a[14]) * type_convert<int32_t>(b[14]);
  c += type_convert<int32_t>(a[15]) * type_convert<int32_t>(b[15]);
}

} // namespace ck_tile
