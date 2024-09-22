/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <ck/utility/type_convert.hpp>

namespace ck {

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
__device__ void inner_product<float2_t, float2_t, float>(
    const float2_t& a,
    const float2_t& b,
    float& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};

  inner_product(
      vector_type<float, 2>{a}.AsType<float>()[I0],
      vector_type<float, 2>{b}.AsType<float>()[I0],
      c);

  inner_product(
      vector_type<float, 2>{a}.AsType<float>()[I1],
      vector_type<float, 2>{b}.AsType<float>()[I1],
      c);
}

template <>
__device__ void inner_product<float4_t, float4_t, float>(
    const float4_t& a,
    const float4_t& b,
    float& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};
  constexpr auto I2 = Number<2>{};
  constexpr auto I3 = Number<3>{};

  inner_product(
      vector_type<float, 4>{a}.AsType<float>()[I0],
      vector_type<float, 4>{b}.AsType<float>()[I0],
      c);

  inner_product(
      vector_type<float, 4>{a}.AsType<float>()[I1],
      vector_type<float, 4>{b}.AsType<float>()[I1],
      c);

  inner_product(
      vector_type<float, 4>{a}.AsType<float>()[I2],
      vector_type<float, 4>{b}.AsType<float>()[I2],
      c);

  inner_product(
      vector_type<float, 4>{a}.AsType<float>()[I3],
      vector_type<float, 4>{b}.AsType<float>()[I3],
      c);
}

template <>
__device__ void inner_product<bhalf_t, bhalf_t, float>(
    const bhalf_t& a,
    const bhalf_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
__device__ void inner_product<half_t, half_t, float>(
    const half_t& a,
    const half_t& b,
    float& c) {
  inner_product(type_convert<float>(a), type_convert<float>(b), c);
}

template <>
__device__ void inner_product<half2_t, half2_t, float>(
    const half2_t& a,
    const half2_t& b,
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
  const vector_type<half_t, 2> a_vector{a};
  const vector_type<half_t, 2> b_vector{b};

  static_for<0, 2, 1>{}([&](auto i) {
    c += type_convert<float>(a_vector.AsType<half_t>()[i]) *
        type_convert<float>(b_vector.AsType<half_t>()[i]);
  });
#endif
}

template <>
__device__ void inner_product<half4_t, half4_t, float>(
    const half4_t& a,
    const half4_t& b,
    float& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};

  inner_product(
      vector_type<half_t, 4>{a}.AsType<half2_t>()[I0],
      vector_type<half_t, 4>{b}.AsType<half2_t>()[I0],
      c);

  inner_product(
      vector_type<half_t, 4>{a}.AsType<half2_t>()[I1],
      vector_type<half_t, 4>{b}.AsType<half2_t>()[I1],
      c);
}

template <>
__device__ void inner_product<half8_t, half8_t, float>(
    const half8_t& a,
    const half8_t& b,
    float& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};
  constexpr auto I2 = Number<2>{};
  constexpr auto I3 = Number<3>{};

  inner_product(
      vector_type<half_t, 8>{a}.AsType<half2_t>()[I0],
      vector_type<half_t, 8>{b}.AsType<half2_t>()[I0],
      c);

  inner_product(
      vector_type<half_t, 8>{a}.AsType<half2_t>()[I1],
      vector_type<half_t, 8>{b}.AsType<half2_t>()[I1],
      c);

  inner_product(
      vector_type<half_t, 8>{a}.AsType<half2_t>()[I2],
      vector_type<half_t, 8>{b}.AsType<half2_t>()[I2],
      c);

  inner_product(
      vector_type<half_t, 8>{a}.AsType<half2_t>()[I3],
      vector_type<half_t, 8>{b}.AsType<half2_t>()[I3],
      c);
}

template <>
__device__ void inner_product<bhalf2_t, bhalf2_t, float>(
    const bhalf2_t& a,
    const bhalf2_t& b,
    float& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};

  inner_product(
      vector_type<bhalf_t, 2>{a}.AsType<bhalf_t>()[I0],
      vector_type<bhalf_t, 2>{b}.AsType<bhalf_t>()[I0],
      c);

  inner_product(
      vector_type<bhalf_t, 2>{a}.AsType<bhalf_t>()[I1],
      vector_type<bhalf_t, 2>{b}.AsType<bhalf_t>()[I1],
      c);
}

template <>
__device__ void inner_product<bhalf4_t, bhalf4_t, float>(
    const bhalf4_t& a,
    const bhalf4_t& b,
    float& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};
  constexpr auto I2 = Number<2>{};
  constexpr auto I3 = Number<3>{};

  inner_product(
      vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I0],
      vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I0],
      c);

  inner_product(
      vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I1],
      vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I1],
      c);

  inner_product(
      vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I2],
      vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I2],
      c);

  inner_product(
      vector_type<bhalf_t, 4>{a}.AsType<bhalf_t>()[I3],
      vector_type<bhalf_t, 4>{b}.AsType<bhalf_t>()[I3],
      c);
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
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};

  inner_product(
      vector_type<int8_t, 2>{a}.AsType<int8_t>()[I0],
      vector_type<int8_t, 2>{b}.AsType<int8_t>()[I0],
      c);

  inner_product(
      vector_type<int8_t, 2>{a}.AsType<int8_t>()[I1],
      vector_type<int8_t, 2>{b}.AsType<int8_t>()[I1],
      c);
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
  const vector_type<int8_t, 4> a_vector{a};
  const vector_type<int8_t, 4> b_vector{b};

  static_for<0, 4, 1>{}([&](auto i) {
    c += type_convert<int32_t>(a_vector.AsType<int8_t>()[i]) *
        type_convert<int32_t>(b_vector.AsType<int8_t>()[i]);
  });
#endif
}

template <>
__device__ void inner_product<int8x8_t, int8x8_t, int32_t>(
    const int8x8_t& a,
    const int8x8_t& b,
    int32_t& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};

  inner_product(
      vector_type<int8_t, 8>{a}.AsType<int8x4_t>()[I0],
      vector_type<int8_t, 8>{b}.AsType<int8x4_t>()[I0],
      c);

  inner_product(
      vector_type<int8_t, 8>{a}.AsType<int8x4_t>()[I1],
      vector_type<int8_t, 8>{b}.AsType<int8x4_t>()[I1],
      c);
}

template <>
__device__ void inner_product<int8x16_t, int8x16_t, int32_t>(
    const int8x16_t& a,
    const int8x16_t& b,
    int32_t& c) {
  constexpr auto I0 = Number<0>{};
  constexpr auto I1 = Number<1>{};
  constexpr auto I2 = Number<2>{};
  constexpr auto I3 = Number<3>{};

  inner_product(
      vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I0],
      vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I0],
      c);

  inner_product(
      vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I1],
      vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I1],
      c);

  inner_product(
      vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I2],
      vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I2],
      c);

  inner_product(
      vector_type<int8_t, 16>{a}.AsType<int8x4_t>()[I3],
      vector_type<int8_t, 16>{b}.AsType<int8x4_t>()[I3],
      c);
}

} // namespace ck
