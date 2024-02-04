/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_bool_switch.h"
#include "ck_fmha_grouped_backward.h"

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true,
    true>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true,
    false>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false,
    true>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false,
    false>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    true>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    false>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false,
    true>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false,
    false>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true,
    true>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true,
    false>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false,
    true>(GroupedBackwardParams& param, hipStream_t stream);

extern template void run_grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false,
    false>(GroupedBackwardParams& param, hipStream_t stream);

void grouped_backward_fp16(GroupedBackwardParams& param, hipStream_t stream) {
  BOOL_SWITCH_2(
      param.has_attn_bias,
      HAS_ATTN_BIAS,
      param.use_fp32_qkv_grad,
      USE_FP32_QKV_GRAD,
      [&] {
        if (param.custom_mask_type == 0) {
          run_grouped_backward_masktype_attnbias_dispatched<
              ck::half_t,
              0,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>(param, stream);
        } else if (param.custom_mask_type == 1) {
          run_grouped_backward_masktype_attnbias_dispatched<
              ck::half_t,
              1,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>(param, stream);
        } else if (param.custom_mask_type == 2) {
          run_grouped_backward_masktype_attnbias_dispatched<
              ck::half_t,
              2,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>(param, stream);
        } else
          throw std::runtime_error("Invalid custom_mask_type value");
      });
};
