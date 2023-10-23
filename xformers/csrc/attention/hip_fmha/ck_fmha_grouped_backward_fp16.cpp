#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_backward.h"
#include "ck_bool_switch.h"

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true,
    true>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true,
    false>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false,
    true>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false,
    false>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    true>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true,
    false>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false,
    true>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false,
    false>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true,
    true>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true,
    false>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false,
    true>;

extern template struct grouped_backward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false,
    false>;

void grouped_backward_fp16(GroupedBackwardParams& param, hipStream_t stream) {
  BOOL_SWITCH_2(
      param.has_attn_bias,
      HAS_ATTN_BIAS,
      param.use_fp32_qkv_grad,
      USE_FP32_QKV_GRAD,
      [&] {
        if (param.custom_mask_type == 0) {
          grouped_backward_masktype_attnbias_dispatched<
              ck::half_t,
              0,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>::Run(param, stream);
        } else if (param.custom_mask_type == 1) {
          grouped_backward_masktype_attnbias_dispatched<
              ck::half_t,
              1,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>::Run(param, stream);
        } else if (param.custom_mask_type == 2) {
          grouped_backward_masktype_attnbias_dispatched<
              ck::half_t,
              2,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>::Run(param, stream);
        } else
          throw std::runtime_error("Invalid custom_mask_type value");
      });
};
