#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_forward.h"
#include "ck_static_switch.h"

extern template struct grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true>;

extern template struct grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false>;

extern template struct grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true>;

extern template struct grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false>;

extern template struct grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true>;

extern template struct grouped_forward_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false>;

void grouped_forward_fp16(GroupedForwardParams& param, hipStream_t stream) {
  BOOL_SWITCH_1(param.has_attn_bias, HAS_ATTN_BIAS, [&] {
    if (param.custom_mask_type == 0)
      grouped_forward_masktype_attnbias_dispatched<ck::half_t, 0, true>::Run(
          param, stream);
    else if (param.custom_mask_type == 1)
      grouped_forward_masktype_attnbias_dispatched<ck::half_t, 1, true>::Run(
          param, stream);
    else if (param.custom_mask_type == 2)
      grouped_forward_masktype_attnbias_dispatched<ck::half_t, 2, true>::Run(
          param, stream);
    else
      throw std::runtime_error("Invalid custom_mask_type value");
  });
};
