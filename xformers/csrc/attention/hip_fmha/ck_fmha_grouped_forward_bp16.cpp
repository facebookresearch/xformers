#include <ck/ck.hpp>
#include "ck_fmha_grouped_forward.h"

void grouped_forward_bp16(GroupedForwardParams& param, hipStream_t stream) {
  if (param.custom_mask_type == 0) {
    if (param.has_attn_bias)
      grouped_forward_masktype_attnbias_dispatched<ck::bhalf_t, 0, true>(
          param, stream);
    else
      grouped_forward_masktype_attnbias_dispatched<ck::bhalf_t, 0, false>(
          param, stream);
  } else if (param.custom_mask_type == 1) {
    if (param.has_attn_bias)
      grouped_forward_masktype_attnbias_dispatched<ck::bhalf_t, 1, true>(
          param, stream);
    else
      grouped_forward_masktype_attnbias_dispatched<ck::bhalf_t, 1, false>(
          param, stream);
  } else if (param.custom_mask_type == 2) {
    if (param.has_attn_bias)
      grouped_forward_masktype_attnbias_dispatched<ck::bhalf_t, 2, true>(
          param, stream);
    else
      grouped_forward_masktype_attnbias_dispatched<ck::bhalf_t, 2, false>(
          param, stream);
  } else
    throw std::runtime_error("Invalid custom_mask_type value");
};
