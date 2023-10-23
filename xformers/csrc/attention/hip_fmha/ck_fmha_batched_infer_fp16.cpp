#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_infer.h"
#include "ck_bool_switch.h"

extern template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    true>;

extern template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    0,
    false>;

extern template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    true>;

extern template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    1,
    false>;

extern template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    true>;

extern template struct batched_infer_masktype_attnbias_dispatched<
    ck::half_t,
    2,
    false>;

void batched_infer_fp16(BatchedForwardParams& param, hipStream_t stream) {
  BOOL_SWITCH_1(param.has_attn_bias, HAS_ATTN_BIAS, [&] {
    if (param.custom_mask_type == 0)
      batched_infer_masktype_attnbias_dispatched<ck::half_t, 0, HAS_ATTN_BIAS>::
          Run(param, stream);
    else if (param.custom_mask_type == 1)
      batched_infer_masktype_attnbias_dispatched<ck::half_t, 1, HAS_ATTN_BIAS>::
          Run(param, stream);
    else if (param.custom_mask_type == 2)
      batched_infer_masktype_attnbias_dispatched<ck::half_t, 2, HAS_ATTN_BIAS>::
          Run(param, stream);
    else
      throw std::runtime_error("Invalid custom_mask_type value");
  });
};
