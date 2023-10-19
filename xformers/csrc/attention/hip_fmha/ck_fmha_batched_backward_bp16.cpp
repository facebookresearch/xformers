#include <ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_backward.h"
#include "ck_static_switch.h"

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    true>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    true,
    false>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false,
    true>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    0,
    false,
    false>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    true,
    true>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    true,
    false>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    true>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    1,
    false,
    false>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    true,
    true>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    true,
    false>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    false,
    true>;

extern template struct batched_backward_masktype_attnbias_dispatched<
    ck::bhalf_t,
    2,
    false,
    false>;

void batched_backward_bp16(BatchedBackwardParams& param, hipStream_t stream) {
  BOOL_SWITCH_2(
      param.has_attn_bias,
      HAS_ATTN_BIAS,
      param.use_fp32_qkv_grad,
      USE_FP32_QKV_GRAD,
      [&] {
        if (param.custom_mask_type == 0)
          batched_backward_masktype_attnbias_dispatched<
              ck::bhalf_t,
              0,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>::Run(param, stream);
        else if (param.custom_mask_type == 1)
          batched_backward_masktype_attnbias_dispatched<
              ck::bhalf_t,
              1,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>::Run(param, stream);
        else if (param.custom_mask_type == 2)
          batched_backward_masktype_attnbias_dispatched<
              ck::bhalf_t,
              2,
              HAS_ATTN_BIAS,
              USE_FP32_QKV_GRAD>::Run(param, stream);
        else
          throw std::runtime_error("Invalid custom_mask_type value");
      });
};
