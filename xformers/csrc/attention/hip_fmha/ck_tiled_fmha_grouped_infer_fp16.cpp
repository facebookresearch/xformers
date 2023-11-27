#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <stdexcept>

#include "ck_bool_switch.h"
#include "ck_tiled_fmha_grouped_infer.h"

extern template void run_grouped_infer_masktype_attnbias_dispatched<
    ck::half_t,
    0>(GroupedForwardParams& param, hipStream_t stream);

extern template void run_grouped_infer_masktype_attnbias_dispatched<
    ck::half_t,
    1>(GroupedForwardParams& param, hipStream_t stream);

extern template void run_grouped_infer_masktype_attnbias_dispatched<
    ck::half_t,
    2>(GroupedForwardParams& param, hipStream_t stream);

void grouped_infer_fp16(GroupedForwardParams& param, hipStream_t stream) {
  if (param.custom_mask_type == 0)
    run_grouped_infer_masktype_attnbias_dispatched<ck::half_t, 0>(
        param, stream);
  else if (param.custom_mask_type == 1)
    run_grouped_infer_masktype_attnbias_dispatched<ck::half_t, 1>(
        param, stream);
  else if (param.custom_mask_type == 2)
    run_grouped_infer_masktype_attnbias_dispatched<ck::half_t, 2>(
        param, stream);
  else
    throw std::runtime_error("Invalid custom_mask_type value");
};
