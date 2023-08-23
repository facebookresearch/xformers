#include <ck/ck.hpp>
#include <stdexcept>

#include "ck_fmha_grouped_backward.h"

void grouped_backward_bp16(GroupedBackwardParams& param, hipStream_t stream) {
  if (param.custom_mask_type == 0)
    grouped_backward_mask_type_dispatched<ck::bhalf_t, 0>(param, stream);
  else if (param.custom_mask_type == 1)
    grouped_backward_mask_type_dispatched<ck::bhalf_t, 1>(param, stream);
  else if (param.custom_mask_type == 2)
    grouped_backward_mask_type_dispatched<ck::bhalf_t, 2>(param, stream);
  else
    throw std::runtime_error("Invalid custom_mask_type value");
};
