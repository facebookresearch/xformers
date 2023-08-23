#include <ck.hpp>
#include <stdexcept>

#include "ck_fmha_batched_backward.h"

void batched_backward_fp16(BatchedBackwardParams& param, hipStream_t stream) {
  if (param.custom_mask_type == 0)
    batched_backward_mask_type_dispatched<ck::half_t, 0>(param, stream);
  else if (param.custom_mask_type == 1)
    batched_backward_mask_type_dispatched<ck::half_t, 1>(param, stream);
  else if (param.custom_mask_type == 2)
    batched_backward_mask_type_dispatched<ck::half_t, 2>(param, stream);
  else
    throw std::runtime_error("Invalid custom_mask_type value");
};
