#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>

#include "ck_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
struct batched_infer_masktype_attnbias_dispatched {
  static void Run(BatchedForwardParams& param, hipStream_t stream){};

  template <typename DeviceOpInstance>
  static void RunWithDeviceOp(
      BatchedForwardParams& param,
      hipStream_t stream){};
};

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
void run_batched_infer_masktype_attnbias_dispatched(
    BatchedForwardParams& param,
    hipStream_t stream) {
  batched_infer_masktype_attnbias_dispatched<
      scalar_t,
      custom_mask_type,
      has_attn_bias>::Run(param, stream);
};
