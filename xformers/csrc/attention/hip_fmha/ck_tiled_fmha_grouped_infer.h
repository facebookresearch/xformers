#pragma once

#include <sstream>
#include <stdexcept>

#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>

#include "ck_fmha_params.h"

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
struct grouped_infer_masktype_attnbias_dispatched {
  static void Run(GroupedForwardParams& param, hipStream_t stream){};

  template <typename DeviceOpInstance>
  static void RunWithDeviceOp(
      GroupedForwardParams& param,
      hipStream_t stream){};
};

template <typename scalar_t, int32_t custom_mask_type, bool has_attn_bias>
void run_grouped_infer_masktype_attnbias_dispatched(
    GroupedForwardParams& param,
    hipStream_t stream) {
  grouped_infer_masktype_attnbias_dispatched<
      scalar_t,
      custom_mask_type,
      has_attn_bias>::Run(param, stream);
};
