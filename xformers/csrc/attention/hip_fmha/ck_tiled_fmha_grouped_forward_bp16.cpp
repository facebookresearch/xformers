/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck/ck.hpp>
#include <ck/utility/data_type.hpp>
#include <stdexcept>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_grouped_forward.h"

// clang-format off
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, true, 32>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, false, 32>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, true, 32>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, false, 32>(
    GroupedForwardParams& param, hipStream_t stream);

extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, true, 64>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, false, 64>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, true, 64>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, false, 64>(
    GroupedForwardParams& param, hipStream_t stream);

extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, true, 128>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, false, 128>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, true, 128>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, false, 128>(
    GroupedForwardParams& param, hipStream_t stream);

extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, true, 256>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, false, false, 256>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, true, 256>(
    GroupedForwardParams& param, hipStream_t stream);
extern template void run_grouped_forward_causalmask_attnbias_dispatched<ck::bhalf_t, true, false, 256>(
    GroupedForwardParams& param, hipStream_t stream);
// clang-format on

void grouped_forward_bp16(GroupedForwardParams& param, hipStream_t stream) {
  BOOL_SWITCH(param.has_attn_bias, HAS_ATTN_BIAS, [&] {
    FMHA_FWD_HEADDIM_SWITCH(param.K, param.Kv, MaxK, [&] {
      if (param.custom_mask_type == 0)
        run_grouped_forward_causalmask_attnbias_dispatched<
            ck::bhalf_t,
            false,
            HAS_ATTN_BIAS,
            MaxK>(param, stream);
      else if (param.custom_mask_type == 1 || param.custom_mask_type == 2)
        run_grouped_forward_causalmask_attnbias_dispatched<
            ck::bhalf_t,
            true,
            HAS_ATTN_BIAS,
            MaxK>(param, stream);
      else
        throw std::runtime_error("Invalid custom_mask_type value");
    });
  });
};
