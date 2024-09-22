/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_grouped_infer_dispatch.h"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_grouped_infer_causalmask_bias_dropout_dispatch(
    GroupedForwardParams& param,
    hipStream_t stream) {
  grouped_infer_causalmask_bias_dropout_dispatch<
      ScalarType,
      kHasCausalMask,
      kHasBias,
      kHasDropout,
      MaxK>::Run(param, stream);
};
