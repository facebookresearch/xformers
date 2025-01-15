/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include "ck_tiled_fmha_batched_forward_dispatch.h"
#include "ck_tiled_fmha_fwd_setting.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_batched_forward_mask_bias_dropout_dispatch(
    BatchedForwardParams& param,
    hipStream_t stream) {
  if constexpr (!kHasDropout) {
    if (get_fmha_fwd_mtile(param.B, param.Hq, param.M) == 128)
      batched_forward_mask_bias_dropout_dispatch<
          ScalarType,
          kHasMask,
          kHasBias,
          kHasDropout,
          MaxK,
          128>::Run(param, stream);
    else
      batched_forward_mask_bias_dropout_dispatch<
          ScalarType,
          kHasMask,
          kHasBias,
          kHasDropout,
          MaxK,
          64>::Run(param, stream);
  } else {
    // at present, dropout of fwd kernel requires 32x32 WarpTile
    batched_forward_mask_bias_dropout_dispatch<
        ScalarType,
        kHasMask,
        kHasBias,
        kHasDropout,
        MaxK,
        128>::Run(param, stream);
  }
};
