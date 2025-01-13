/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <algorithm>
#include "ck_tiled_fmha_fwd_setting.h"
#include "ck_tiled_fmha_grouped_forward_dispatch.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_grouped_forward_mask_bias_dropout_dispatch(
    GroupedForwardParams& param,
    hipStream_t stream) {
  if constexpr (!kHasDropout) {
    if (get_fmha_fwd_mtile(param.num_batches, param.Hq, param.max_seqlen_q) ==
        128)
      grouped_forward_mask_bias_dropout_dispatch<
          ScalarType,
          kHasMask,
          kHasBias,
          kHasDropout,
          MaxK,
          128>::Run(param, stream);
    else
      grouped_forward_mask_bias_dropout_dispatch<
          ScalarType,
          kHasMask,
          kHasBias,
          kHasDropout,
          MaxK,
          64>::Run(param, stream);
  } else {
    // at present, dropout of fwd kernel requires 32x32 WarpTile
    grouped_forward_mask_bias_dropout_dispatch<
        ScalarType,
        kHasMask,
        kHasBias,
        kHasDropout,
        MaxK,
        128>::Run(param, stream);
  }
};
