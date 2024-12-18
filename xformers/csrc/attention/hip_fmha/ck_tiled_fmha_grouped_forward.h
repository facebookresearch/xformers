/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_grouped_forward_dispatch.h"
#include "ck_tiled_fmha_grouped_forward_splitkv_dispatch.h"
#include "ck_tiled_fmha_seqlen_q_switch.h"

template <
    typename ScalarType,
    bool kHasMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_grouped_forward_mask_bias_dropout_dispatch(
    GroupedForwardParams& param,
    hipStream_t stream) {
  // currently split-kv implementation does not support dropout
  if constexpr (!kHasDropout) {
#ifndef FMHA_FWD_SPLITKV_NOT_USED
    if (param.use_split_kv) {
      if constexpr (MaxK <= 256) {
      FMHA_FWD_SEQLEN_Q_SWITCH(param.max_seqlen_q, MaxSeqlenQ, [&] {
        grouped_forward_splitkv_mask_bias_dropout_dispatch<
            ScalarType,
            kHasMask,
            kHasBias,
            MaxK,
            MaxSeqlenQ>::Run(param, stream);
      });
      }
    } else
#endif
      grouped_forward_mask_bias_dropout_dispatch<
          ScalarType,
          kHasMask,
          kHasBias,
          kHasDropout,
          MaxK>::Run(param, stream);
  } else {
    grouped_forward_mask_bias_dropout_dispatch<
        ScalarType,
        kHasMask,
        kHasBias,
        kHasDropout,
        MaxK>::Run(param, stream);
  }
};
