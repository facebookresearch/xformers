/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_batched_infer_dispatch.h"
#include "ck_tiled_fmha_batched_infer_splitkv_dispatch.h"

template <
    typename ScalarType,
    bool kHasCausalMask,
    bool kHasBias,
    bool kHasDropout,
    ck_tile::index_t MaxK>
void run_batched_infer_causalmask_bias_dropout_dispatch(
    BatchedForwardParams& param,
    hipStream_t stream) {
  // currently split-kv implementation does not support dropout
  if constexpr (!kHasDropout) {
    if (!param.use_split_kv)
      batched_infer_causalmask_bias_dropout_dispatch<
          ScalarType,
          kHasCausalMask,
          kHasBias,
          kHasDropout,
          MaxK>::Run(param, stream);
    else
      batched_infer_splitkv_causalmask_bias_dropout_dispatch<
          ScalarType,
          kHasCausalMask,
          kHasBias,
          MaxK>::Run(param, stream);
  } else {
    batched_infer_causalmask_bias_dropout_dispatch<
        ScalarType,
        kHasCausalMask,
        kHasBias,
        kHasDropout,
        MaxK>::Run(param, stream);
  }
};
