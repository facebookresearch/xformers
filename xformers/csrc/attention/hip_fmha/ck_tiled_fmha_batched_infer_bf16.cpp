/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck_tile/core.hpp>
#include <stdexcept>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_batched_infer.h"

#include "instances/fmha_batched_infer_bf16_instances_ref.h"

void batched_infer_bf16(BatchedForwardParams& param, hipStream_t stream) {
  const bool has_dropout = (param.dropout_prob > 0.0f);
  BOOL_SWITCH_2(param.has_attn_bias, kHasBias, has_dropout, kHasDropout, [&] {
    FMHA_FWD_HEADDIM_SWITCH(param.K, param.Kv, MaxK, [&] {
      if (param.custom_mask_type == 0 && param.window_size <= 0)
        run_batched_infer_mask_bias_dropout_dispatch<
            ck_tile::bf16_t,
            false,
            kHasBias,
            kHasDropout,
            MaxK>(param, stream);
      else if (
          param.custom_mask_type == 1 || param.custom_mask_type == 2 ||
          param.window_size > 0)
        run_batched_infer_mask_bias_dropout_dispatch<
            ck_tile::bf16_t,
            true,
            kHasBias,
            kHasDropout,
            MaxK>(param, stream);
      else
        throw std::runtime_error("Invalid custom_mask_type value");
    });
  });
};
