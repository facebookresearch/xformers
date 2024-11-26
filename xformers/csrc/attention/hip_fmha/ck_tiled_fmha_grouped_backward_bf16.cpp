/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck_tile/core.hpp>
#include <stdexcept>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_grouped_backward.h"
#include "ck_tiled_headdim_switch.h"

#include "instances/fmha_grouped_backward_bf16_instances_ref.h"

void grouped_backward_bf16(GroupedBackwardParams& param, hipStream_t stream) {
  const bool has_dropout = (param.dropout_prob > 0.0f);
  BOOL_SWITCH_3(
      param.has_attn_bias,
      kHasBias,
      param.bias_has_grad,
      kHasBiasGrad,
      has_dropout,
      kHasDropout,
      [&] {
        if constexpr (kHasBias || !kHasBiasGrad) {
          FMHA_BWD_HEADDIM_SWITCH(param.K, param.Kv, MaxK, [&] {
            if (param.custom_mask_type == 0 && param.window_size <= 0)
              run_grouped_backward_mask_bias_dropout_dispatch<
                  ck_tile::bf16_t,
                  false,
                  kHasBias,
                  kHasBiasGrad,
                  kHasDropout,
                  MaxK>(param, stream);
            else if (
                param.custom_mask_type == 1 || param.custom_mask_type == 2 ||
                param.window_size > 0)
              run_grouped_backward_mask_bias_dropout_dispatch<
                  ck_tile::bf16_t,
                  true,
                  kHasBias,
                  kHasBiasGrad,
                  kHasDropout,
                  MaxK>(param, stream);
            else
              throw std::runtime_error("Invalid custom_mask_type value");
          });
        } else
          throw std::runtime_error(
              "bias_has_grad should be false when has_attn_bias is false!");
      });
};
