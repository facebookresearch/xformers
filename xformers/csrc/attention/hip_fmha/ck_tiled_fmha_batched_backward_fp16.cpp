/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck_tile/core.hpp>
#include <stdexcept>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_batched_backward.h"
#include "ck_tiled_headdim_switch.h"

// clang-format off
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, true, true, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, false, true, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, false, false, true, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, true, true, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, false, true, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, false, false, true, 32>(
    BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, true, false, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, false, false, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, false, false, false, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, true, false, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, false, false, 32>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, false, false, false, 32>(
    BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, true, true, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, false, true, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, false, false, true, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, true, true, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, false, true, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, false, false, true, 64>(
    BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, true, false, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, false, false, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, false, false, false, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, true, false, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, false, false, 64>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, false, false, false, 64>(
    BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, true, true, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, false, true, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, false, false, true, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, true, true, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, false, true, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, false, false, true, 128>(
    BatchedBackwardParams& param, hipStream_t stream);

extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, true, false, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, true, false, false, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, false, false, false, false, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, true, false, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, true, false, false, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
extern template void run_batched_backward_causalmask_bias_dropout_dispatch<ck_tile::fp16_t, true, false, false, false, 128>(
    BatchedBackwardParams& param, hipStream_t stream);
// clang-format on

void batched_backward_fp16(BatchedBackwardParams& param, hipStream_t stream) {
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
            if (param.custom_mask_type == 0)
              run_batched_backward_causalmask_bias_dropout_dispatch<
                  ck_tile::fp16_t,
                  false,
                  kHasBias,
                  kHasBiasGrad,
                  kHasDropout,
                  MaxK>(param, stream);
            else if (param.custom_mask_type == 1 || param.custom_mask_type == 2)
              run_batched_backward_causalmask_bias_dropout_dispatch<
                  ck_tile::fp16_t,
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
