/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <ck_tile/core.hpp>
#include <stdexcept>

#include "ck_tiled_bool_switch.h"
#include "ck_tiled_fmha_batched_forward.h"
#include "ck_tiled_headdim_switch.h"

// clang-format off
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, true, 32>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, true, 32>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, true, 32>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, true, 32>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, false, 32>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, false, 32>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, false, 32>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, false, 32>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, true, 64>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, true, 64>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, true, 64>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, true, 64>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, false, 64>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, false, 64>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, false, 64>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, false, 64>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, true, 128>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, true, 128>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, true, 128>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, true, 128>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, false, 128>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, false, 128>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, false, 128>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, false, 128>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, true, 256>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, true, 256>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, true, 256>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, true, 256>(
    BatchedForwardParams& param, hipStream_t stream);

extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, true, false, 256>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, false, false, false, 256>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, true, false, 256>(
    BatchedForwardParams& param, hipStream_t stream);
extern template void run_batched_forward_causalmask_bias_dropout_dispatch<ck_tile::bf16_t, true, false, false, 256>(
    BatchedForwardParams& param, hipStream_t stream);
// clang-format on

void batched_forward_bf16(BatchedForwardParams& param, hipStream_t stream) {
  const bool has_dropout = (param.dropout_prob > 0.0f);
  BOOL_SWITCH_2(param.has_attn_bias, kHasBias, has_dropout, kHasDropout, [&] {
    FMHA_FWD_HEADDIM_SWITCH(param.K, param.Kv, MaxK, [&] {
      if (param.custom_mask_type == 0)
        run_batched_forward_causalmask_bias_dropout_dispatch<
            ck_tile::bf16_t,
            false,
            kHasBias,
            kHasDropout,
            MaxK>(param, stream);
      else if (param.custom_mask_type == 1 || param.custom_mask_type == 2)
        run_batched_forward_causalmask_bias_dropout_dispatch<
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
