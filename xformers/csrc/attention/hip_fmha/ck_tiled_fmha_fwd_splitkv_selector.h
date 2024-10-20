/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cmath>
#include "ck_fmha_util.h"
#include "ck_tiled_fmha_fwd_setting.h"

static int get_num_kv_splits_heuristic(
    int num_batches,
    int num_heads,
    int max_seqlen_q,
    int max_headdim,
    int max_splits) {
  // m_tile size is the size for dividing the seqlen_q
  int mtile_size;

  if (max_headdim <= 32) {
    mtile_size = FmhaFwdSplitKVShape<32>::kM0;
  } else if (max_headdim <= 64) {
    mtile_size = FmhaFwdSplitKVShape<64>::kM0;
  } else if (max_headdim <= 128) {
    mtile_size = FmhaFwdSplitKVShape<128>::kM0;
  } else {
    mtile_size = FmhaFwdSplitKVShape<256>::kM0;
  };

  int num_SMs = get_number_of_cu() * 2;

  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  int batch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, mtile_size);

  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nhead_mblocks >= 0.8f * num_SMs) {
    return 1;
  }

  max_splits = std::min({max_splits, num_SMs});

  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);

  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    float n_blocks = float(batch_nhead_mblocks * num_splits) / num_SMs;
    float eff = n_blocks / std::ceil(n_blocks);

    if (eff > max_efficiency) {
      max_efficiency = eff;
    }
    efficiency.push_back(eff);
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      return num_splits;
    }
  }
  return 1;
}
