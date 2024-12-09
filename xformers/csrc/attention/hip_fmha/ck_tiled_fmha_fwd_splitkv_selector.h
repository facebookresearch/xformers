/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cmath>
#include <tuple>
#include "ck_fmha_util.h"
#include "ck_tiled_fmha_fwd_setting.h"
#include "ck_tiled_fmha_fwd_splitkv_setting.h"
#include "ck_tiled_fmha_fwd_splitkv_smallq_setting.h"
#include "ck_tiled_fmha_seqlen_q_switch.h"

// generate a list of numbers as num_splits to consider, the list of numbers is
// like 1, 2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320
static int generate_splits_list(int i) {
  if (i <= 0)
    return 1;

  if (i <= 5)
    return 1 << (i - 1);
  else
    return (i - 5) * 32;
};

static std::pair<bool, int> get_num_kv_splits_heuristic(
    int num_batches,
    int num_heads,
    int max_seqlen_q,
    int max_headdim,
    int max_splits) {
  int num_SMs = get_number_of_cu();
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };

  int mtile_size_for_pipeline_default = 128;
  int mtile_size_for_splitkv = 64;
  int mtile_size_for_splitkv_smallq = 16;

  // get mtile_size_for_pipline_default
  if (max_headdim <= 32) {
    mtile_size_for_pipeline_default = fwd_get_mtile_size<32>();
  } else if (max_headdim <= 64) {
    mtile_size_for_pipeline_default = fwd_get_mtile_size<64>();
  } else if (max_headdim <= 96) {
    mtile_size_for_pipeline_default = fwd_get_mtile_size<96>();
  } else if (max_headdim <= 128) {
    mtile_size_for_pipeline_default = fwd_get_mtile_size<128>();
  } else {
    mtile_size_for_pipeline_default = fwd_get_mtile_size<256>();
  };

  // get mtile_size_for_splitkv
  FMHA_FWD_SEQLEN_Q_SWITCH(max_seqlen_q, MaxSeqLenQ, [&] {
    if (max_headdim <= 32) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<32, MaxSeqLenQ>();
    } else if (max_headdim <= 64) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<64, MaxSeqLenQ>();
    } else if (max_headdim <= 96) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<96, MaxSeqLenQ>();
    } else if (max_headdim <= 128) {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<128, MaxSeqLenQ>();
    } else {
      mtile_size_for_splitkv = fwd_splitkv_get_mtile_size<256, MaxSeqLenQ>();
    };
  });

  // get mtile_size_for_splitkv_smallq
  if (max_headdim <= 32) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<32>();
  } else if (max_headdim <= 64) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<64>();
  } else if (max_headdim <= 96) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<96>();
  } else if (max_headdim <= 128) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<128>();
  } else {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<256>();
  };

  if (max_seqlen_q >= mtile_size_for_pipeline_default) {
    int batch_nhead_mblocks = num_batches * num_heads *
        ceildiv(max_seqlen_q, mtile_size_for_pipeline_default);

    if (batch_nhead_mblocks >= 0.8f * num_SMs)
      return std::make_pair(false, 1);
  }

  bool use_splitkv = true;

  // m_tile size is the size for dividing the seqlen_q
  int mtile_size;

  if (max_seqlen_q <= mtile_size_for_splitkv_smallq)
    mtile_size = mtile_size_for_splitkv_smallq;
  else
    mtile_size = mtile_size_for_splitkv;

  int batch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, mtile_size);

  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nhead_mblocks >= 0.8f * num_SMs) {
    return std::make_pair(use_splitkv, 1);
  }

  /*
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
        return std::make_pair(use_splitkv, num_splits);
      }
    }
    return std::make_pair(use_splitkv, 1);
  */

  max_splits = std::min({max_splits, num_SMs});

  int max_check = 1;

  while (generate_splits_list(max_check) <= max_splits)
    max_check++;

  int num_splits = 1;
  for (int i = 1; i < max_check; i++) {
    num_splits = generate_splits_list(i);

    if (batch_nhead_mblocks * num_splits >= 0.8 * num_SMs)
      break;
  };

  return std::make_pair(use_splitkv, num_splits);
}

static bool use_splitkv_smallq(int max_seqlen_q, int max_headdim) {
  int mtile_size_for_splitkv_smallq = 16;

  // get mtile_size_for_splitkv_smallq
  if (max_headdim <= 32) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<32>();
  } else if (max_headdim <= 64) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<64>();
  } else if (max_headdim <= 96) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<96>();
  } else if (max_headdim <= 128) {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<128>();
  } else {
    mtile_size_for_splitkv_smallq = fwd_splitkv_smallq_get_mtile_size<256>();
  };

  if (max_seqlen_q <= mtile_size_for_splitkv_smallq)
    return true;
  else
    return false;
}
