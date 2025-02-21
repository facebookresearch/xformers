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
// like 1, 2, 4, 8, 16, 32, 64, 96, 128, 160
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

  int mtile_size_for_pipeline_default = get_fmha_fwd_least_mtile();
  int mtile_size_for_splitkv = 64;
  int mtile_size_for_splitkv_smallq = 16;

  // get mtile_size_for_splitkv
  mtile_size_for_splitkv =
      get_mtile_size_for_splitkv(max_seqlen_q, max_headdim);

  // get mtile_size_for_splitkv_smallq
  mtile_size_for_splitkv_smallq =
      get_mtile_size_for_splitkv_smallq(max_headdim);

  // hdim-512 is not supported by splitkv-kernel at present
  if (max_headdim > 256)
    return std::make_pair(false, 1);

  if (max_seqlen_q >= mtile_size_for_pipeline_default) {
    int batch_nhead_mblocks = num_batches * num_heads *
        ceildiv(max_seqlen_q, mtile_size_for_pipeline_default);

    if (batch_nhead_mblocks >= 0.8f * num_SMs)
      return std::make_pair(false, 1);
  }

  bool use_splitkv = true;

  // m_tile size is the size for dividing the seqlen_q
  // we first tries to use the normal splitkv kernel
  int mtile_size = mtile_size_for_splitkv;
  int batch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, mtile_size);

  // resort to splitkv-smallq kernel for avoiding wasting of computation or for
  // better CU occupancy
  if (max_seqlen_q <= mtile_size_for_splitkv_smallq)
    mtile_size = mtile_size_for_splitkv_smallq;

  batch_nhead_mblocks =
      num_batches * num_heads * ceildiv(max_seqlen_q, mtile_size);

  // If we have enough workgroups to fill all the SMs, then just use 1 split
  if (batch_nhead_mblocks >= 0.9f * num_SMs) {
    return std::make_pair(use_splitkv, 1);
  }

  max_splits = std::min({max_splits, num_SMs});

  int max_check = 1;

  while (generate_splits_list(max_check) <= max_splits)
    max_check++;

  int num_splits = 2;
  for (int i = 2; i < max_check; i++) {
    num_splits = generate_splits_list(i);

    if (batch_nhead_mblocks * num_splits >= num_SMs)
      break;
  };

  return std::make_pair(use_splitkv, num_splits);
}
