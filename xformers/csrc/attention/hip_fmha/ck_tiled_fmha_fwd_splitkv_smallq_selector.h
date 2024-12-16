/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_fwd_splitkv_smallq_setting.h"

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
