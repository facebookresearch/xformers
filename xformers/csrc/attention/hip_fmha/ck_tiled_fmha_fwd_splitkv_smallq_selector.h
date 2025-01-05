/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "ck_tiled_fmha_fwd_splitkv_setting.h"
#include "ck_tiled_fmha_fwd_splitkv_smallq_setting.h"

/// This method determines whether to use normal or smallq splitkv kernel
static bool use_splitkv_smallq(int max_seqlen_q, int max_headdim) {
  int mtile_size_for_splitkv_smallq =
      get_mtile_size_for_splitkv_smallq(max_headdim);

  // resort to splitkv-smallq kernel for avoiding wasting of computation
  if (max_seqlen_q <= mtile_size_for_splitkv_smallq)
    return true;

  return false;
}
