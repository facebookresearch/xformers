/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <iostream>

#include <torch/library.h>

namespace {

// For testing xFormers building and binding
bool is_ck_fmha_available(double val) {
  std::cout << "ck fmha is really here, val=" << val << std::endl;
  return (true);
};

} // namespace

TORCH_LIBRARY_FRAGMENT(xformers, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "xformers::is_ck_fmha_available(float val) -> bool"));
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::is_ck_fmha_available"),
      TORCH_FN(is_ck_fmha_available));
}
