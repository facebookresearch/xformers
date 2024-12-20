/*
 * Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck_tile/core.hpp>

template <typename DataType>
struct FmhaFwdTypeConfig;

template <>
struct FmhaFwdTypeConfig<ck_tile::fp16_t> {
  using QDataType = ck_tile::fp16_t;
  using KDataType = ck_tile::fp16_t;
  using VDataType = ck_tile::fp16_t;
  using BiasDataType = ck_tile::fp16_t;
  using RandValOutputDataType = unsigned short;
  using LSEDataType =
      float; // data type for lse(logsumexp L_j = max_j + log(l_j))
  using SaccDataType = float; // data type for first gemm accumulation
  using SMPLComputeDataType = float; // data type for reduction, softmax
  using PDataType = ck_tile::fp16_t; // data type for A matrix of second gemm
  using OaccDataType = float; // data type for second gemm accumulation
  using ODataType = ck_tile::fp16_t;
};

template <>
struct FmhaFwdTypeConfig<ck_tile::bf16_t> {
  using QDataType = ck_tile::bf16_t;
  using KDataType = ck_tile::bf16_t;
  using VDataType = ck_tile::bf16_t;
  using BiasDataType = ck_tile::bf16_t;
  using RandValOutputDataType = unsigned short;
  using LSEDataType =
      float; // data type for lse(logsumexp L_j = max_j + log(l_j))
  using SaccDataType = float; // data type for first gemm accumulation
  using SMPLComputeDataType = float; // data type for reduction, softmax
  using PDataType = ck_tile::bf16_t; // data type for A matrix of second gemm
  using OaccDataType = float; // data type for second gemm accumulation
  using ODataType = ck_tile::bf16_t;
};

static constexpr bool IsVLayoutRowMajor = true;
