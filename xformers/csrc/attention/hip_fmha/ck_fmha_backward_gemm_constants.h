/*
 * Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <ck/ck.hpp>
#include "ck_fmha_op_helper.h"

// list the template parameters that will not be tuned,
// the commented lines gives the tunable template parameters
struct GemmOpConstantsBatchedBackward_V1 {
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  // static constexpr ck::index_t KPerBlock;
  // static constexpr ck::index_t Gemm1NPerBlock;
  static constexpr ck::index_t Gemm1KPerBlock = 32;
  static constexpr ck::index_t Gemm2KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t B1K1 = 2;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 1;
  // static constexpr ck::index_t Gemm1NXdlPerWave;
  static constexpr ck::index_t Gemm2NXdlPerWave = 1;
  using ABlockTransferThreadClusterLengths_AK0_M_AK1 = S<4, 64, 1>;
  using ABlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using ABlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t ABlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t ABlockTransferSrcScalarPerVector;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr bool ABlockLdsExtraM = true;
  using BBlockTransferThreadClusterLengths_BK0_N_BK1 = S<4, 64, 1>;
  using BBlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using BBlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t BBlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t BBlockTransferSrcScalarPerVector;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr bool BBlockLdsExtraN = true;
  // static constexpr ck::index_t Acc0BiasTransferSrcScalarPerVector;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  // static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle;
  // using
  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
  // static constexpr ck::index_t
  // CShuffleBlockTransferScalarPerVector_NPerBlock;
};

// list the template parameters that will not be tuned,
// the commented lines gives the tunable template parameters
struct GemmOpConstantsBatchedBackward_V2 {
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 128;
  // static constexpr ck::index_t Gemm1NPerBlock;
  static constexpr ck::index_t Gemm1KPerBlock = 32;
  static constexpr ck::index_t Gemm2KPerBlock = 64;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t B1K1 = 2;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 1;
  // static constexpr ck::index_t Gemm1NXdlPerWave;
  static constexpr ck::index_t Gemm2NXdlPerWave = 1;
  using ABlockTransferThreadClusterLengths_AK0_M_AK1 = S<4, 64, 1>;
  using ABlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using ABlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t ABlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t ABlockTransferSrcScalarPerVector;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr bool ABlockLdsExtraM = true;
  using BBlockTransferThreadClusterLengths_BK0_N_BK1 = S<4, 64, 1>;
  using BBlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using BBlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t BBlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t BBlockTransferSrcScalarPerVector;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr bool BBlockLdsExtraN = true;
  // static constexpr ck::index_t Acc0BiasTransferSrcScalarPerVector;
  using B1BlockTransferThreadClusterLengths_BK0_N_BK1 = S<8, 32, 1>;
  using B1BlockTransferThreadClusterArrangeOrder = S<0, 2, 1>;
  using B1BlockTransferSrcAccessOrder = S<0, 2, 1>;
  static constexpr ck::index_t B1BlockTransferSrcVectorDim = 1;
  // static constexpr ck::index_t B1BlockTransferSrcScalarPerVector;
  static constexpr ck::index_t B1BlockTransferDstScalarPerVector_BK1 = 2;
  static constexpr bool B1BlockLdsExtraN = false;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  // static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle;
  // using
  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
  // static constexpr ck::index_t
  // CShuffleBlockTransferScalarPerVector_NPerBlock;
};

// list the template parameters that will not be tuned,
// the commented lines gives the tunable template parameters
struct GemmOpConstantsGroupedBackward_V1 {
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  // static constexpr ck::index_t KPerBlock;
  // static constexpr ck::index_t Gemm1NPerBlock;
  static constexpr ck::index_t Gemm1KPerBlock = 32;
  static constexpr ck::index_t Gemm2KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t B1K1 = 2;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 1;
  // static constexpr ck::index_t Gemm1NXdlPerWave;
  static constexpr ck::index_t Gemm2NXdlPerWave = 1;
  using ABlockTransferThreadClusterLengths_AK0_M_AK1 = S<4, 64, 1>;
  using ABlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using ABlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t ABlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t ABlockTransferSrcScalarPerVector;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr bool ABlockLdsExtraM = true;
  using BBlockTransferThreadClusterLengths_BK0_N_BK1 = S<4, 64, 1>;
  using BBlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using BBlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t BBlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t BBlockTransferSrcScalarPerVector;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr bool BBlockLdsExtraN = true;
  // static constexpr ck::index_t Acc0BiasTransferSrcScalarPerVector;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  // static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle;
  // using
  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
  // static constexpr ck::index_t
  // CShuffleBlockTransferScalarPerVector_NPerBlock;
};

// list the template parameters that will not be tuned,
// the commented lines gives the tunable template parameters
struct GemmOpConstantsGroupedBackward_V2 {
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 128;
  // static constexpr ck::index_t Gemm1NPerBlock;
  static constexpr ck::index_t Gemm1KPerBlock = 32;
  static constexpr ck::index_t Gemm2KPerBlock = 64;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t B1K1 = 2;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 1;
  // static constexpr ck::index_t Gemm1NXdlPerWave;
  static constexpr ck::index_t Gemm2NXdlPerWave = 1;
  using ABlockTransferThreadClusterLengths_AK0_M_AK1 = S<4, 64, 1>;
  using ABlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using ABlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t ABlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t ABlockTransferSrcScalarPerVector;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr bool ABlockLdsExtraM = true;
  using BBlockTransferThreadClusterLengths_BK0_N_BK1 = S<4, 64, 1>;
  using BBlockTransferThreadClusterArrangeOrder = S<1, 0, 2>;
  using BBlockTransferSrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t BBlockTransferSrcVectorDim = 2;
  // static constexpr ck::index_t BBlockTransferSrcScalarPerVector;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr bool BBlockLdsExtraN = true;
  // static constexpr ck::index_t Acc0BiasTransferSrcScalarPerVector;
  using B1BlockTransferThreadClusterLengths_BK0_N_BK1 = S<8, 32, 1>;
  using B1BlockTransferThreadClusterArrangeOrder = S<0, 2, 1>;
  using B1BlockTransferSrcAccessOrder = S<0, 2, 1>;
  static constexpr ck::index_t B1BlockTransferSrcVectorDim = 1;
  // static constexpr ck::index_t B1BlockTransferSrcScalarPerVector;
  static constexpr ck::index_t B1BlockTransferDstScalarPerVector_BK1 = 2;
  static constexpr bool B1BlockLdsExtraN = false;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  // static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle;
  // using
  // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock;
  // static constexpr ck::index_t
  // CShuffleBlockTransferScalarPerVector_NPerBlock;
};
