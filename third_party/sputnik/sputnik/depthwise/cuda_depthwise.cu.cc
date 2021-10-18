// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sputnik/common.h"
#include "sputnik/depthwise/computer.h"
#include "sputnik/depthwise/cuda_depthwise.h"
#include "sputnik/depthwise/depthwise_config.h"
#include "sputnik/depthwise/filter_tile.h"
#include "sputnik/depthwise/input_tile.h"
#include "sputnik/depthwise/output_tile.h"
#include "sputnik/depthwise/shape_utils.h"
#include "sputnik/depthwise/width_padding.h"
#include "sputnik/load_store.h"
#include "sputnik/tiling_utils.h"

#include "glog/logging.h"

namespace sputnik {

namespace {

/**
 * @brief Depthwise convolution w/ NCHW layout.
 */
template <typename Config>
struct DepthwiseKernel {
  // Input filter is kKernelSize x kKernelSize
  static constexpr int kKernelSize = Config::kKernelSize;

  // Implicit padding. Added to both spatial dimensions on
  // either side.
  static constexpr int kPadding = Config::kPadding;

  // Stride to access the input filter.
  static constexpr int kStride = Config::kStride;

  // Thread block output tile dimensions.
  static constexpr int kBlockItemsX = Config::kBlockItemsX;
  static constexpr int kBlockItemsY = Config::kBlockItemsY;

  // Thread output tile dimensions.
  static constexpr int kThreadItemsX = Config::kThreadItemsX;
  static constexpr int kThreadItemsY = Config::kThreadItemsY;

  // Thread block configuration.
  static constexpr int kBlockDimX = Config::kBlockDimX;
  static constexpr int kBlockDimY = Config::kBlockDimY;

  // Type of loads/stores to use.
  typedef typename Config::MemOp MemOp;

  static constexpr int kVectorWidth = sizeof(MemOp) / sizeof(float);

  typedef FilterTile<
      kKernelSize,
      kBlockDimX>
      FilterTile;

  typedef InputShape<
      kKernelSize,
      kPadding,
      kStride>
      InputShape;

  typedef OutputShape<
      kKernelSize,
      kPadding,
      kStride>
      OutputShape;

  typedef WidthPadding<
      MemOp,
      kKernelSize,
      kPadding,
      kStride,
      kBlockItemsX>
      WidthPadding;

  typedef InputTile<
      MemOp,
      kKernelSize,
      kPadding,
      kStride,
      kBlockItemsX,
      kBlockItemsY,
      kBlockDimX,
      kBlockDimY>
      InputTile;

  typedef Computer<
      MemOp,
      kKernelSize,
      kPadding,
      kStride,
      kBlockItemsX,
      kBlockItemsY,
      kThreadItemsX,
      kThreadItemsY>
      Computer;

  typedef OutputTile<
      MemOp,
      kBlockItemsX,
      kBlockItemsY,
      kThreadItemsX,
      kThreadItemsY>
      OutputTile;

  // Validate the config.
  static_assert(kPadding == 1 && kKernelSize == 3,
                "Currently only supports 3x3 with 'same' padding.");

  // We could generalize this to allow wider loads on the input.
  static_assert(kVectorWidth <= kThreadItemsX,
                "Vector width must be <= kThreadItemsX");

  static __device__ __forceinline__ void KernelFn(
      int n, int c, int h, int w, const float* __restrict__ in,
      const float* __restrict__ filters, const float* __restrict__ bias,
      float* __restrict__ out) {
    // The filters have shape [channels, height, width]. All the threads
    // in this block use the same filter. The filters almost always
    // multiply to an odd number (3x3, 5x5, etc.) so we can't use vector
    // instructions, but the filters are also small so we don't care
    // very much.
    constexpr int kNumWeights = kKernelSize * kKernelSize;
    float filter_fragment[kNumWeights];

    // Allocate the shared buffer for each thread block so we can use
    // it to broadcast the filters to all threads.
    //
    // The width padding is used to enable the use of vector memory
    // instructions on the input loads. We round up to the nearest
    // multiple of 4 that gives us greater than 3 padding values,
    // which is the maximum needed to enable 4-wide loads.
    //
    // The single row of height padding is needed so we can
    // explicitly pad the input tile in smem.
    constexpr int kInputTileX = kBlockItemsX + WidthPadding::Get();
    constexpr int kInputTileY = InputShape::NoPad(kBlockItemsY) + kPadding;
    constexpr int kNumInputs = kInputTileX * kInputTileY;
    __shared__ float in_tile[kNumInputs + 4];

    //
    /// Broadcast the filter weights across the thread block.
    //
    const int kChannelIdx = blockIdx.z % c;
    const int kFilterOffset = kChannelIdx * kNumWeights;
    FilterTile filter_tile_loader(kFilterOffset, filters, in_tile,
                                  filter_fragment);
    filter_tile_loader.Load();
    __syncthreads();

    //
    /// Load the input image tile to shared memory.
    //
    const int kInputOffsetW = blockIdx.x * kBlockItemsX * kStride - kPadding;
    const int kBaseInputOffsetW = Max(kInputOffsetW, 0);
    const int kInputOffsetH = blockIdx.y * kBlockItemsY * kStride - kPadding;
    const int kBaseInputOffsetH = Max(kInputOffsetH, 0);
    const int kBaseInputOffset =
        kBaseInputOffsetW + kBaseInputOffsetH * w + blockIdx.z * h * w;

    // Possibly offset the input tile so that we have a place
    // to store zeros for the height padding.
    const int kTileOffsetW = 4;
    const int kTileOffsetH = kInputOffsetH < 0 ? kPadding : 0;
    const int kTileOffset = kTileOffsetH * kInputTileX + kTileOffsetW;
    InputTile input_tile_loader(w, kBaseInputOffset, in, kTileOffset, in_tile);
    input_tile_loader.Load();
    __syncthreads();

    //
    /// Do the maths.
    //

    // Offset to the start of the actual data for this thread block.
    const int kSmemOffset = kBaseInputOffset & (kVectorWidth - 1);
    float* image_tile =
        OffsetCast<float>(in_tile, (kSmemOffset + 4) * sizeof(float));

    // If we need to left-pad the data, back up one value to make
    // space. Note that this is safe to do even when kSmemOffset is
    // 0 because we padded our shared memory allocation with 4 floats.
    if (kInputOffsetW < 0) image_tile -= 1;

    // Accumulator registers for the outputs. Initialize to zero
    // so we can accumulator in-place.
    float output_fragment[kThreadItemsX * kThreadItemsY] = {};
    Computer computer(h, w, image_tile, filter_fragment, output_fragment);
    computer.Compute();

    //
    /// Store the results.
    //

    // Possibly apply the bias and ReLU.
    if (bias != nullptr) {
      const float bias_value = Load(bias + kChannelIdx);

      constexpr int kOutputFragmentSize = kThreadItemsX * kThreadItemsY;
#pragma unroll
      for (int out_idx = 0; out_idx < kOutputFragmentSize; ++out_idx) {
        output_fragment[out_idx] += bias_value;
        output_fragment[out_idx] =
            output_fragment[out_idx] > 0 ? output_fragment[out_idx] : 0;
      }
    }

    const int kOutputHeight = OutputShape::Get(h);
    const int kOutputWidth = OutputShape::Get(w);
    const int kImageOffset = blockIdx.z * kOutputHeight * kOutputWidth;
    OutputTile output_storer(kOutputHeight, kOutputWidth, output_fragment,
                             kImageOffset, out);
    output_storer.Store();
  }
};

template <typename Config>
__global__ void __launch_bounds__(Config::kBlockDimX * Config::kBlockDimY)
  Kernel(int n, int c, int h, int w,
	 const float* __restrict__ in,
	 const float* __restrict__ filters,
	 const float* __restrict__ bias,
	 float* __restrict__ out) {
  DepthwiseKernel<Config>::KernelFn(n, c, h, w, in, filters, bias, out);
}

}  // namespace

constexpr bool DivBy(int x, int y) { return x % y == 0 ? true : false; }

bool VectorCompat(int h, int w, int out_h, int out_w, int vw) {
  return DivBy(w, vw) && DivBy(h, vw) && DivBy(out_w, vw) && DivBy(out_h, vw);
}

cudaError_t CudaDepthwiseBiasRelu(int n, int c, int h, int w,
                                  const float* __restrict__ in, int kernel_size,
                                  int padding, int stride,
                                  const float* __restrict__ filters,
                                  const float* __restrict__ bias,
                                  float* __restrict__ out,
                                  cudaStream_t stream) {
  int out_dim = (h - kernel_size + 2 * padding) / stride + 1;
  if (out_dim > 32) {
    return CudaDepthwiseEx<64, 64, 4, 4>(n, c, h, w, in, kernel_size, padding,
                                         stride, filters, bias, out, stream);
  } else if (out_dim > 16) {
    return CudaDepthwiseEx<32, 32, 2, 2>(n, c, h, w, in, kernel_size, padding,
                                         stride, filters, bias, out, stream);
  } else if (out_dim > 8) {
    return CudaDepthwiseEx<16, 16, 2, 2>(n, c, h, w, in, kernel_size, padding,
                                         stride, filters, bias, out, stream);
  }
  return CudaDepthwiseEx<8, 8, 1, 2>(n, c, h, w, in, kernel_size, padding,
                                     stride, filters, bias, out, stream);
}

cudaError_t CudaDepthwise(int n, int c, int h, int w,
                          const float* __restrict__ in, int kernel_size,
                          int padding, int stride,
                          const float* __restrict__ filters,
                          float* __restrict__ out, cudaStream_t stream) {
  return CudaDepthwiseBiasRelu(n, c, h, w, in, kernel_size, padding, stride,
                               filters, /*bias=*/nullptr, out, stream);
}

// Helper to avoid compiling invalid vector kernel.
template <int kThreadItemsX, typename MemOp>
struct Vector {
  typedef MemOp T;
};
template <typename MemOp>
struct Vector<2, MemOp> {
  typedef float2 T;
};
template <typename MemOp>
struct Vector<1, MemOp> {
  typedef float T;
};

template <int kBlockItemsX, int kBlockItemsY, int kThreadItemsX,
          int kThreadItemsY>
cudaError_t CudaDepthwiseEx(int n, int c, int h, int w,
                            const float* __restrict__ in, int kernel_size,
                            int padding, int stride,
                            const float* __restrict__ filters,
                            const float* __restrict__ bias,
                            float* __restrict__ out, cudaStream_t stream) {
  // Launch the largest vector instructions that fit in the thread tile.
  // Sanity check the input arguments.
  CHECK_EQ(h, w) << "Spatial dimensions must match.";
  CHECK_EQ(kernel_size, 3) << "Currently only supports 3x3 filters";
  CHECK_EQ(padding, 1) << "Currently only supports padding == 1";
  CHECK(stride == 1 || stride == 2)
      << "Currently only supports stride of 1 or 2";

  constexpr int kKernelSize = 3;
  constexpr int kPadding = 1;
  int out_h = (h - kKernelSize + 2 * kPadding) / stride + 1;
  int out_w = (w - kKernelSize + 2 * kPadding) / stride + 1;

  if (stride == 1) {
    if (DivBy(kThreadItemsX, 4) && VectorCompat(h, w, out_h, out_w, 4)) {
      typedef typename Vector<kThreadItemsX, float4>::T MemOp;
      typedef DepthwiseConfig<MemOp, 3, 1, 1, kBlockItemsX, kBlockItemsY,
                              kThreadItemsX, kThreadItemsY>
          Config;
      return CudaDepthwiseEx<Config>(n, c, h, w, in, filters, bias, out,
                                     stream);
    } else if (DivBy(kThreadItemsX, 2) && VectorCompat(h, w, out_h, out_w, 2)) {
      typedef typename Vector<kThreadItemsX, float2>::T MemOp;
      typedef DepthwiseConfig<MemOp, 3, 1, 1, kBlockItemsX, kBlockItemsY,
                              kThreadItemsX, kThreadItemsY>
          Config;
      return CudaDepthwiseEx<Config>(n, c, h, w, in, filters, bias, out,
                                     stream);
    } else {
      typedef DepthwiseConfig<float, 3, 1, 1, kBlockItemsX, kBlockItemsY,
                              kThreadItemsX, kThreadItemsY>
          Config;
      return CudaDepthwiseEx<Config>(n, c, h, w, in, filters, bias, out,
                                     stream);
    }
  } else {
    if (DivBy(kThreadItemsX, 4) && VectorCompat(h, w, out_h, out_w, 4)) {
      typedef typename Vector<kThreadItemsX, float4>::T MemOp;
      typedef DepthwiseConfig<MemOp, 3, 1, 2, Min(kBlockItemsX, 32),
                              Min(kBlockItemsY, 32), kThreadItemsX,
                              kThreadItemsY>
          Config;
      return CudaDepthwiseEx<Config>(n, c, h, w, in, filters, bias, out,
                                     stream);
    } else if (DivBy(kThreadItemsX, 2) && VectorCompat(h, w, out_h, out_w, 2)) {
      typedef typename Vector<kThreadItemsX, float2>::T MemOp;
      typedef DepthwiseConfig<MemOp, 3, 1, 2, Min(kBlockItemsX, 32),
                              Min(kBlockItemsY, 32), kThreadItemsX,
                              kThreadItemsY>
          Config;
      return CudaDepthwiseEx<Config>(n, c, h, w, in, filters, bias, out,
                                     stream);
    } else {
      typedef DepthwiseConfig<float, 3, 1, 2, Min(kBlockItemsX, 32),
                              Min(kBlockItemsY, 32), kThreadItemsX,
                              kThreadItemsY>
          Config;
      return CudaDepthwiseEx<Config>(n, c, h, w, in, filters, bias, out,
                                     stream);
    }
  }
}

#define INSTANTIATE_TILED(fn, bx, by, tx, ty)                               \
  template cudaError_t fn<bx, by, tx, ty>(int, int, int, int, const float*, \
                                          int, int, int, const float*,      \
                                          const float*, float*, cudaStream_t)

#ifdef SPUTNIK_BUILD_TEST
INSTANTIATE_TILED(CudaDepthwiseEx, 64, 64, 8, 8);
INSTANTIATE_TILED(CudaDepthwiseEx, 64, 64, 4, 8);
INSTANTIATE_TILED(CudaDepthwiseEx, 64, 64, 4, 4);
INSTANTIATE_TILED(CudaDepthwiseEx, 32, 32, 4, 8);
INSTANTIATE_TILED(CudaDepthwiseEx, 32, 32, 4, 4);
INSTANTIATE_TILED(CudaDepthwiseEx, 32, 32, 4, 2);
INSTANTIATE_TILED(CudaDepthwiseEx, 32, 32, 2, 2);
INSTANTIATE_TILED(CudaDepthwiseEx, 16, 16, 4, 2);
INSTANTIATE_TILED(CudaDepthwiseEx, 16, 16, 2, 4);
INSTANTIATE_TILED(CudaDepthwiseEx, 16, 16, 2, 2);
INSTANTIATE_TILED(CudaDepthwiseEx, 8, 8, 2, 1);
INSTANTIATE_TILED(CudaDepthwiseEx, 8, 8, 1, 2);
#endif  // SPUTNIK_BUILD_TEST

#undef INSTANTIATE_TILED

template <typename Config>
cudaError_t CudaDepthwiseEx(int n, int c, int h, int w,
                            const float* __restrict__ in,
                            const float* __restrict__ filters,
                            const float* __restrict__ bias,
                            float* __restrict__ out, cudaStream_t stream) {
  typedef OutputShape<Config::kKernelSize, Config::kPadding, Config::kStride>
      OutputShape;

  int grid_dim_x = RoundUpTo(OutputShape::Get(w), Config::kBlockItemsX) /
                   Config::kBlockItemsX;
  int grid_dim_y = RoundUpTo(OutputShape::Get(h), Config::kBlockItemsY) /
                   Config::kBlockItemsY;
  dim3 grid_dim(grid_dim_x, grid_dim_y, n * c);
  dim3 block_dim(Config::kBlockDimX, Config::kBlockDimY);

  Kernel<Config><<<grid_dim, block_dim, 0, stream>>>(
      n, c, h, w, in, filters, bias, out);
  return cudaGetLastError();
}

}  // namespace sputnik
