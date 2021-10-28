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

#include "sputnik/cuda_utils.h"
#include "sputnik/depthwise/cuda_depthwise.h"
#include "sputnik/depthwise/depthwise_config.h"
#include "sputnik/matrix_utils.h"
#include "sputnik/test_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/random/random.h"

namespace sputnik {

using ::testing::NanSensitiveFloatNear;
using ::testing::Pointwise;

template <int kDimN_, int kDimC_, int kDimH_, int kDimW_, int kKernelSize_,
          int kPadding_, int kStride_>
struct Problem {
  static constexpr int kDimN = kDimN_;
  static constexpr int kDimC = kDimC_;
  static constexpr int kDimH = kDimH_;
  static constexpr int kDimW = kDimW_;
  static constexpr int kKernelSize = kKernelSize_;
  static constexpr int kPadding = kPadding_;
  static constexpr int kStride = kStride_;

  static_assert(kDimH == kDimW);

  static constexpr int kDimOutH =
      (kDimH - kKernelSize + 2 * kPadding) / kStride + 1;
  static constexpr int kDimOutW =
      (kDimW - kKernelSize + 2 * kPadding) / kStride + 1;
};

template <typename Problem>
class DepthwiseTest : public ::testing::Test {
 public:
  const int kDimN = Problem::kDimN;
  const int kDimC = Problem::kDimC;
  const int kDimH = Problem::kDimH;
  const int kDimW = Problem::kDimW;
  const int kKernelSize = Problem::kKernelSize;
  const int kPadding = Problem::kPadding;
  const int kStride = Problem::kStride;
  const int kDimOutH = Problem::kDimOutH;
  const int kDimOutW = Problem::kDimOutW;

  // Random number generator for creating matrices.
  absl::BitGen generator_;

  void Pad(const float* in, float* out) {
    // Zero the entire output.
    int kNumOutput =
        kDimN * kDimC * (kDimH + kPadding * 2) * (kDimW + kPadding * 2);
    std::memset(out, 0, kNumOutput * sizeof(float));

    // Copy the input images over.
    for (int i = 0; i < kDimN * kDimC; ++i) {
      const float* input = in + i * kDimH * kDimW;
      float* output = out + i * (kDimH + kPadding * 2) * (kDimW + kPadding * 2);

      for (int h_idx = 0; h_idx < kDimH; ++h_idx) {
        for (int w_idx = 0; w_idx < kDimW; ++w_idx) {
          int out_idx =
              (h_idx + kPadding) * (kDimW + kPadding * 2) + w_idx + kPadding;
          output[out_idx] = input[h_idx * kDimW + w_idx];
        }
      }
    }
  }

  void DepthwiseConvolution(const float* in, const float* filters, float* out) {
    // Pad the input image.
    const int kDimInH = kDimH + kPadding * 2;
    const int kDimInW = kDimW + kPadding * 2;
    const int kNumOutput = kDimN * kDimC * kDimInH * kDimInW;
    std::vector<float> padded_input(kNumOutput);
    Pad(in, padded_input.data());

    // Compute the convolution.
    in = padded_input.data();
    for (int i = 0; i < kDimN * kDimC; ++i) {
      for (int h_idx = 0; h_idx < kDimOutH; ++h_idx) {
        for (int w_idx = 0; w_idx < kDimOutW; ++w_idx) {
          float accumulator = 0.0f;
          for (int fy_idx = 0; fy_idx < kKernelSize; ++fy_idx) {
            for (int fx_idx = 0; fx_idx < kKernelSize; ++fx_idx) {
              const int kWeightIdx = (i % kDimC) * kKernelSize * kKernelSize +
                                     fy_idx * kKernelSize + fx_idx;
              float weight = filters[kWeightIdx];

              const int kInIdxNC = i * kDimInH * kDimInW;
              const int kInIdxH = h_idx * kStride + fy_idx;
              const int kInIdxW = w_idx * kStride + fx_idx;
              const int kInIdx = kInIdxNC + kInIdxH * kDimInW + kInIdxW;
              float pixel = in[kInIdx];

              accumulator += weight * pixel;
            }
          }
          const int kOutIdxNC = i * kDimOutH * kDimOutW;
          const int kOutIdx = kOutIdxNC + h_idx * kDimOutW + w_idx;
          out[kOutIdx] = accumulator;
        }
      }
    }
  }
};

typedef ::testing::Types<
    Problem<1, 1, 32, 32, 3, 1, 1>,
    Problem<1, 2, 32, 32, 3, 1, 1>,
    Problem<4, 2, 32, 32, 3, 1, 1>,
    Problem<1, 1, 64, 64, 3, 1, 1>,
    Problem<1, 1, 28, 28, 3, 1, 1>,
    Problem<1, 10, 128, 128, 3, 1, 1>,
    Problem<1, 1, 30, 30, 3, 1, 1>,
    Problem<1, 1, 64, 64, 3, 1, 2>,
    Problem<1, 1, 60, 60, 3, 1, 2>,
    Problem<1, 1, 56, 56, 3, 1, 2>,
    Problem<1, 1, 128, 128, 3, 1, 2>,
    Problem<1, 2, 112, 112, 3, 1, 2>,
    // MobileNetV1 width 1.8
    Problem<1, 56, 112, 112, 3, 1, 1>,
    Problem<1, 115, 112, 112, 3, 1, 2>,
    Problem<1, 232, 56, 56, 3, 1, 1>,
    Problem<1, 232, 56, 56, 3, 1, 2>,
    Problem<1, 464, 28, 28, 3, 1, 1>,
    Problem<1, 464, 28, 28, 3, 1, 2>,
    Problem<1, 920, 14, 14, 3, 1, 1>,
    Problem<1, 920, 14, 14, 3, 1, 2>,
    Problem<1, 1840, 7, 7, 3, 1, 1>,
    // MobileNetV1 width 1.7
    Problem<1, 56, 112, 112, 3, 1, 1>,
    Problem<1, 108, 112, 112, 3, 1, 2>,
    Problem<1, 216, 56, 56, 3, 1, 1>,
    Problem<1, 216, 56, 56, 3, 1, 2>,
    Problem<1, 432, 28, 28, 3, 1, 1>,
    Problem<1, 432, 28, 28, 3, 1, 2>,
    Problem<1, 872, 14, 14, 3, 1, 1>,
    Problem<1, 872, 14, 14, 3, 1, 2>,
    Problem<1, 1744, 7, 7, 3, 1, 1>,
    // MobileNet width 1.6
    Problem<1, 48, 112, 112, 3, 1, 1>,
    Problem<1, 102, 112, 112, 3, 1, 2>,
    Problem<1, 208, 56, 56, 3, 1, 1>,
    Problem<1, 208, 56, 56, 3, 1, 2>,
    Problem<1, 408, 28, 28, 3, 1, 1>,
    Problem<1, 408, 28, 28, 3, 1, 2>,
    Problem<1, 816, 14, 14, 3, 1, 1>,
    Problem<1, 816, 14, 14, 3, 1, 2>,
    Problem<1, 1640, 7, 7, 3, 1, 1>,
    // MobileNet width 1.5
    Problem<1, 48, 112, 112, 3, 1, 1>,
    Problem<1, 96, 112, 112, 3, 1, 2>,
    Problem<1, 192, 56, 56, 3, 1, 1>,
    Problem<1, 192, 56, 56, 3, 1, 2>,
    Problem<1, 384, 28, 28, 3, 1, 1>,
    Problem<1, 384, 28, 28, 3, 1, 2>,
    Problem<1, 768, 14, 14, 3, 1, 1>,
    Problem<1, 768, 14, 14, 3, 1, 2>,
    Problem<1, 1536, 7, 7, 3, 1, 1>,
    // MobileNet width 1.4
    Problem<1, 48, 112, 112, 3, 1, 1>,
    Problem<1, 89, 112, 112, 3, 1, 2>,
    Problem<1, 176, 56, 56, 3, 1, 1>,
    Problem<1, 176, 56, 56, 3, 1, 2>,
    Problem<1, 360, 28, 28, 3, 1, 1>,
    Problem<1, 360, 28, 28, 3, 1, 2>,
    Problem<1, 720, 14, 14, 3, 1, 1>,
    Problem<1, 720, 14, 14, 3, 1, 2>,
    Problem<1, 1432, 7, 7, 3, 1, 1>,
    // MobileNet width 1.3
    Problem<1, 40, 112, 112, 3, 1, 1>,
    Problem<1, 83, 112, 112, 3, 1, 2>,
    Problem<1, 168, 56, 56, 3, 1, 1>,
    Problem<1, 168, 56, 56, 3, 1, 2>,
    Problem<1, 336, 28, 28, 3, 1, 1>,
    Problem<1, 336, 28, 28, 3, 1, 2>,
    Problem<1, 664, 14, 14, 3, 1, 1>,
    Problem<1, 664, 14, 14, 3, 1, 2>,
    Problem<1, 1328, 7, 7, 3, 1, 1>>
    TestProblems;

TYPED_TEST_SUITE(DepthwiseTest, TestProblems);

TYPED_TEST(DepthwiseTest, CudaDepthwise) {
  const int kDimN = this->kDimN;
  const int kDimC = this->kDimC;
  const int kDimH = this->kDimH;
  const int kDimW = this->kDimW;
  const int kDimOutH = this->kDimOutH;
  const int kDimOutW = this->kDimOutW;
  const int kKernelSize = this->kKernelSize;
  const int kPadding = this->kPadding;
  const int kStride = this->kStride;

  Matrix in(kDimN * kDimC, kDimH * kDimW * 2, &this->generator_);
  Matrix filters(kDimC, kKernelSize * kKernelSize, &this->generator_);
  Matrix out(kDimN * kDimC, kDimOutH * kDimOutW, &this->generator_);
  CudaMatrix<float> in_gpu(in), filters_gpu(filters), out_gpu(out);

  // Run the kernel.
  CUDA_CALL(CudaDepthwise(kDimN, kDimC, kDimH, kDimW, in_gpu.Values(),
                          kKernelSize, kPadding, kStride, filters_gpu.Values(),
                          out_gpu.Values(),
                          /*stream=*/0));
  CUDA_CALL(cudaStreamSynchronize(/*stream=*/0));

  // Run the reference kernel.
  this->DepthwiseConvolution(in.Values(), filters.Values(), out.Values());

  Matrix results(out_gpu);
  auto comparator = Pointwise(NanSensitiveFloatNear(1e-05), ToVector(out));
  ASSERT_THAT(ToVector(results), comparator);
}

typedef std::function<cudaError_t(int, int, int, int, const float*, int, int,
                                  int, const float*, const float*, float*,
                                  cudaStream_t)>
    DepthwiseFn;

template <typename Problem>
void TestFn(const DepthwiseFn kFn, DepthwiseTest<Problem>* obj) {
  const int kDimN = obj->kDimN;
  const int kDimC = obj->kDimC;
  const int kDimH = obj->kDimH;
  const int kDimW = obj->kDimW;
  const int kDimOutH = obj->kDimOutH;
  const int kDimOutW = obj->kDimOutW;
  const int kKernelSize = obj->kKernelSize;
  const int kPadding = obj->kPadding;
  const int kStride = obj->kStride;

  Matrix in(kDimN * kDimC, kDimH * kDimW * 2, &obj->generator_);
  Matrix filters(kDimC, kKernelSize * kKernelSize, &obj->generator_);
  Matrix out(kDimN * kDimC, kDimOutH * kDimOutW, &obj->generator_);
  CudaMatrix<float> in_gpu(in), filters_gpu(filters), out_gpu(out);

  // Run the kernel.
  CUDA_CALL(kFn(kDimN, kDimC, kDimH, kDimW, in_gpu.Values(), kKernelSize,
                kPadding, kStride, filters_gpu.Values(),
                /*bias=*/nullptr, out_gpu.Values(),
                /*stream=*/0));
  CUDA_CALL(cudaStreamSynchronize(/*stream=*/0));

  // Run the reference kernel.
  obj->DepthwiseConvolution(in.Values(), filters.Values(), out.Values());

  Matrix results(out_gpu);
  auto comparator = Pointwise(NanSensitiveFloatNear(1e-05), ToVector(out));
  ASSERT_THAT(ToVector(results), comparator);
}

#define CONCAT_(x, y) x##y
#define CONCAT(x, y) CONCAT_(x, y)
#define ANONYMOUS_NAME(x) CONCAT(x, __COUNTER__)

#define REGISTER_TESTCASE(name, fn) \
  TYPED_TEST(DepthwiseTest, name) { TestFn(fn, this); }

#define REGISTER_TESTCASE_HELPER(tname, fn, bx, by, tx, ty) \
  const auto& tname = fn<bx, by, tx, ty>;                   \
  REGISTER_TESTCASE(fn##_##bx##x##by##x##tx##x##ty, tname);

#define REGISTER_TILED_TESTCASE(fn, bx, by, tx, ty) \
  REGISTER_TESTCASE_HELPER(ANONYMOUS_NAME(dwise), fn, bx, by, tx, ty);

REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 64, 64, 8, 8);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 64, 64, 4, 8);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 64, 64, 4, 4);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 32, 32, 4, 8);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 32, 32, 4, 4);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 32, 32, 4, 2);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 32, 32, 2, 2);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 16, 16, 4, 2);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 16, 16, 2, 4);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 16, 16, 2, 2);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 8, 8, 2, 1);
REGISTER_TILED_TESTCASE(CudaDepthwiseEx, 8, 8, 1, 2);

#undef REGISTER_TILED_TESTCASE
#undef REGISTER_TILED_TESTCASE_HELPER
#undef REGISTER_TESTCASE
#undef CONCAT_
#undef CONCAT
#undef ANONYMOUS_NAME

}  // namespace sputnik
