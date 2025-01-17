/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
Implementation of the element_wise epilogue of the DualGemm kernel
with a custom activation function passed as a template parameter.
(third_party/cutlass/examples/45_dual_gemm/dual_gemm.cu)

DualGemm defined as:
  D0 = epilogue0(X @ B0, C0)
  D1 = epilogue1(X @ B1, C1)
  D2 = element_wise(D0, D1)

where element_wise(D0, D1) = eltwise_mul(activation_func(D0), D1)

Code from CUTLASS examples used as reference:
  third_party/cutlass/examples/45_dual_gemm/thread/left_silu_and_mul.h
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/functional.h"
#include "cutlass/numeric_conversion.h"

template <
    typename ElementOutput_, // Type used for load and store
    int Count, // Number of elements computed per operation
    template <typename> typename ActivationFn_, // Activation functor
    typename ElementAccumulator_ = ElementOutput_, // Accumulator type
    typename ElementCompute_ = ElementOutput_, // Type used for internal compute
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
class EpilogueLHSActivationAndMul {
 public:
  static int const kCount = Count;
  static cutlass::FloatRoundStyle const kRound = Round;

  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = ElementCompute_;
  using ActivationFnElementCompute = ActivationFn_<ElementCompute>;

  using FragmentOutput = cutlass::Array<ElementOutput, kCount>;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kCount>;
  using FragmentCompute = cutlass::Array<ElementCompute, kCount>;
  using ActivationFnFragmentCompute = ActivationFn_<FragmentCompute>;

  struct Params {};

 public:
  CUTLASS_HOST_DEVICE
  EpilogueLHSActivationAndMul(Params const& /*params*/) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    assert(false);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(
      FragmentAccumulator const& input_lhs,
      FragmentAccumulator const& input_rhs) const {
    cutlass::NumericArrayConverter<
        ElementCompute,
        ElementAccumulator,
        kCount,
        kRound>
        acc_to_compute;

    cutlass::
        NumericArrayConverter<ElementOutput, ElementCompute, kCount, kRound>
            compute_to_out;

    FragmentCompute casted_lhs = acc_to_compute(input_lhs);
    FragmentCompute casted_rhs = acc_to_compute(input_rhs);

    ActivationFnFragmentCompute activation_func;
    cutlass::multiplies<FragmentCompute> mul_func;

    auto activation_lhs_out = activation_func(casted_lhs);
    return compute_to_out(mul_func(activation_lhs_out, casted_rhs));
  }

  CUTLASS_HOST_DEVICE
  ElementOutput operator()(
      ElementAccumulator const& input_lhs,
      ElementAccumulator const& input_rhs) const {
    ElementCompute casted_lhs(input_lhs);
    ElementCompute casted_rhs(input_rhs);

    ActivationFnElementCompute activation_func;
    cutlass::multiplies<ElementCompute> mul_func;

    auto activation_lhs_out = activation_func(casted_lhs);
    return ElementOutput(mul_func(activation_lhs_out, casted_rhs));
  }
};
