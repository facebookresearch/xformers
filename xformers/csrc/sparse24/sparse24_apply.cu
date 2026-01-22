#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/device.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include "pt_stable_utils.h"
#include "sparse24_metadata.h"
#include "sparse24_pack.h"

using namespace xformers::sp24;

namespace {

template <typename KT>
__global__ void __launch_bounds__(32 /* num_threads */)
    sparse24_apply_kernel(typename KT::Params p) {
  KT::sparse24_apply_kernel(p);
}

// Apply a 2:4 sparsify pattern computed with
// `sparse24_sparsify_both_ways_kernel` to another Tensor
template <typename Element, typename MetadataFormat, bool kIsMeta>
std::
    tuple<
        torch::stable::Tensor, // packed
        torch::stable::Tensor, // packed_meta_reordered
        torch::stable::Tensor, // packed_trans
        torch::stable::Tensor // packed_trans_meta_reordered
        >
    sparse24_apply_typed(
        torch::stable::Tensor input, // Tensor to sparsify
        torch::stable::Tensor
            threads_masks // Returned by `sparse24_sparsify_both_ways`
    ) {
  using KT = KernelTypes<Element>;
  // TODO: Technically we should be able to deal with that
  // by running on the transpose of `input` and swapping
  // `packed` & `packed_t`.
  // This would require to adapt the `threads_masks` a bit tho.
  if (input.stride(1) != 1) {
    input = xf_contiguous(input);
  }
  std::optional<torch::stable::accelerator::DeviceGuard> device_guard;
  if (!kIsMeta) {
    device_guard.emplace(input.device().index());
  }

  STD_TORCH_CHECK(input.dim() == 2);
  STD_TORCH_CHECK(input.stride(1) == 1);
  STD_TORCH_CHECK(input.stride(0) % 8 == 0);
  STD_TORCH_CHECK(input.size(1) % 32 == 0, "Wrong alignment shape[1]");

  auto rows = input.size(0);
  auto cols = input.size(1);

  auto [compressed, packed, packed_meta_reordered] =
      MetadataFormat::create_compressed_representation(
          rows, cols, input, false);
  auto [compressed_trans, packed_trans, packed_trans_meta_reordered] =
      MetadataFormat::create_compressed_representation(
          cols, rows, input, false);

  typename KT::Params p;
  p.input_s0 = input.stride(0);
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);

  p.packed_stride = packed.stride(0);
  p.packed_trans_stride = packed_trans.stride(0);

  if (!kIsMeta) {
    p.input = (Element const*)input.data_ptr();
    p.packed = (Element*)packed.data_ptr();
    p.packed_trans = (Element*)packed_trans.data_ptr();
    p.threads_masks = (uint64_t*)threads_masks.data_ptr();
  }

  STD_TORCH_CHECK(threads_masks.dim() == 3);
  STD_TORCH_CHECK(
      threads_masks.size(0) == p.getBlocksGrid().x * p.getThreadsGrid().x);
  STD_TORCH_CHECK(
      threads_masks.size(1) == p.getBlocksGrid().y * p.getThreadsGrid().y);
  STD_TORCH_CHECK(threads_masks.stride(1) == sizeof(p.threads_masks[0]));
  STD_TORCH_CHECK(threads_masks.size(2) == sizeof(p.threads_masks[0]));
  STD_TORCH_CHECK(threads_masks.stride(2) == 1);
  STD_TORCH_CHECK(
      threads_masks.scalar_type() == torch::headeronly::ScalarType::Byte);

  if (!kIsMeta) {
    size_t smem_bytes = 0;
    sparse24_apply_kernel<KT>
        <<<p.getBlocksGrid(),
           p.getThreadsGrid(),
           smem_bytes,
           xf_getCurrentCUDAStream()>>>(p);
    STD_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return std::make_tuple(
      compressed,
      packed_meta_reordered,
      compressed_trans,
      packed_trans_meta_reordered);
}

template <bool kIsMeta>
std::
    tuple<
        torch::stable::Tensor, // packed
        torch::stable::Tensor, // packed_meta_reordered
        torch::stable::Tensor, // packed_trans
        torch::stable::Tensor // packed_trans_meta_reordered
        >
    sparse24_apply(
        torch::stable::Tensor input, // Tensor to sparsify
        torch::stable::Tensor
            threads_masks, // Returned by `sparse24_sparsify_both_ways`
        std::string backend) {
  auto runTyped = [&](auto type) {
    using ElementT = decltype(type);
    if (backend == "cusparselt") {
      return sparse24_apply_typed<ElementT, MetadataCuSparseLtSm80, kIsMeta>(
          input, threads_masks);
    } else {
      STD_TORCH_CHECK(
          backend == "cutlass",
          "backend argument only supports `cutlass` or `cusparselt`");
      return sparse24_apply_typed<ElementT, MetadataCutlassSm80, kIsMeta>(
          input, threads_masks);
    }
  };

  if (input.scalar_type() == torch::headeronly::ScalarType::Half) {
    return runTyped(cutlass::half_t());
  } else {
    STD_TORCH_CHECK(
        input.scalar_type() == torch::headeronly::ScalarType::Half ||
        input.scalar_type() == torch::headeronly::ScalarType::BFloat16);
    return runTyped(cutlass::bfloat16_t());
  }
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl("sparse24_apply", TORCH_BOX(sparse24_apply<false>));
}

STABLE_TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl("sparse24_apply", TORCH_BOX(sparse24_apply<true>));
}
