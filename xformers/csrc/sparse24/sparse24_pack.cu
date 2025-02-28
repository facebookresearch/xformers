#include <ATen/ScalarOps.h>
#include <ATen/Tensor.h>
#include <ATen/autocast_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/types.h>
#include "compute_sparse_tile.h"
#include "sparse24_metadata.h"
#include "sparse24_pack.h"

using namespace xformers::sp24;

namespace {
template <typename KT, typename Metadata, typename Algorithm>
__global__ void __launch_bounds__(32 /* num_threads */, 20)
    sparse24_sparsify_both_ways_kernel(
        typename KT::Params p,
        Metadata metadata,
        Algorithm algo) {
  KT::sparse24_sparsify_both_ways_kernel(p, metadata, algo);
}

template <typename Element, typename MetadataFormat, bool kIsMeta>
std::
    tuple<
        at::Tensor, // packed
        at::Tensor, // packed_meta_reordered
        at::Tensor, // packed_trans
        at::Tensor, // packed_trans_meta_reordered
        at::Tensor // threads_masks
        >
    sparse24_sparsify_both_ways_typed(
        const at::Tensor input,
        std::string algorithm) {
  using KT = KernelTypes<Element>;
  std::optional<at::cuda::CUDAGuard> device_guard;
  if (!kIsMeta) {
    device_guard.emplace(input.device());
  }

  TORCH_CHECK(input.dim() == 2, "Can only sparsify 2d tensors");
  TORCH_CHECK(
      input.stride(1) == 1,
      "Can only sparsify contiguous tensors. Sparsify the transpose otherwise.");

  auto rows = input.size(0);
  auto cols = input.size(1);

  auto [compressed, packed, packed_meta_reordered] =
      MetadataFormat::create_compressed_representation(rows, cols, input, true);
  auto [compressed_trans, packed_trans, packed_trans_meta_reordered] =
      MetadataFormat::create_compressed_representation(cols, rows, input, true);
  TORCH_CHECK(
      input.size(1) % 32 == 0, "Number of cols should be multiple of 32");

  typename KT::Params p;
  p.input_s0 = input.stride(0);
  p.input_dim0 = input.size(0);
  p.input_dim1 = input.size(1);
  p.packed_stride = packed.stride(0);
  p.packed_trans_stride = packed_trans.stride(0);

  MetadataFormat metadata = MetadataFormat(
      packed_meta_reordered, packed_trans_meta_reordered, rows, cols);
  at::Tensor threads_masks = at::empty(
      {p.getBlocksGrid().x * p.getThreadsGrid().x,
       p.getBlocksGrid().y * p.getThreadsGrid().y,
       sizeof(p.threads_masks[0])},
      input.options().dtype(at::ScalarType::Byte));
  if (!kIsMeta) {
    p.input = (Element const*)input.data_ptr();
    p.packed = (Element*)packed.data_ptr();
    p.packed_trans = (Element*)packed_trans.data_ptr();
    p.threads_masks = (uint64_t*)threads_masks.data_ptr();
  }

  bool kernel_launched = false;
  auto launchKernel = [&](auto algo, std::string const& algo_name) {
    if (algo_name == algorithm) {
      kernel_launched = true;
      if (kIsMeta) {
        return;
      }
      size_t smem_bytes = 0;
      sparse24_sparsify_both_ways_kernel<KT>
          <<<p.getBlocksGrid(),
             p.getThreadsGrid(),
             smem_bytes,
             at::cuda::getCurrentCUDAStream()>>>(p, metadata, algo);
    }
  };
  named_algorithms(launchKernel);
  TORCH_CHECK(kernel_launched, "Unknown algorithm \"", algorithm, "\"");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(
      compressed,
      packed_meta_reordered,
      compressed_trans,
      packed_trans_meta_reordered,
      threads_masks);
}

template <bool kIsMeta = false>
std::
    tuple<
        at::Tensor, // packed
        at::Tensor, // packed_meta_reordered
        at::Tensor, // packed_trans
        at::Tensor, // packed_trans_meta_reordered
        at::Tensor // threads_masks
        >
    sparse24_sparsify_both_ways(
        const at::Tensor input,
        std::string algorithm,
        std::string backend) {
  auto runTyped = [&](auto type) {
    using ElementT = decltype(type);
    if (backend == "cusparselt") {
      return sparse24_sparsify_both_ways_typed<
          ElementT,
          MetadataCuSparseLtSm80,
          kIsMeta>(input, algorithm);
    } else {
      TORCH_CHECK(
          backend == "cutlass",
          "backend argument only supports `cutlass` or `cusparselt`");
      return sparse24_sparsify_both_ways_typed<
          ElementT,
          MetadataCutlassSm80,
          kIsMeta>(input, algorithm);
    }
  };

  if (input.scalar_type() == at::ScalarType::Half) {
    return runTyped(cutlass::half_t());
  } else {
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::Half ||
        input.scalar_type() == at::ScalarType::BFloat16);
    return runTyped(cutlass::bfloat16_t());
  }
}

std::
    tuple<
        at::Tensor, // packed
        at::Tensor, // packed_meta_reordered
        at::Tensor, // packed_trans
        at::Tensor, // packed_trans_meta_reordered
        at::Tensor // threads_masks
        >
    sparse24_sparsify_both_ways_autocast(
        const at::Tensor input,
        std::string algorithm,
        std::string backend) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::Autocast);
  auto exec_type = at::autocast::get_autocast_dtype(at::kCUDA);
  return sparse24_sparsify_both_ways(
      at::autocast::cached_cast(exec_type, input), algorithm, backend);
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse24_sparsify_both_ways"),
      TORCH_FN(sparse24_sparsify_both_ways<false>));
}

TORCH_LIBRARY_IMPL(xformers, Meta, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse24_sparsify_both_ways"),
      TORCH_FN(sparse24_sparsify_both_ways<true>));
}

TORCH_LIBRARY_IMPL(xformers, Autocast, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::sparse24_sparsify_both_ways"),
      TORCH_FN(sparse24_sparsify_both_ways_autocast));
}
