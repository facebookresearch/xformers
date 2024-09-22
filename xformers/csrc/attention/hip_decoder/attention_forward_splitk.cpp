#include <ATen/Dispatch.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include "ck_attention_forward_decoder_splitk.h"

namespace {
constexpr int32_t kThreadsPerWavefront = 64;
constexpr int32_t kWavefrontsPerBlock = 4;
constexpr int32_t kMaxHeadDimension = 4 * kThreadsPerWavefront;
constexpr int32_t kMaxKVSequenceLength = 4096;
constexpr int32_t kLoopUnroll = 16;
constexpr int32_t kLoopUnrollTail = 2;
using compute_t = float;
} // namespace

namespace {

template <typename c10_t>
struct c10_to_data_t;
template <>
struct c10_to_data_t<float> {
  using type = float;
};

template <>
struct c10_to_data_t<c10::Half> {
  using type = ck::half_t;
};

template <>
struct c10_to_data_t<c10::BFloat16> {
  using type = ck::bhalf_t;
};
} // namespace

#define AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, ...) \
  AT_DISPATCH_CASE(SCALARTYPE1, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE2, __VA_ARGS__)                           \
  AT_DISPATCH_CASE(SCALARTYPE3, __VA_ARGS__)

#define AT_DISPATCH_SWITCH_3(                               \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      AT_DISPATCH_CASE_3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, __VA_ARGS__))

namespace {

template <
    int32_t ThreadsPerWavefront,
    int32_t WavefrontsPerBlock>
at::Tensor& efficient_attention_forward_decoder_splitk_ck_out_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int64_t split_k,
    at::Tensor& split_max,
    at::Tensor& split_sumexp,
    at::Tensor& split_O,
    at::Tensor& O) {
  static_assert(4 * ThreadsPerWavefront == kMaxHeadDimension, "");
  static_assert(WavefrontsPerBlock <= ThreadsPerWavefront, "");

  at::OptionalDeviceGuard guard(XQ.device());
  TORCH_CHECK(XQ.is_cuda());
  TORCH_CHECK(cache_K.is_cuda());
  TORCH_CHECK(cache_V.is_cuda());

  TORCH_CHECK(!seq_kv_lens || seq_kv_lens->is_cuda());

  TORCH_CHECK(cache_K.size(1) / split_k <= kMaxKVSequenceLength);
  TORCH_CHECK(cache_K.size(4) <= kMaxHeadDimension);

  constexpr auto rank = 5;

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);

  TORCH_CHECK(B <= 1024);
  TORCH_CHECK(M <= 1024);
  TORCH_CHECK(H <= 1024);

  dim3 blocks(B * H * M * G, split_k);
  dim3 threads(ThreadsPerWavefront, WavefrontsPerBlock);

  int32_t smem_softmax = kMaxKVSequenceLength * sizeof(compute_t) +
      WavefrontsPerBlock * sizeof(compute_t);
  int32_t smem_output = kMaxHeadDimension * sizeof(compute_t) *
      threads.y; // 4 * threadsPerBlock * sizeof(float) == sizeof(O[b][0][h][:])
  const size_t lds_bytes = max(smem_softmax, smem_output);
  auto stream = at::cuda::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float,
      XQ.scalar_type(),
      "efficient_attention_forward_decoder_splitk_ck",
      [&] {
        using ck_data_t = c10_to_data_t<scalar_t>::type;
        using device_op_t =
            ck::tensor_operation::device::FMHADecoderSplitKDeviceOp<
                ck_data_t,
                kMaxKVSequenceLength,
                kLoopUnroll,
                kLoopUnrollTail,
                compute_t>;
        auto op = device_op_t{};

        auto XQ_acc =
            XQ.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto K_acc =
            cache_K.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto V_acc =
            cache_V.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto split_O_acc =
            split_O
                .packed_accessor32<scalar_t, 1 + rank, at::RestrictPtrTraits>();
        auto O_acc =
            O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto seq_acc_ptr = seq_kv_lens
            ? seq_kv_lens
                  ->packed_accessor32<int32_t, 1, at::RestrictPtrTraits>()
                  .data()
            : nullptr;
        auto split_max_acc =
            split_max.packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto split_sumexp_acc =
            split_sumexp
                .packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto arg = device_op_t::Argument(
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(split_O_acc.data()),
            split_max_acc.data(),
            split_sumexp_acc.data(),
            seq_acc_ptr,
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            XQ_acc.stride(3),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.stride(3),
            split_O_acc.stride(0),
            XQ_acc.size(1),
            XQ_acc.size(2),
            XQ_acc.size(3),
            XQ_acc.size(4),
            K_acc.size(1),
            K_acc.size(3) == 1,
            qk_scale,
            split_k,
            blocks,
            threads,
            lds_bytes);

        auto invoker = device_op_t::Invoker{};
        (void)invoker.Run(&arg, {stream});
      });

  return O;
}

template <int32_t ThreadsPerWavefront, int32_t WavefrontsPerBlock>
at::Tensor efficient_attention_forward_decoder_splitk_ck_impl(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int64_t split_k) {
  auto O = at::empty_like(XQ);
  constexpr auto rank = 5;

  TORCH_CHECK(XQ.dim() == rank);
  TORCH_CHECK(cache_K.dim() == rank);
  TORCH_CHECK(cache_V.dim() == rank);

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);
  auto K = XQ.size(4);

  auto O_splits = at::empty({split_k, B, M, G, H, K}, XQ.options());
  auto split_max =
      at::empty({B, M, G, H, split_k}, XQ.options().dtype(at::kFloat));
  auto split_sumexp = at::empty_like(split_max);

  efficient_attention_forward_decoder_splitk_ck_out_impl<
      ThreadsPerWavefront,
      WavefrontsPerBlock>(
      XQ,
      cache_K,
      cache_V,
      seq_kv_lens,
      qk_scale,
      split_k,
      split_max,
      split_sumexp,
      O_splits,
      O);

  return O;
}

at::Tensor efficient_attention_forward_decoder_splitk_ck(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int64_t split_k) {
  return efficient_attention_forward_decoder_splitk_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(
      XQ, cache_K, cache_V, seq_kv_lens, qk_scale, split_k);
}
} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME(
          "xformers::efficient_attention_forward_decoder_splitk_ck"),
      TORCH_FN(efficient_attention_forward_decoder_splitk_ck));
}

#ifdef ATTN_FWD_SPLITK_DECODER_MAIN

#include <torch/torch.h>

// clang-format off

/*

(1) hipify
 > pip install -e /xformers

 For obtaining the executed build commands, add `--verbose`.
 For efficient utilization of CPU cores for compilation use MAX_JOBS env variable.

(2) compile
 > mkdir build
 > cd build
 > cmake /xformers/xformers/csrc/attention/hip_fmha/ \
       -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
       -D CMAKE_PREFIX_PATH=/opt/rocm \
       -D CMAKE_BUILD_TYPE=Debug \
       -D GPU_TARGETS="native" 
  > make

(3a) run correctness check
 > ./attention_forward_splitk_decoder_main

(3b) run specific input shape
 > ./attention_forward_splitk_decoder_main n_keys padding batch_size n_heads is_multiquery dtype n_wavefronts_per_block
*/

// clang-format on

static std::tuple<at::Tensor, at::Tensor, at::Tensor> split_attention_torch(
    const at::Tensor& Q,
    const at::Tensor& K,
    const at::Tensor& V,
    const at::Tensor& k_seqlens,
    const int32_t split_k,
    const int32_t block_size) {
  auto Q_scaled = at::div(Q, sqrt(Q.size(-1)));

  std::vector<at::Tensor> O_splits;
  std::vector<at::Tensor> m_splits;
  std::vector<at::Tensor> l_splits;

  for (int32_t split_idx = 0; split_idx < split_k; ++split_idx) {
    std::vector<at::Tensor> O_batch;
    std::vector<at::Tensor> m_batch;
    std::vector<at::Tensor> l_batch;

    for (size_t b = 0; b < k_seqlens.numel(); ++b) {
      auto seqlen = k_seqlens[b].item<int64_t>();
      const int64_t t_low =
          split_idx * (seqlen / split_k / block_size) * block_size;
      const int64_t t_high = (split_idx + 1 < split_k)
          ? (1 + split_idx) * (seqlen / split_k / block_size) * block_size
          : seqlen;

      const bool empty = t_low == t_high;

      auto S = at::einsum(
          "mghk, nghk -> mghn",
          {Q_scaled[b],
           at::slice(K[b], /*dim*/ 0, /*start*/ t_low, /*end*/ t_high)},
          /* einsum eval path */ at::nullopt);
      auto m = empty
          ? at::empty_like(S)
          : std::get<0>(at::max(S, /* dim */ -1, /* keepdim */ true));
      auto s = at::exp(at::sub(S, m));
      auto l = at::sum(s, /* dim */ -1, /* keepdim */ true);
      auto O = at::einsum(
          "mghn, nghk -> mghk",
          {s, at::slice(V[b], /*dim*/ 0, /*start*/ t_low, /*end*/ t_high)},
          /* einsum eval path */ at::nullopt);
      if (empty) {
        m = at::empty_like(at::slice(O, -1, 0, 1));
        l = at::zeros_like(m);
        m.fill_(ck::NumericLimits<float>::Lowest());
      }
      O_batch.push_back(O);
      m_batch.push_back(m);
      l_batch.push_back(l);
    }

    auto O_cat = at::stack(O_batch);
    auto m_cat = at::stack(m_batch);
    auto l_cat = at::stack(l_batch);

    O_splits.push_back(O_cat);
    m_splits.push_back(m_cat);
    l_splits.push_back(l_cat);
  }

  auto O_cat = at::stack(O_splits);
  auto m_cat = at::transpose(at::stack(m_splits), 0, -1);
  auto l_cat = at::transpose(at::stack(l_splits), 0, -1);

  return std::make_tuple(O_cat, m_cat, l_cat);
}

static at::Tensor split_reduce_torch(
    const at::Tensor& O_splits,
    const at::Tensor& m_splits,
    const at::Tensor& l_splits,
    int32_t split_k) {
  auto O = at::zeros_like(at::slice(O_splits, 0, 0, 1));
  auto global_max =
      at::empty_like(at::slice(m_splits, -1, 0, 1)).fill_(-65535.);
  auto global_sumexp = at::zeros_like(global_max);

  for (int32_t split_idx = 0; split_idx < split_k; ++split_idx) {
    auto local_O = at::slice(O_splits, 0, split_idx, split_idx + 1);
    auto local_max = at::slice(m_splits, -1, split_idx, split_idx + 1);
    auto local_sumexp = at::slice(l_splits, -1, split_idx, split_idx + 1);

    auto log_alpha = at::neg(at::abs(at::sub(local_max, global_max)));
    auto alpha = at::exp(log_alpha);
    alpha.nan_to_num_(1.);

    auto pick_new = at::less(local_max, global_max);
    auto pick_current_coef = at::where(pick_new, 1., alpha);
    auto pick_new_coef = at::where(pick_new, alpha, 1.);

    O = at::add(at::mul(pick_current_coef, O), at::mul(pick_new_coef, local_O));
    global_sumexp = at::add(
        at::mul(pick_current_coef, global_sumexp),
        at::mul(pick_new_coef, local_sumexp));
    global_max = at::max(local_max, global_max);
  }

  return at::div(O, global_sumexp);
}

static at::Tensor efficient_attention_forward_decoder_splitk_torch(
    const at::Tensor& XQ, // [B, 1, G, H, D]
    const at::Tensor& cache_K, // [B, kMaxKVSequenceLength, G, H or 1, D]
    const at::Tensor& cache_V, // [B, kMaxKVSequenceLength, G, H or 1, D]
    at::optional<at::Tensor> seq_kv_lens, // [B]
    double qk_scale,
    int32_t split_k,
    int32_t block_size) {
  auto [O_split, m, l] = split_attention_torch(
      XQ, cache_K, cache_V, *seq_kv_lens, split_k, block_size);
  auto O = split_reduce_torch(O_split, m, l, split_k);
  return O.reshape_as(XQ);
}

namespace ck {
namespace tensor_operation {
namespace device {

template <typename scalar_t, typename compute_t = float>
struct FMHADecoderSplitAttentionDeviceOp : public BaseOperator {
  using DeviceOp = FMHADecoderSplitAttentionDeviceOp;
  struct Argument : public BaseArgument {
    const scalar_t* __restrict__ XQ;
    const scalar_t* __restrict__ cache_K;
    const scalar_t* __restrict__ cache_V;
    scalar_t* __restrict__ O;
    scalar_t* __restrict__ split_O;
    compute_t* __restrict__ split_max;
    compute_t* __restrict__ split_sumexp;
    const int32_t* __restrict__ seq_kv_lens;
    const ptrdiff_t XQ_stride_b;
    const ptrdiff_t XQ_stride_m;
    const ptrdiff_t XQ_stride_g;
    const ptrdiff_t XQ_stride_h;
    const ptrdiff_t K_stride_b;
    const ptrdiff_t K_stride_m;
    const ptrdiff_t K_stride_g;
    const ptrdiff_t K_stride_h;
    const ptrdiff_t O_stride_split;
    const int32_t Q_size_m;
    const int32_t Q_size_g;
    const int32_t Q_size_h;
    const int32_t Q_size_k;
    const int32_t K_size_m;
    const bool multiquery;
    const float qk_scale;
    const int32_t split_k;

    const dim3 grid_dim;
    const dim3 block_dim;
    const size_t lds_bytes;

    Argument(
        const scalar_t* __restrict__ XQ,
        const scalar_t* __restrict__ cache_K,
        const scalar_t* __restrict__ cache_V,
        scalar_t* __restrict__ O,
        scalar_t* __restrict__ split_O,
        compute_t* __restrict__ split_max,
        compute_t* __restrict__ split_sumexp,
        const int32_t* __restrict__ seq_kv_lens,
        const ptrdiff_t XQ_stride_b,
        const ptrdiff_t XQ_stride_m,
        const ptrdiff_t XQ_stride_g,
        const ptrdiff_t XQ_stride_h,
        const ptrdiff_t K_stride_b,
        const ptrdiff_t K_stride_m,
        const ptrdiff_t K_stride_g,
        const ptrdiff_t K_stride_h,
        const ptrdiff_t O_stride_split,
        const int32_t Q_size_m,
        const int32_t Q_size_g,
        const int32_t Q_size_h,
        const int32_t Q_size_k,
        const int32_t K_size_m,
        const bool multiquery,
        const float qk_scale,
        const int32_t split_k,
        // launch params
        const dim3 grid_dim,
        const dim3 block_dim,
        const size_t lds_bytes)
        : XQ(XQ),
          cache_K(cache_K),
          cache_V(cache_V),
          O(O),
          split_O(split_O),
          split_max(split_max),
          split_sumexp(split_sumexp),
          seq_kv_lens(seq_kv_lens),
          XQ_stride_b(XQ_stride_b),
          XQ_stride_m(XQ_stride_m),
          XQ_stride_g(XQ_stride_g),
          XQ_stride_h(XQ_stride_h),
          K_stride_b(K_stride_b),
          K_stride_m(K_stride_m),
          K_stride_g(K_stride_g),
          K_stride_h(K_stride_h),
          O_stride_split(O_stride_split),
          Q_size_m(Q_size_m),
          Q_size_g(Q_size_g),
          Q_size_h(Q_size_h),
          Q_size_k(Q_size_k),
          K_size_m(K_size_m),
          multiquery(multiquery),
          qk_scale(qk_scale),
          split_k(split_k),
          // launch params
          grid_dim(grid_dim),
          block_dim(block_dim),
          lds_bytes(lds_bytes) {}

    std::string str() const {
      std::ostringstream oss;
      oss << "Argument { " << std::endl
          << "    XQ: " << XQ << std::endl
          << "    cache_K: " << cache_K << std::endl
          << "    cache_V: " << cache_V << std::endl
          << "    O: " << O << std::endl
          << "    split_O: " << split_O << std::endl
          << "    split_max: " << split_max << std::endl
          << "    split_sumexp: " << split_sumexp << std::endl
          << "    seq_kv_lens: " << seq_kv_lens << std::endl
          << "    XQ_stride_b: " << XQ_stride_b << std::endl
          << "    XQ_stride_m: " << XQ_stride_m << std::endl
          << "    XQ_stride_g: " << XQ_stride_g << std::endl
          << "    XQ_stride_h: " << XQ_stride_h << std::endl
          << "    K_stride_b: " << K_stride_b << std::endl
          << "    K_stride_m: " << K_stride_m << std::endl
          << "    K_stride_g: " << K_stride_g << std::endl
          << "    K_stride_h: " << K_stride_h << std::endl
          << "    O_stride_split: " << O_stride_split << std::endl
          << "    Q_size_m: " << Q_size_m << std::endl
          << "    Q_size_g: " << Q_size_g << std::endl
          << "    Q_size_h: " << Q_size_h << std::endl
          << "    Q_size_k: " << Q_size_k << std::endl
          << "    K_size_m: " << K_size_m << std::endl
          << "    multiquery: " << multiquery << std::endl
          << "    qk_scale: " << qk_scale << std::endl
          << "    split_k: " << split_k << std::endl
          << std::endl
          << "    grid_dim: " << grid_dim.x << "." << grid_dim.y << "."
          << grid_dim.z << std::endl
          << "    block_dim: " << block_dim.x << "." << block_dim.y << "."
          << block_dim.z << std::endl
          << "    lds_bytes: " << lds_bytes << std::endl
          << "}";
      return oss.str();
    }
  };

  struct Invoker : public BaseInvoker {
    using Argument = DeviceOp::Argument;
    float Run(
        const Argument& arg,
        const StreamConfig& stream_config = StreamConfig{}) {
      auto threads_per_wavefront = arg.block_dim.x;
      auto Q_size_k_alignment_necessary = 0;

      for (auto vec_size : {4, 2, 1}) {
        if (arg.Q_size_k <= vec_size * threads_per_wavefront) {
          Q_size_k_alignment_necessary = vec_size;
        }
      }

      if (!Q_size_k_alignment_necessary) {
        throw std::runtime_error("Unsupported Q_size_k");
      }

      if (arg.Q_size_k % Q_size_k_alignment_necessary) {
        throw std::runtime_error("Unsupported alignment for Q_size_k");
      }

      float split_attention_result = launch_and_time_kernel(
          stream_config,
          Q_size_k_alignment_necessary == 4
              ? efficient_attention_forward_decoder_splitk_ck_kernel<
                    scalar_t,
                    4,
                    kLoopUnroll,
                    kLoopUnrollTail,
                    kMaxKVSequenceLength,
                    compute_t>
              : Q_size_k_alignment_necessary == 2
              ? efficient_attention_forward_decoder_splitk_ck_kernel<
                    scalar_t,
                    2,
                    kLoopUnroll,
                    kLoopUnrollTail,
                    kMaxKVSequenceLength,
                    compute_t>
              : Q_size_k_alignment_necessary == 1
              ? efficient_attention_forward_decoder_splitk_ck_kernel<
                    scalar_t,
                    1,
                    kLoopUnroll,
                    kLoopUnrollTail,
                    kMaxKVSequenceLength,
                    compute_t>
              : nullptr,
          arg.grid_dim,
          arg.block_dim,
          arg.lds_bytes,
          arg.XQ,
          arg.cache_K,
          arg.cache_V,
          arg.split_O,
          arg.split_max,
          arg.split_sumexp,
          arg.seq_kv_lens,
          arg.XQ_stride_b,
          arg.XQ_stride_m,
          arg.XQ_stride_g,
          arg.XQ_stride_h,
          arg.K_stride_b,
          arg.K_stride_m,
          arg.K_stride_g,
          arg.K_stride_h,
          arg.O_stride_split,
          arg.Q_size_m,
          arg.Q_size_g,
          arg.Q_size_h,
          arg.Q_size_k,
          arg.K_size_m,
          arg.multiquery,
          arg.qk_scale,
          arg.split_k);

      return split_attention_result;
    }
  };
};

template <typename scalar_t, typename compute_t = float>
struct FMHADecoderSplitReduceDeviceOp : public BaseOperator {
  using DeviceOp = FMHADecoderSplitReduceDeviceOp;
  struct Argument : public BaseArgument {
    const scalar_t* __restrict__ split_O;
    const compute_t* __restrict__ split_max;
    const compute_t* __restrict__ split_sumexp;
    scalar_t* __restrict__ O;

    const int32_t O_size_m;
    const int32_t O_size_g;
    const int32_t O_size_h;
    const int32_t O_size_k;

    const ptrdiff_t O_stride_split;
    const ptrdiff_t O_stride_b;
    const ptrdiff_t O_stride_m;
    const ptrdiff_t O_stride_g;
    const ptrdiff_t O_stride_h;

    const int32_t split_k;

    const dim3 grid_dim;
    const dim3 block_dim;
    const size_t lds_bytes;

    Argument(
        const scalar_t* __restrict__ split_O,
        const compute_t* __restrict__ split_max,
        const compute_t* __restrict__ split_sumexp,
        scalar_t* __restrict__ O,
        const int32_t O_size_m,
        const int32_t O_size_g,
        const int32_t O_size_h,
        const int32_t O_size_k,
        const ptrdiff_t O_stride_split,
        const ptrdiff_t O_stride_b,
        const ptrdiff_t O_stride_m,
        const ptrdiff_t O_stride_g,
        const ptrdiff_t O_stride_h,
        const int32_t split_k,
        // launch params
        const dim3 grid_dim,
        const dim3 block_dim,
        const size_t lds_bytes)
        : split_O(split_O),
          split_max(split_max),
          split_sumexp(split_sumexp),
          O(O),
          O_size_m(O_size_m),
          O_size_g(O_size_g),
          O_size_h(O_size_h),
          O_size_k(O_size_k),
          O_stride_split(O_stride_split),
          O_stride_b(O_stride_b),
          O_stride_m(O_stride_m),
          O_stride_g(O_stride_g),
          O_stride_h(O_stride_h),
          split_k(split_k),
          // launch params
          grid_dim(grid_dim),
          block_dim(block_dim),
          lds_bytes(lds_bytes) {}

    std::string str() const {
      std::ostringstream oss;
      oss << "Argument { " << std::endl
          << "    O: " << O << std::endl
          << "    split_O: " << split_O << std::endl
          << "    split_max: " << split_max << std::endl
          << "    split_sumexp: " << split_sumexp << std::endl
          << "    O_stride_b: " << O_stride_b << std::endl
          << "    O_stride_m: " << O_stride_m << std::endl
          << "    O_stride_g: " << O_stride_g << std::endl
          << "    O_stride_h: " << O_stride_h << std::endl
          << "    O_stride_split: " << O_stride_split << std::endl
          << "    O_size_m: " << O_size_m << std::endl
          << "    O_size_g: " << O_size_g << std::endl
          << "    O_size_h: " << O_size_h << std::endl
          << "    O_size_k: " << O_size_k << std::endl
          << "    split_k: " << split_k << std::endl
          << std::endl
          << "    grid_dim: " << grid_dim.x << "." << grid_dim.y << "."
          << grid_dim.z << std::endl
          << "    block_dim: " << block_dim.x << "." << block_dim.y << "."
          << block_dim.z << std::endl
          << "    lds_bytes: " << lds_bytes << std::endl
          << "}";
      return oss.str();
    }
  };

  struct Invoker : public BaseInvoker {
    using Argument = DeviceOp::Argument;
    float Run(
        const Argument& arg,
        const StreamConfig& stream_config = StreamConfig{}) {
      auto threads_per_wavefront = arg.block_dim.x;
      auto O_size_k_alignment_necessary = 0;

      for (auto vec_size : {4, 2, 1}) {
        if (arg.O_size_k <= vec_size * threads_per_wavefront) {
          O_size_k_alignment_necessary = vec_size;
        }
      }

      if (!O_size_k_alignment_necessary) {
        throw std::runtime_error("Unsupported O_size_k");
      }

      if (arg.O_size_k % O_size_k_alignment_necessary) {
        throw std::runtime_error("Unsupported alignment for O_size_k");
      }

      const dim3 reduce_gridsize = {arg.grid_dim.x};
      const dim3 reduce_blocksize = {arg.block_dim.x};
      constexpr int32_t reduce_lds_bytes = 0;
      float reduce_result = launch_and_time_kernel(
          stream_config,
          O_size_k_alignment_necessary == 4
              ? efficient_attention_forward_decoder_splitk_reduce_ck_kernel<
                    scalar_t,
                    4>
              : O_size_k_alignment_necessary == 2
              ? efficient_attention_forward_decoder_splitk_reduce_ck_kernel<
                    scalar_t,
                    2>
              : O_size_k_alignment_necessary == 1
              ? efficient_attention_forward_decoder_splitk_reduce_ck_kernel<
                    scalar_t,
                    1>
              : nullptr,
          reduce_gridsize,
          reduce_blocksize,
          reduce_lds_bytes,
          arg.split_O,
          arg.split_max,
          arg.split_sumexp,
          arg.O,
          arg.O_size_m,
          arg.O_size_g,
          arg.O_size_h,
          arg.O_size_k,
          arg.O_stride_split,
          arg.O_stride_b,
          arg.O_stride_m,
          arg.O_stride_g,
          arg.O_stride_h,
          arg.split_k);
      return reduce_result;
    }
  };
};
} // namespace device
} // namespace tensor_operation
} // namespace ck

static std::tuple<at::Tensor, at::Tensor, at::Tensor> split_attention_hip(
    const at::Tensor& XQ,
    const at::Tensor& K,
    const at::Tensor& V,
    const at::Tensor& seqlen,
    const int32_t split_k,
    const int32_t wavefronts_per_block) {
  at::OptionalDeviceGuard guard(XQ.device());

  auto B = XQ.size(0);
  auto M = XQ.size(1);
  auto G = XQ.size(2);
  auto H = XQ.size(3);
  auto D = XQ.size(4);

  double qk_scale = 1. / sqrt(D);

  auto O = at::empty_like(XQ);
  constexpr auto rank = 5;
  auto split_O = at::zeros({split_k, B, M, G, H, D}, XQ.options());
  auto split_max =
      at::empty({B, M, G, H, split_k}, XQ.options().dtype(at::kFloat))
          .fill_(ck::NumericLimits<float>::Lowest());
  auto split_sumexp = at::zeros_like(split_max);

  dim3 blocks(B * H * M * G, split_k);
  dim3 threads(kThreadsPerWavefront, wavefronts_per_block);

  int32_t smem_softmax =
      kMaxKVSequenceLength * sizeof(float) + threads.y * sizeof(float);
  int32_t smem_output = kMaxHeadDimension * sizeof(float) *
      wavefronts_per_block; // 4 * threadsPerBlock * sizeof(float) ==
                            // sizeof(O[b][0][h][:])
  const size_t lds_bytes = max(smem_softmax, smem_output);
  auto stream = at::cuda::getCurrentHIPStream().stream();

  AT_DISPATCH_SWITCH_3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float,
      XQ.scalar_type(),
      "efficient_attention_forward_decoder_split_attention_ck_test",
      [&] {
        using ck_data_t = c10_to_data_t<scalar_t>::type;
        using device_op_t =
            ck::tensor_operation::device::FMHADecoderSplitAttentionDeviceOp<
                ck_data_t>;
        auto op = device_op_t{};

        auto XQ_acc =
            XQ.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto K_acc =
            K.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto V_acc =
            V.packed_accessor64<scalar_t, rank, at::RestrictPtrTraits>();
        auto split_O_acc =
            split_O
                .packed_accessor32<scalar_t, 1 + rank, at::RestrictPtrTraits>();
        auto O_acc =
            O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto seq_acc =
            seqlen.packed_accessor32<int32_t, 1, at::RestrictPtrTraits>();
        auto split_max_acc =
            split_max.packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto split_sumexp_acc =
            split_sumexp
                .packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto arg = device_op_t::Argument(
            reinterpret_cast<const ck_data_t* __restrict__>(XQ_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(K_acc.data()),
            reinterpret_cast<const ck_data_t* __restrict__>(V_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            reinterpret_cast<ck_data_t* __restrict__>(split_O_acc.data()),
            split_max_acc.data(),
            split_sumexp_acc.data(),
            seq_acc.data(),
            XQ_acc.stride(0),
            XQ_acc.stride(1),
            XQ_acc.stride(2),
            XQ_acc.stride(3),
            K_acc.stride(0),
            K_acc.stride(1),
            K_acc.stride(2),
            K_acc.stride(3),
            split_O_acc.stride(0),
            XQ_acc.size(1),
            XQ_acc.size(2),
            XQ_acc.size(3),
            XQ_acc.size(4),
            K_acc.size(1),
            K_acc.size(3) == 1,
            qk_scale,
            split_k,
            blocks,
            threads,
            lds_bytes);

        auto invoker = device_op_t::Invoker{};
        (void)invoker.Run(arg, {stream});
      });
  return std::make_tuple(split_O, split_max, split_sumexp);
}

static at::Tensor split_reduce_hip(
    const at::Tensor& split_O,
    const at::Tensor& split_max,
    const at::Tensor& split_sumexp,
    const int32_t split_k) {
  at::OptionalDeviceGuard guard(split_O.device());

  auto B = split_O.size(1);
  auto M = split_O.size(2);
  auto G = split_O.size(3);
  auto H = split_O.size(4);
  auto D = split_O.size(5);

  TORCH_CHECK_EQ(split_k, split_O.size(0));
  TORCH_CHECK_EQ(split_k, split_max.size(-1));
  TORCH_CHECK_EQ(split_k, split_sumexp.size(-1));

  constexpr auto rank = 5;

  TORCH_CHECK_EQ(split_O.dim(), 1 + rank);
  TORCH_CHECK_EQ(split_max.dim(), rank);
  TORCH_CHECK_EQ(split_sumexp.dim(), rank);

  auto O = at::zeros({B, M, G, H, D}, split_O.options());

  auto stream = at::cuda::getCurrentHIPStream().stream();
  auto lds_bytes = 0;

  dim3 blocks(B * H * M * G);
  dim3 threads(kThreadsPerWavefront);

  AT_DISPATCH_SWITCH_3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float,
      O.scalar_type(),
      "efficient_attention_forward_decoder_split_reduce_ck_test",
      [&] {
        using ck_data_t = c10_to_data_t<scalar_t>::type;
        using device_op_t =
            ck::tensor_operation::device::FMHADecoderSplitReduceDeviceOp<
                ck_data_t>;
        auto op = device_op_t{};

        auto split_O_acc =
            split_O
                .packed_accessor32<scalar_t, 1 + rank, at::RestrictPtrTraits>();
        auto O_acc =
            O.packed_accessor32<scalar_t, rank, at::RestrictPtrTraits>();
        auto split_max_acc =
            split_max.packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto split_sumexp_acc =
            split_sumexp
                .packed_accessor32<float, rank, at::RestrictPtrTraits>();
        auto arg = device_op_t::Argument(
            reinterpret_cast<const ck_data_t* __restrict__>(split_O_acc.data()),
            split_max_acc.data(),
            split_sumexp_acc.data(),
            reinterpret_cast<ck_data_t* __restrict__>(O_acc.data()),
            O_acc.size(1),
            O_acc.size(2),
            O_acc.size(3),
            O_acc.size(4),
            split_O_acc.stride(0),
            O_acc.stride(0),
            O_acc.stride(1),
            O_acc.stride(2),
            O_acc.stride(3),
            split_k,
            blocks,
            threads,
            lds_bytes);

        auto invoker = device_op_t::Invoker{};
        (void)invoker.Run(arg, {stream});
      });
  return O;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> generate_inputs(
    const int32_t padding,
    const int32_t B,
    const int32_t Hq,
    const int32_t Hkv,
    const decltype(torch::kFloat32) dtype = torch::kFloat32) {
  const int32_t D = 4 * kThreadsPerWavefront;
  const int32_t G = Hq / Hkv;
  const int32_t num_queries = 1;

  at::manual_seed(1);

  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .layout(torch::kStrided)
                     .device(torch::kCUDA, 1)
                     .requires_grad(false);
  auto int_options = options.dtype(torch::kInt);
  auto XQ = at::randn({B, num_queries, G, Hq, D}, options);
  auto K = (G == 1) ? at::randn({B, padding, G, Hkv, D}, options)
                    : at::randn({B, padding, G, 1, D}, options)
                          .expand({B, padding, G, Hq, D});
  auto V = at::randn_like(K);
  auto seqlen = at::randint(num_queries, padding + 1, {B}, int_options);

  return std::make_tuple(XQ, K, V, seqlen);
}

static float percent_mismatch(const at::Tensor& a, const at::Tensor& b) {
  auto mask =
      at::isclose(a, b, /*atol*/ 1e-3, /*rtol*/ 1e-5, /*equal_nan*/ false);
  auto percent_match = at::sum(mask.to(torch::kFloat32)) / mask.numel();
  return 1. - percent_match.item<float>();
}

static void test_split_attention(
    int32_t padding,
    int32_t batch_size,
    int32_t Hq,
    int32_t Hkv,
    int32_t split_k) {
  auto [XQ, K, V, seqlen] = generate_inputs(padding, batch_size, Hq, Hkv);

  auto [O_ref, m_ref, l_ref] = split_attention_torch(
      XQ, K, V, seqlen, split_k, /* block_size */ kWavefrontsPerBlock * 16);

  auto [O_hip, m_hip, l_hip] =
      split_attention_hip(XQ, K, V, seqlen, split_k, kWavefrontsPerBlock);

  auto O_percent_mismatch = percent_mismatch(O_ref, O_hip);
  auto m_percent_mismatch = percent_mismatch(m_ref, m_hip);
  auto l_percent_mismatch = percent_mismatch(l_ref, l_hip);

  printf(
      "[Test split attention] Padding=%d BS=%d Hq=%d Hkv=%d split_k=%d Mismatched split_O "
      "elements percentage: %.2f Mismatched split_max elements percentage: %.2f Mismatched "
      "split_sumexp elements percentage: %.2f\n",
      padding,
      batch_size,
      Hq,
      Hkv,
      split_k,
      O_percent_mismatch,
      m_percent_mismatch,
      l_percent_mismatch);
}

static void test_split_reduce(
    int32_t padding,
    int32_t batch_size,
    int32_t Hq,
    int32_t Hkv,
    int32_t split_k) {
  auto [XQ, K, V, seqlen] = generate_inputs(padding, batch_size, Hq, Hkv);

  auto [O_ref, m_ref, l_ref] =
      split_attention_hip(XQ, K, V, seqlen, split_k, kWavefrontsPerBlock);

  auto O_torch = split_reduce_torch(
      O_ref, m_ref.unsqueeze(0), l_ref.unsqueeze(0), split_k);
  auto O_hip = split_reduce_hip(O_ref, m_ref, l_ref, split_k);

  auto hip_torch_mismatch = percent_mismatch(O_hip, O_torch);
  printf(
      "[Test split reduce] Padding=%d BS=%d Hq=%d Hkv=%d split_k=%d Mismatched elements "
      "percentage: %.2f \n",
      padding,
      batch_size,
      Hq,
      Hkv,
      split_k,
      hip_torch_mismatch);
}

static void test_splitk_decoder_e2e_correctness(
    int32_t padding,
    int32_t batch_size,
    int32_t Hq,
    int32_t Hkv,
    int32_t split_k) {
  auto [XQ, K, V, seqlen] = generate_inputs(padding, batch_size, Hq, Hkv);

  double qk_scale = 1. / sqrt(XQ.size(-1));

  auto result = efficient_attention_forward_decoder_splitk_ck_impl<
      kThreadsPerWavefront,
      kWavefrontsPerBlock>(XQ, K, V, seqlen, qk_scale, split_k);
  auto gold_result = efficient_attention_forward_decoder_splitk_torch(
      XQ, K, V, seqlen, qk_scale, /* split_k */ 1, /* block_size */ 1);
  auto e2e_mismatch = percent_mismatch(result, gold_result);
  printf(
      "[Test e2e split-k decoder] Padding=%d BS=%d Hq=%d Hkv=%d split_k=%d Mismatched "
      "elements percentage: %.2f\n",
      padding,
      batch_size,
      Hq,
      Hkv,
      split_k,
      e2e_mismatch);
}

int main(int argc, char** argv) {
  if (argc == 1) {
    for (auto padding : {32, 4096}) {
      for (auto batch_size : {1, 8}) {
        for (auto Hq : {16}) {
          for (auto Hkv : {16}) {
            for (auto split_k : {1, 2, 4, 8, 16}) {
              test_splitk_decoder_e2e_correctness(
                  padding, batch_size, Hq, Hkv, split_k);
            }
          }
        }
      }
    }

    for (auto padding : {32, 4096}) {
      for (auto batch_size : {1, 8}) {
        for (auto Hq : {16}) {
          for (auto Hkv : {16}) {
            for (auto split_k : {1, 2, 4, 8, 16}) {
              test_split_attention(padding, batch_size, Hq, Hkv, split_k);
            }
          }
        }
      }
    }

    for (auto padding : {32, 4096}) {
      for (auto batch_size : {1, 8}) {
        for (auto Hq : {16}) {
          for (auto Hkv : {16}) {
            for (auto split_k : {1, 2}) {
              test_split_reduce(padding, batch_size, Hq, Hkv, split_k);
            }
          }
        }
      }
    }
  } else {
    const auto args = std::vector<std::string>(argv + 1, argv + argc);
    if (args.size() != 6) {
      std::cout << "Usage: ./a.out padding batch_size nq_heads nkv_heads dtype "
                   "n_wavefronts_per_block"
                << std::endl;
      return 0;
    }
    const int32_t padding = std::stoi(args[0]);
    const int32_t batch_size = std::stoi(args[1]);
    const int32_t nq_heads = std::stoi(args[2]);
    const int32_t nkv_heads = std::stoi(args[3]);
    const auto dtype = (args[4] == "f32") ? torch::kFloat32
        : (args[4] == "f16")              ? torch::kFloat16
                                          : torch::kBFloat16;
    const int32_t n_wavefronts_per_block = std::stoi(args[5]);

    auto [Q, K, V, seq] =
        generate_inputs(padding, batch_size, nq_heads, nkv_heads, dtype);
    auto O = at::empty_like(Q);

    constexpr auto splitk_dim = 0;
    constexpr auto split_k = 1;
    auto O_splits = at::stack(O, splitk_dim);

    auto split_max = at::empty(
        {batch_size, padding, Q.size(2), Q.size(3), split_k},
        Q.options().dtype(at::kFloat));
    auto split_sumexp = at::empty_like(split_max);

    const double qk_scale = 1. / sqrt(Q.size(-1));
    auto call_ptr =
        decltype(&efficient_attention_forward_decoder_splitk_ck_out_impl<
                 kThreadsPerWavefront,
                 kWavefrontsPerBlock>){};

#define SWITCH_CASE_SET_CALLPTR(n)                                      \
  case (n):                                                             \
    call_ptr = &efficient_attention_forward_decoder_splitk_ck_out_impl< \
        kThreadsPerWavefront,                                           \
        (n)>;                                                           \
    break;

    switch (n_wavefronts_per_block) {
      SWITCH_CASE_SET_CALLPTR(1);
      SWITCH_CASE_SET_CALLPTR(2);
      SWITCH_CASE_SET_CALLPTR(4);
      SWITCH_CASE_SET_CALLPTR(8);
      SWITCH_CASE_SET_CALLPTR(16);

      default:
        call_ptr = nullptr;
        break;
    }
#undef SWITCH_CASE_SET_CALLPTR

    if (call_ptr) {
      call_ptr(
          Q,
          K,
          V,
          seq,
          qk_scale,
          split_k,
          split_max,
          split_sumexp,
          O_splits,
          O);
    } else {
      std::cout << "Warning: no kernel was found for wavefronts_per_block="
                << n_wavefronts_per_block << std::endl;
    }
  }
  return 0;
}

#endif // MAIN

#undef AT_DISPATCH_CASE_3
#undef AT_DISPATCH_SWITCH_3
