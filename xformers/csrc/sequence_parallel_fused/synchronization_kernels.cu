#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <cuda/std/chrono>

// CUDA atomics are only supported for sm_60+ on *nix and sm_70+ on Windows.
#define CUDA_ARCH_SUPPORTS_ATOMICS                    \
  (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700 || \
   (!defined(_MSC_VER) && __CUDA_ARCH__ >= 600))
#if CUDA_ARCH_SUPPORTS_ATOMICS
#include <cuda/std/atomic>
#include <cuda/std/version>
// cuda::atomic_ref is only available in libcudacxx 1.7.0+ (which corresponds to
// CUDA 11.6)
#define LIBCUDACXX_PROVIDES_ATOMIC_REF \
  (_LIBCUDACXX_CUDA_API_VERSION >= 001007000)
#endif

#if defined(_MSC_VER)

#if defined(NDEBUG)
extern "C" {
#if defined(__CUDA_ARCH__)
__host__ __device__
#endif // __CUDA_ARCH__
    void
    _wassert(wchar_t const* _Message, wchar_t const* _File, unsigned _Line);
}
#endif // NDEBUG

#define KERNEL_FAIL_ASSERT(message)            \
  (void)(_wassert(                             \
             _CRT_WIDE(message),               \
             _CRT_WIDE(__FILE__),              \
             static_cast<unsigned>(__LINE__)), \
         0);

#else // _MSC_VER

#if defined(NDEBUG)
extern "C" {
#if (defined(__CUDA_ARCH__) && !(defined(__clang__) && defined(__CUDA__)))
__host__ __device__
#endif // __CUDA_ARCH__
    void
    __assert_fail(
        const char* assertion,
        const char* file,
        unsigned int line,
        const char* function) throw() __attribute__((__noreturn__));
}
#endif // NDEBUG

#define KERNEL_FAIL_ASSERT(message) \
  __assert_fail(                    \
      message, __FILE__, static_cast<unsigned int>(__LINE__), __func__);

#endif // _MSC_VER

namespace {

constexpr int kMaxWorldSize = 8;

constexpr int kNumSpinsBetweenTimeoutChecks = 1000;

__device__ uint64_t getNsSinceEpoch() {
  return cuda::std::chrono::duration_cast<cuda::std::chrono::nanoseconds>(
             cuda::std::chrono::system_clock::now().time_since_epoch())
      .count();
}

#if CUDA_ARCH_SUPPORTS_ATOMICS
#if LIBCUDACXX_PROVIDES_ATOMIC_REF
class Atomic {
 public:
  __device__ explicit Atomic(int* ptr) : ref_(*ptr) {}

  __device__ int load() {
    return ref_.load(cuda::std::memory_order_acquire);
  }

  __device__ void store(int val) {
    ref_.store(val, cuda::std::memory_order_release);
  }

 private:
  cuda::atomic_ref<int, cuda::thread_scope_system> ref_;
};
#else // LIBCUDACXX_PROVIDES_ATOMIC_REF
class Atomic {
 public:
  __device__ explicit Atomic(int* ptr)
      : ref_(*reinterpret_cast<cuda::atomic<int, cuda::thread_scope_system>*>(
            ptr)) {}

  __device__ int load() {
    return ref_.load(cuda::std::memory_order_acquire);
  }

  __device__ void store(int val) {
    ref_.store(val, cuda::std::memory_order_release);
  }

 private:
  cuda::atomic<int, cuda::thread_scope_system>& ref_;
};
#endif // LIBCUDACXX_PROVIDES_ATOMIC_REF
#else // CUDA_ARCH_SUPPORTS_ATOMICS
// For architectures that don't support atomics, fall back to less safe load and
// store that bypasses the L1 cache and only stores in the global L2 cache. This
// is copied from CUTLASS's semaphore.
class Atomic {
 public:
  __device__ explicit Atomic(int* ptr) : ptr_(ptr) {}

  __device__ int load() {
    int val;
    asm volatile("ld.global.cg.b32 %0, [%1];\n" : "=r"(val) : "l"(ptr_));
    return val;
  }

  __device__ void store(int val) {
    asm volatile("st.global.cg.b32 [%0], %1;\n" : : "l"(ptr_), "r"(val));
  }

 private:
  int* ptr_;
};
#endif // CUDA_ARCH_SUPPORTS_ATOMICS

__global__ void write_values_kernel(
    const std::array<int*, kMaxWorldSize> ptrs,
    size_t numPtrs,
    int seqNum) {
  assert(blockIdx.x == 0);
  assert(blockIdx.y == 0);
  assert(blockIdx.z == 0);
  assert(threadIdx.x == 0);
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);

  for (int i = 0; i < numPtrs; i += 1) {
    Atomic atomic(ptrs[i]);
    // We use the "release" memory order because a successful store signals that
    // we've completed an operation on some memory and we're transferring
    // control of it to some consumer, hence we want all our operations to be
    // flushed and visible by the time we do the store.
    atomic.store(seqNum);
  }
}

__global__ void wait_values_kernel(
    const std::array<int*, kMaxWorldSize> ptrs,
    size_t numPtrs,
    int seqNum,
    uint64_t timeoutNs) {
  assert(blockIdx.x == 0);
  assert(blockIdx.y == 0);
  assert(blockIdx.z == 0);
  assert(threadIdx.x == 0);
  assert(threadIdx.y == 0);
  assert(threadIdx.z == 0);

  uint64_t startTimeNs = getNsSinceEpoch();

  for (int i = 0; i < numPtrs; i += 1) {
    Atomic atomic(ptrs[i]);
    // We use the "acquire" memory order because a successful load means we've
    // been given control of some memory and we can perform some operations on
    // it, hence we want to see the correct state of that memory by the time we
    // do the load, and we don't want any of our ops to be reordered before the
    // load.
    uint64_t numSpins = 0;
    while (atomic.load() != seqNum) {
      numSpins += 1;
      if (numSpins == kNumSpinsBetweenTimeoutChecks) {
        if (getNsSinceEpoch() - startTimeNs >= timeoutNs) {
          KERNEL_FAIL_ASSERT(
              "xFormers's fused kernels for sequence parallelism timed out waiting for a peer GPU. To prevent downstream computations from operating on corrupted data, we're bringing the CUDA context down with us.");
        }
        numSpins = 0;
      }
    }
  }
}

void write_values(
    torch::TensorList targets,
    torch::Scalar value,
    c10::Stream stream) {
  TORCH_CHECK(targets.size() <= kMaxWorldSize);
  for (const auto i : c10::irange(targets.size())) {
    TORCH_CHECK(targets[i].dim() == 0);
    TORCH_CHECK(targets[i].scalar_type() == c10::ScalarType::Int);
    TORCH_CHECK(targets[i].is_cuda());
  }
  TORCH_CHECK(value.type() == c10::ScalarType::Long);

  std::array<int*, kMaxWorldSize> rawTargets;
  for (const auto i : c10::irange(targets.size())) {
    rawTargets[i] = targets[i].data_ptr<int>();
  }

  write_values_kernel<<<1, 1, 0, c10::cuda::CUDAStream(stream)>>>(
      rawTargets, targets.size(), static_cast<int>(value.toLong()));
  C10_CUDA_CHECK(cudaGetLastError());
}

void wait_values(
    torch::TensorList sources,
    torch::Scalar value,
    c10::Stream stream,
    torch::Scalar timeoutS) {
  TORCH_CHECK(sources.size() <= kMaxWorldSize);
  for (const auto i : c10::irange(sources.size())) {
    TORCH_CHECK(sources[i].dim() == 0);
    TORCH_CHECK(sources[i].scalar_type() == c10::ScalarType::Int);
    TORCH_CHECK(sources[i].is_cuda());
  }
  TORCH_CHECK(value.type() == c10::ScalarType::Long);
  TORCH_CHECK(timeoutS.type() == c10::ScalarType::Long);

  std::array<int*, kMaxWorldSize> rawSources;
  for (const auto i : c10::irange(sources.size())) {
    rawSources[i] = sources[i].data_ptr<int>();
  }

  wait_values_kernel<<<1, 1, 0, c10::cuda::CUDAStream(stream)>>>(
      rawSources,
      sources.size(),
      static_cast<int>(value.toLong()),
      static_cast<uint64_t>(timeoutS.toLong()) * 1000000000);
  C10_CUDA_CHECK(cudaGetLastError());
}

} // namespace

TORCH_LIBRARY_IMPL(xformers, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("xformers::write_values"), TORCH_FN(write_values));
  m.impl(TORCH_SELECTIVE_NAME("xformers::wait_values"), TORCH_FN(wait_values));
}
