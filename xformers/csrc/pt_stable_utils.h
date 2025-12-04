#pragma once

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/TensorAccessor.h>
#include <torch/headeronly/util/Metaprogramming.h>

namespace {

template <class... T, std::size_t... I>
std::tuple<T...> unbox_to_tuple_impl(
    StableIValue* stack,
    std::index_sequence<I...>) {
  return std::make_tuple(
      torch::stable::detail::to<std::remove_cv_t<std::remove_reference_t<T>>>(
          stack[I])...);
}

template <class... T>
std::tuple<T...> unbox_to_tuple(StableIValue* stack) {
  return unbox_to_tuple_impl<T...>(
      stack, std::make_index_sequence<sizeof...(T)>());
}

template <class... T, std::size_t... I>
void box_from_tuple_impl(
    StableIValue* stack,
    std::tuple<T...> vals,
    std::index_sequence<I...>) {
  ((stack[I] = torch::stable::detail::from<
        std::remove_cv_t<std::remove_reference_t<T>>>(std::get<I>(vals))),
   ...);
}

template <class... T>
void box_from_tuple(StableIValue* stack, std::tuple<T...> vals) {
  box_from_tuple_impl<T...>(
      stack, vals, std::make_index_sequence<sizeof...(T)>());
}

template <
    typename ReturnType,
    typename ParameterTypeList,
    typename FuncT,
    FuncT* func>
struct boxer_impl {};

template <
    typename... ReturnTypes,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    std::tuple<ReturnTypes...>,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  void operator()(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    assert(num_args == sizeof...(ParameterTypes));
    assert(num_outputs == sizeof...(ReturnTypes));
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    auto res = std::apply(func, args);
    box_from_tuple<ReturnTypes...>(stack, res);
  }
};

template <
    typename ReturnType,
    typename... ParameterTypes,
    typename FuncT,
    FuncT* func>
struct boxer_impl<
    ReturnType,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  void operator()(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    assert(num_args == sizeof...(ParameterTypes));
    assert(num_outputs == 1);
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    auto res = std::apply(func, args);
    stack[0] = torch::stable::detail::from<ReturnType>(res);
    // box_from_tuple<std::tuple<ReturnType>>(stack, std::make_tuple(res));
  }
};

template <typename... ParameterTypes, typename FuncT, FuncT* func>
struct boxer_impl<
    void,
    c10::guts::typelist::typelist<ParameterTypes...>,
    FuncT,
    func> {
  void operator()(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    assert(num_args == sizeof...(ParameterTypes));
    assert(num_outputs == 0);
    std::tuple<ParameterTypes...> args =
        unbox_to_tuple<ParameterTypes...>(stack);
    std::apply(func, args);
  }
};

template <typename FuncT, FuncT* func>
struct boxer {
  using FunctionTrait = c10::guts::infer_function_traits_t<FuncT>;

  static void boxed_fn(
      StableIValue* stack,
      uint64_t num_args,
      uint64_t num_outputs) {
    boxer_impl<
        typename FunctionTrait::return_type,
        c10::guts::typelist::map_t<
            std::remove_reference_t,
            typename FunctionTrait::parameter_types>,
        FuncT,
        func>()(stack, num_args, num_outputs);
  }
};

} // namespace

#define XF_BOXED_FN(func)                                             \
  (boxer<                                                             \
      std::remove_pointer_t<std::remove_reference_t<decltype(func)>>, \
      (func)>::boxed_fn)

#define XF_CUDA_CHECK(EXPR)                                                    \
  do {                                                                         \
    const cudaError_t __err = EXPR;                                            \
    /* TODO Call stable version of c10::cuda::c10_cuda_check_implementation */ \
    if (__err != cudaSuccess) {                                                \
      throw std::runtime_error(cudaGetErrorString(__err));                     \
    }                                                                          \
  } while (0)

#define XF_CUDA_KERNEL_LAUNCH_CHECK() XF_CUDA_CHECK(cudaGetLastError())

#define XF_CUDA_DRIVER_CHECK(EXPR)                   \
  do {                                               \
    const CUresult __err = EXPR;                     \
    if (__err != CUDA_SUCCESS) {                     \
      throw std::runtime_error("CUDA driver error"); \
    }                                                \
  } while (0)

cudaDeviceProp* xf_getCurrentDeviceProperties();

namespace {

cudaStream_t xf_getCurrentCUDAStream(
    torch::stable::accelerator::DeviceIndex index = -1) {
  // cudaStream_t ret;
  // TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(index, &ret));
  // return ret;
  return reinterpret_cast<cudaStream_t>(
      torch::stable::accelerator::getCurrentStream(
          torch::stable::accelerator::getCurrentDeviceIndex())
          .id());
}

template <typename dtype, size_t ndim>
auto xf_packed_accessor(const torch::stable::Tensor& t) {
  return torch::headeronly::HeaderOnlyGenericPackedTensorAccessor<dtype, ndim>(
      t.mutable_data_ptr<dtype>(), t.sizes().data(), t.strides().data());
}

template <typename T>
constexpr __host__ __device__ inline T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

inline int32_t xf_get_layout(const torch::stable::Tensor& self) {
  int32_t layout;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(self.get(), &layout));
  return layout;
}

inline bool xf_is_sparse(const torch::stable::Tensor& self) {
  return xf_get_layout(self) != aoti_torch_layout_strided();
}

inline torch::stable::Tensor xf_view_dtype(
    const torch::stable::Tensor& self,
    torch::headeronly::ScalarType dtype) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dtype)};
  // view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::view", "dtype", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_slice(
    const torch::stable::Tensor& self,
    int64_t dim,
    std::optional<int64_t> start,
    std::optional<int64_t> end) {
  const auto num_args = 5;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(start),
      torch::stable::detail::from(end),
      torch::stable::detail::from(1)};
  // slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt?
  // end=None, SymInt step=1) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::slice", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_select(
    const torch::stable::Tensor& self,
    int64_t dim,
    int64_t index) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(index)};
  // select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::select", "int", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_permute(
    const torch::stable::Tensor& self,
    std::vector<int64_t> dims) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self), torch::stable::detail::from(dims)};
  // permute(Tensor(a) self, int[] dims) -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::permute", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_contiguous(
    const torch::stable::Tensor& self,
    int32_t memory_format = aoti_torch_memory_format_contiguous_format()) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(memory_format),
  };
  // contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format)
  // -> Tensor(a)
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::contiguous", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_zeros(
    std::vector<int64_t> size,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  const auto num_args = 5;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(size),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(device),
      torch::stable::detail::from(pin_memory)};
  // zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None,
  // Device? device=None, bool? pin_memory=None) -> Tensor
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::zeros", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

template <typename T>
inline torch::stable::Tensor xf_full(
    std::vector<int64_t> size,
    T fill_value,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt,
    std::optional<torch::stable::Device> device = std::nullopt,
    std::optional<bool> pin_memory = std::nullopt) {
  const auto num_args = 6;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(size),
      torch::stable::detail::from(fill_value),
      torch::stable::detail::from(dtype),
      torch::stable::detail::from(std::nullopt),
      torch::stable::detail::from(device),
      torch::stable::detail::from(pin_memory)};
  // full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout?
  // layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::full", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_cumsum(
    const torch::stable::Tensor& self,
    int dim,
    std::optional<torch::headeronly::ScalarType> dtype = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(dim),
      torch::stable::detail::from(dtype)};
  // cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::cumsum", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

inline torch::stable::Tensor xf_resize_(
    const torch::stable::Tensor& self,
    std::vector<int64_t> size,
    int32_t memory_format = aoti_torch_memory_format_contiguous_format()) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(size),
      torch::stable::detail::from(memory_format)};
  // resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat?
  // memory_format=None) -> Tensor(a!)
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::resize_", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<torch::stable::Tensor>(stack[0]);
}

template <typename T>
inline T xf_item(const torch::stable::Tensor& self) {
  // const auto num_args = 1;
  // std::array<StableIValue, num_args> stack{
  //     torch::stable::detail::from(self)};
  // // item(Tensor self) -> Scalar
  // TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
  //     "aten::item", "", stack.data(), TORCH_ABI_VERSION));
  // return torch::stable::detail::to<T>(stack[0]);
  torch::stable::Tensor cpu_self = torch::stable::empty(
      self.sizes(),
      self.scalar_type(),
      std::nullopt,
      torch::stable::Device(torch::headeronly::kCPU));
  torch::stable::copy_(cpu_self, self, /*non_blocking=*/false);
  static_assert(std::is_trivially_copyable_v<T>, "");
  T res = *cpu_self.const_data_ptr<T>();
  return res;
}

size_t xf_element_size(const torch::stable::Tensor& self) {
#define RETURN_SIZEOF_IF_MATCHES_(cpp_type, dtype)                  \
  if (self.scalar_type() == torch::headeronly::ScalarType::dtype) { \
    return sizeof(cpp_type);                                        \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(RETURN_SIZEOF_IF_MATCHES_)
#undef RETURN_SIZEOF_IF_MATCHES_
  throw std::runtime_error("Unsupported dtype");
}

} // namespace
