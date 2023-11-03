#include <pybind11/pybind11.h>

// Must come first to load TORCH_VERSION_FOO.
#include <torch/torch.h>

#if TORCH_VERSION_MAJOR > 1 || \
    (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 13)
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#else
#include <c10d/ProcessGroup.hpp>
#endif

#include <torch/csrc/jit/python/pybind_utils.h>

namespace py = pybind11;

namespace {

// Starting with PyTorch 2.2, we will be able to do boxing/unboxing in Python.
// See https://github.com/pytorch/pytorch/pull/111997.
// In the meantime, we had to implement the conversions in C++ ourselves.

py::object box_process_group(
    c10::intrusive_ptr<c10d::ProcessGroup> process_group) {
  return torch::jit::toPyObject(c10::IValue(process_group));
}

c10::intrusive_ptr<c10d::ProcessGroup> unbox_process_group(
    const py::object& obj) {
  return torch::jit::toIValue(
             obj,
             c10::getCustomClassType<c10::intrusive_ptr<c10d::ProcessGroup>>())
      .toCustomClass<c10d::ProcessGroup>();
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("box_process_group", &box_process_group);
  module.def("unbox_process_group", &unbox_process_group);
}
