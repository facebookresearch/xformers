#include "ck_fmha_global_workspace_allocator.h"

GlobalWorkspace::GlobalWorkspace(){};

void* GlobalWorkspace::allocate(size_t sizeInBytes, hipStream_t stream) {
  std::lock_guard<std::mutex> lck(mtx_);

  auto it = buffers_.find(stream);

  if (it != buffers_.end()) {
    size_t curr_size = it->second.first;

    // if requested size is bigger than existing buffer, allocate a bigger
    // buffer; else re-use the existing buffer
    if (curr_size < sizeInBytes) {
      c10::cuda::HIPCachingAllocator::raw_delete(it->second.second);

      void* new_buf = c10::hip::HIPCachingAllocator::raw_alloc(sizeInBytes);
      it->second.first = sizeInBytes;
      it->second.second = new_buf;

      return new_buf;
    } else
      return it->second.second;
  } else {
    // allocate a buffer and keep it for the stream
    void* new_buf = c10::hip::HIPCachingAllocator::raw_alloc(sizeInBytes);

    auto size_buf = std::make_pair(sizeInBytes, new_buf);

    buffers_.insert(std::make_pair(stream, size_buf));

    return new_buf;
  };
};

GlobalWorkspace* GlobalWorkspace::getGlobalWorkspacePtr() {
  if (singleton_ == nullptr)
    singleton_ = new GlobalWorkspace();

  return singleton_;
};

GlobalWorkspace* GlobalWorkspace::singleton_ = nullptr;
