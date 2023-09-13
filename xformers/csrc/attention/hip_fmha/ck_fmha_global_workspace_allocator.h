#pragma once

#include <map>
#include <mutex>

#include <c10/hip/HIPCachingAllocator.h>
#include <ck/ck.hpp>

class GlobalWorkspace {
 private:
  static GlobalWorkspace* singleton_;

  std::map<hipStream_t, std::pair<size_t, void*>> buffers_;
  std::mutex mtx_;

 protected:
  GlobalWorkspace();

 public:
  // for each stream, we assume only one workspace buffer is needed, so
  // next allocation will implicitly de-allocate or reuse previous allocation
  // for this stream
  void* allocate(size_t sizeInBytes, hipStream_t stream);

  static GlobalWorkspace* getGlobalWorkspacePtr();

  GlobalWorkspace(const GlobalWorkspace&) = delete;
  GlobalWorkspace(GlobalWorkspace&&) = delete;
  GlobalWorkspace& operator=(const GlobalWorkspace&) = delete;
  GlobalWorkspace& operator=(GlobalWorkspace&&) = delete;
};
