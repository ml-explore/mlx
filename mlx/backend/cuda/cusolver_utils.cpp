// Copyright © 2026 Apple Inc.
#include "mlx/backend/cuda/cusolver_utils.h"
#include "mlx/backend/gpu/device_info.h"

#include <fmt/format.h>

namespace mlx::core {

namespace {

auto& cusolver_handles_cache() {
  struct CusolverHandle {
    ~CusolverHandle() {
      if (handle) {
        // Not checked: runs at thread exit where a throw would terminate.
        cusolverDnDestroy(handle);
      }
    }
    cusolverDnHandle_t handle{nullptr};
  };
  static thread_local std::vector<CusolverHandle> cache(gpu::device_count());
  return cache;
}

} // namespace

cusolverDnHandle_t get_cusolver_handle(cu::Device& device) {
  auto& storage = cusolver_handles_cache().at(device.cuda_device());
  if (!storage.handle) {
    device.make_current();
    CHECK_CUSOLVER_ERROR(cusolverDnCreate(&storage.handle));
  }
  return storage.handle;
}

void init_cusolver_handles_cache() {
  cusolver_handles_cache();
}

void check_cusolver_error(const char* name, cusolverStatus_t err) {
  if (err != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(
        fmt::format("{} failed with code: {}.", name, static_cast<int>(err)));
  }
}

} // namespace mlx::core
