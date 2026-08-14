// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cusolver_utils.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

auto& cusolver_handles_cache() {
  struct CusolverHandle {
    ~CusolverHandle() {
      if (handle) {
        CHECK_CUSOLVER_ERROR(cusolverDnDestroy(handle));
      }
    }
    cusolverDnHandle_t handle{nullptr};
  };
  static thread_local std::vector<CusolverHandle> cache(gpu::device_count());
  return cache;
}

} // namespace

void check_cusolver_error(const char* name, cusolverStatus_t err) {
  if (err != CUSOLVER_STATUS_SUCCESS) {
    // cuSOLVER has no status-to-string helper, so report the raw code.
    throw std::runtime_error(
        fmt::format("{} failed with code: {}.", name, static_cast<int>(err)));
  }
}

void init_cusolver_handles_cache() {
  cusolver_handles_cache();
}

cusolverDnHandle_t get_cusolver_handle(cu::CommandEncoder& encoder) {
  auto& device = encoder.device();
  auto& storage = cusolver_handles_cache().at(device.cuda_device());
  if (!storage.handle) {
    device.make_current();
    CHECK_CUSOLVER_ERROR(cusolverDnCreate(&storage.handle));
  }
  CHECK_CUSOLVER_ERROR(cusolverDnSetStream(storage.handle, encoder.stream()));
  return storage.handle;
}

} // namespace mlx::core
