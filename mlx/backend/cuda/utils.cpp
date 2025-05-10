// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/device.h"

#include <fmt/format.h>

namespace mlx::core {

CudaStream::CudaStream(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

CudaStream::~CudaStream() {
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

void check_cuda_error(const char* name, cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, cudaGetErrorString(err)));
  }
}

} // namespace mlx::core
