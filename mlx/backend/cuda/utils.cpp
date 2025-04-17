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

// TODO: The implementation is identical to meta/utils.cpp
dim3 get_2d_grid_dims(const Shape& shape, const Strides& strides) {
  size_t grid_x = 1;
  size_t grid_y = 1;
  for (int i = 0; i < shape.size(); ++i) {
    if (strides[i] == 0) {
      continue;
    }
    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }
  }
  if (grid_y > UINT32_MAX || grid_x > UINT32_MAX) {
    throw std::runtime_error("Unable to safely factor shape.");
  }
  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }
  return dim3{static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1};
}

} // namespace mlx::core
