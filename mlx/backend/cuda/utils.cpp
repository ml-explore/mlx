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
dim3 get_block_dims(int dim0, int dim1, int dim2, int pow2) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (dim0 >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim1 >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim2 >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == pow2) {
      break;
    }
  }
  return {1u << pows[0], 1u << pows[1], 1u << pows[2]};
}

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
