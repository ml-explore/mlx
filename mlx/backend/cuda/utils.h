// Copyright © 2025 Apple Inc.

// This file include utilies that are used by C++ code (i.e. .cpp files).

#pragma once

#include "mlx/array.h"

#include <cuda_runtime.h>

namespace mlx::core {

namespace cu {
class Device;
}

// Cuda stream managed with RAII.
class CudaStream {
 public:
  explicit CudaStream(cu::Device& device);
  ~CudaStream();

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  operator cudaStream_t() const {
    return stream_;
  }

 private:
  cudaStream_t stream_;
};

// Throw exception if the cuda API does not succeed.
void check_cuda_error(const char* name, cudaError_t err);

// The macro version that prints the command that failed.
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

// Compute the thread block dimensions which fit the given input dimensions.
dim3 get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10);

// Computes a 2D grid where each element is < UINT_MAX.
dim3 get_2d_grid_dims(const Shape& shape, const Strides& strides);

} // namespace mlx::core
