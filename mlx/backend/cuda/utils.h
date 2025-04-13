// Copyright © 2025 Apple Inc.

#pragma once

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

} // namespace mlx::core
