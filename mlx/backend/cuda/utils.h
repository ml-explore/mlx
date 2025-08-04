// Copyright © 2025 Apple Inc.

// This file include utilies that are used by C++ code (i.e. .cpp files).

#pragma once

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace mlx::core {

namespace cu {
class Device;
}

struct Dtype;

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

// Move-able RAII handle of cudaGraphExec_t.
class CudaGraphExec {
 public:
  CudaGraphExec(cudaGraphExec_t handle = nullptr);
  CudaGraphExec(CudaGraphExec&& other);
  ~CudaGraphExec();

  CudaGraphExec(const CudaGraphExec&) = delete;
  CudaGraphExec& operator=(const CudaGraphExec&) = delete;

  void instantiate(cudaGraph_t graph);
  void reset();

  operator cudaGraphExec_t() const {
    return handle_;
  }

 private:
  cudaGraphExec_t handle_;
};

// Throw exception if the cuda API does not succeed.
void check_cublas_error(const char* name, cublasStatus_t err);
void check_cuda_error(const char* name, cudaError_t err);
void check_cuda_error(const char* name, CUresult err);

// The macro version that prints the command that failed.
#define CHECK_CUBLAS_ERROR(cmd) check_cublas_error(#cmd, (cmd))
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

// Convert Dtype to CUDA C++ types.
const char* dtype_to_cuda_type(const Dtype& dtype);

} // namespace mlx::core
