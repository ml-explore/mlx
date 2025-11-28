// Copyright © 2025 Apple Inc.

#pragma once

#include <cublasLt.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace mlx::core {

// Throw exception if the cuda API does not succeed.
void check_cublas_error(const char* name, cublasStatus_t err);
void check_cuda_error(const char* name, cudaError_t err);
void check_cuda_error(const char* name, CUresult err);
void check_cudnn_error(const char* name, cudnnStatus_t err);

// The macro version that prints the command that failed.
#define CHECK_CUBLAS_ERROR(cmd) check_cublas_error(#cmd, (cmd))
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))
#define CHECK_CUDNN_ERROR(cmd) check_cudnn_error(#cmd, (cmd))

// Base class for RAII managed CUDA resources.
template <typename Handle, cudaError_t (*Destroy)(Handle)>
class CudaHandle {
 public:
  CudaHandle(Handle handle = nullptr) : handle_(handle) {}

  CudaHandle(CudaHandle&& other) : handle_(other.handle_) {
    assert(this != &other);
    other.handle_ = nullptr;
  }

  ~CudaHandle() {
    // Skip if there was an error to avoid throwing in the destructors
    if (cudaPeekAtLastError() != cudaSuccess) {
      return;
    }
    reset();
  }

  CudaHandle(const CudaHandle&) = delete;
  CudaHandle& operator=(const CudaHandle&) = delete;

  CudaHandle& operator=(CudaHandle&& other) {
    assert(this != &other);
    reset();
    std::swap(handle_, other.handle_);
    return *this;
  }

  void reset() {
    if (handle_ != nullptr) {
      CHECK_CUDA_ERROR(Destroy(handle_));
      handle_ = nullptr;
    }
  }

  operator Handle() const {
    return handle_;
  }

 protected:
  Handle handle_;
};

namespace cu {
class Device;
}; // namespace cu

// Wrappers of CUDA resources.
class CudaGraph : public CudaHandle<cudaGraph_t, cudaGraphDestroy> {
 public:
  using CudaHandle::CudaHandle;
  explicit CudaGraph(cu::Device& device);
  void end_capture(cudaStream_t stream);
};

class CudaGraphExec : public CudaHandle<cudaGraphExec_t, cudaGraphExecDestroy> {
 public:
  void instantiate(cudaGraph_t graph);
};

class CudaStream : public CudaHandle<cudaStream_t, cudaStreamDestroy> {
 public:
  explicit CudaStream(cu::Device& device);
};

} // namespace mlx::core
