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

// Throw exception if the cuda API does not succeed.
void check_cublas_error(const char* name, cublasStatus_t err);
void check_cuda_error(const char* name, cudaError_t err);
void check_cuda_error(const char* name, CUresult err);

// The macro version that prints the command that failed.
#define CHECK_CUBLAS_ERROR(cmd) check_cublas_error(#cmd, (cmd))
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

// Convert Dtype to CUDA C++ types.
const char* dtype_to_cuda_type(const Dtype& dtype);

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
