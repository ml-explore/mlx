// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"

#include <fmt/format.h>

namespace mlx::core {

inline const char* cublasGetStatusStringLocal(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDART_VERSION >= 11000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    default: return "Unknown cuBLAS status";
  }
}
  
inline void check_cublas_error(const char* name, cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    cudaError_t cuda_err = cudaGetLastError();
    const char* cuda_err_name = cudaGetErrorName(cuda_err);
    const char* cuda_err_str  = cudaGetErrorString(cuda_err);

    std::string msg = fmt::format(
        "[cuBLAS] {} failed:\n"
        "  cuBLAS status : {} ({})\n"
        "  CUDA last err : {} ({})\n",
        name,
        cublasGetStatusStringLocal(status),
        static_cast<int>(status),
        cuda_err_name ? cuda_err_name : "unknown",
        cuda_err_str ? cuda_err_str : "unknown");

    throw std::runtime_error(msg);
  }
}

void check_cuda_error(const char* name, cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, cudaGetErrorString(err)));
  }
}

void check_cuda_error(const char* name, CUresult err) {
  if (err != CUDA_SUCCESS) {
    const char* err_str = "Unknown error";
    cuGetErrorString(err, &err_str);
    throw std::runtime_error(fmt::format("{} failed: {}", name, err_str));
  }
}

const char* dtype_to_cuda_type(const Dtype& dtype) {
  switch (dtype) {
    case bool_:
      return "bool";
    case int8:
      return "int8_t";
    case int16:
      return "int16_t";
    case int32:
      return "int32_t";
    case int64:
      return "int64_t";
    case uint8:
      return "uint8_t";
    case uint16:
      return "uint16_t";
    case uint32:
      return "uint32_t";
    case uint64:
      return "uint64_t";
    case float16:
      return "__half";
    case bfloat16:
      return "__nv_bfloat16";
    case float32:
      return "float";
    case float64:
      return "double";
    case complex64:
      return "complex64_t";
    default:
      return "unknown";
  }
}

CudaGraph::CudaGraph(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaGraphCreate(&handle_, 0));
}

void CudaGraph::end_capture(cudaStream_t stream) {
  assert(handle_ == nullptr);
  CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &handle_));
}

void CudaGraphExec::instantiate(cudaGraph_t graph) {
  assert(handle_ == nullptr);
  CHECK_CUDA_ERROR(cudaGraphInstantiate(&handle_, graph, nullptr, nullptr, 0));
}

CudaStream::CudaStream(cu::Device& device) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&handle_, cudaStreamNonBlocking));
}

} // namespace mlx::core
