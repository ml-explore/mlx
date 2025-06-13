// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"

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

const char* dtype_to_cuda_type(const Dtype& dtype) {
  if (dtype == float16) {
    return "__half";
  }
  if (dtype == bfloat16) {
    return "__nv_bfloat16";
  }
#define SPECIALIZE_DtypeToString(CPP_TYPE, DTYPE) \
  if (dtype == DTYPE) {                           \
    return #CPP_TYPE;                             \
  }
  MLX_FORALL_DTYPES(SPECIALIZE_DtypeToString)
#undef SPECIALIZE_DtypeToString
  return nullptr;
}

} // namespace mlx::core
