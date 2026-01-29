// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/dtype_utils.h"

#include <sstream>

namespace mlx::core {

void check_rocblas_error(const char* name, rocblas_status err) {
  if (err != rocblas_status_success) {
    std::ostringstream oss;
    oss << name << " failed with code: " << static_cast<int>(err) << ".";
    throw std::runtime_error(oss.str());
  }
}

void check_hip_error(const char* name, hipError_t err) {
  if (err != hipSuccess) {
    std::ostringstream oss;
    oss << name << " failed: " << hipGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}

const char* dtype_to_hip_type(const Dtype& dtype) {
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
      return "__hip_bfloat16";
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

HipGraph::HipGraph(rocm::Device& device) {
  device.make_current();
  CHECK_HIP_ERROR(hipGraphCreate(&handle_, 0));
}

void HipGraph::end_capture(hipStream_t stream) {
  assert(handle_ == nullptr);
  CHECK_HIP_ERROR(hipStreamEndCapture(stream, &handle_));
}

void HipGraphExec::instantiate(hipGraph_t graph) {
  assert(handle_ == nullptr);
  CHECK_HIP_ERROR(hipGraphInstantiate(&handle_, graph, nullptr, nullptr, 0));
}

HipStream::HipStream(rocm::Device& device) {
  device.make_current();
  CHECK_HIP_ERROR(hipStreamCreateWithFlags(&handle_, hipStreamNonBlocking));
}

} // namespace mlx::core
