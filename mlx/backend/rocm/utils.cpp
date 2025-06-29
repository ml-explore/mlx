// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/dtype_utils.h"

#include <fmt/format.h>

namespace mlx::core {

HipStream::HipStream(rocm::Device& device) {
  device.make_current();
  CHECK_HIP_ERROR(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
}

HipStream::~HipStream() {
  CHECK_HIP_ERROR(hipStreamDestroy(stream_));
}

void check_hip_error(const char* name, hipError_t err) {
  if (err != hipSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", name, hipGetErrorString(err)));
  }
}

const char* dtype_to_hip_type(const Dtype& dtype) {
  if (dtype == float16) {
    return "__half";
  }
  if (dtype == bfloat16) {
    return "__hip_bfloat16";
  }
  if (dtype == complex64) {
    return "hipFloatComplex";
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