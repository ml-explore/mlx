// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"

#include <cusolverDn.h>

namespace mlx::core {

void check_cusolver_error(const char* name, cusolverStatus_t err);

#define CHECK_CUSOLVER_ERROR(cmd) check_cusolver_error(#cmd, (cmd))

void init_cusolver_handles_cache();

// Returns the per-device handle, bound to the encoder's stream.
cusolverDnHandle_t get_cusolver_handle(cu::CommandEncoder& encoder);

namespace cusolver_utils {

// cuSOLVER is column-major, and a column-major lower triangle is a row-major
// upper one, so the fill mode is the opposite of `upper`. Same reasoning as
// mlx/backend/cpu/cholesky.cpp.
inline cublasFillMode_t uplo_for(bool upper) {
  return upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
}

inline int64_t batch_count(const array& a) {
  auto n = a.shape(-1);
  if (n == 0) {
    return 0;
  }
  return a.size() / (static_cast<int64_t>(n) * n);
}

} // namespace cusolver_utils

} // namespace mlx::core
