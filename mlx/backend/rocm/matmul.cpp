// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"

namespace mlx::core::rocm {

void matmul_hip(
    float* a,
    float* b,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  // This is a placeholder - in a real implementation, this would use rocBLAS
  // auto& device = get_current_device();
  // rocblas_sgemm(device.rocblas_handle(), ...);

  // For now, just a placeholder
  (void)a;
  (void)b;
  (void)c;
  (void)m;
  (void)n;
  (void)k;
  (void)stream;
}

} // namespace mlx::core::rocm