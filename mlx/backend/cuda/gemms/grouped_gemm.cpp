// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/cuda/device.h"

#include <stdexcept>

namespace mlx::core {

void grouped_mm(
    const array& a,
    const array& b,
    const array& offsets,
    array& out,
    cu::CommandEncoder& encoder) {
  if (encoder.device().compute_capability_major() < 8) {
    throw std::runtime_error(
        "[grouped_mm] Grouped matmul requires sm80 or newer.");
  }
  cudnn_grouped_mm(a, b, offsets, out, encoder);
}

} // namespace mlx::core
