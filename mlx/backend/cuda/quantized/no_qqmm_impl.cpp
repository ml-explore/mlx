// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qqmm_impl.h"

namespace mlx::core {
void qqmm_impl(
    cu::CommandEncoder&,
    int,
    int,
    int,
    bool,
    int64_t,
    bool,
    int64_t,
    array&,
    const array&,
    const array&,
    const array&,
    const array&,
    Dtype,
    QuantizationMode,
    float) {
  throw std::runtime_error(
      "[QQMatmul::eval_gpu] QQMM is only supported with CUDA 12.8 or higher.");
}
} // namespace mlx::core
