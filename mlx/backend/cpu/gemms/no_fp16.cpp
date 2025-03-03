// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/gemm.h"

namespace mlx::core {

template <>
void matmul<float16_t>(
    const float16_t*,
    const float16_t*,
    float16_t*,
    bool,
    bool,
    size_t,
    size_t,
    size_t,
    float,
    float,
    size_t,
    const Shape&,
    const Strides&,
    const Shape&,
    const Strides&) {
  throw std::runtime_error("[Matmul::eval_cpu] float16 not supported.");
}

} // namespace mlx::core
