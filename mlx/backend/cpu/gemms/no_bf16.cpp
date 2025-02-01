// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/gemm.h"

namespace mlx::core {

template <>
void matmul<bfloat16_t>(
    const array&,
    const array&,
    array&,
    bool,
    bool,
    size_t,
    size_t,
    float,
    float) {
  throw std::runtime_error("[Matmul::eval_cpu] bfloat16 not supported.");
}

} // namespace mlx::core
