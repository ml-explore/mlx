// Copyright © 2026 Apple Inc.
#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

namespace mlx::core {
void qqmm_impl(
    cu::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    Dtype out_dtype,
    QuantizationMode mode,
    float alpha = 1.0f);

} // namespace mlx::core
