// Copyright © 2025 Apple Inc.

#pragma once
#include "mlx/array.h"

namespace mlx::core {

template <typename T>
void matmul(
    const T* a,
    const T* b,
    T* out,
    bool a_transposed,
    bool b_transposed,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    float alpha,
    float beta,
    int64_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides);

} // namespace mlx::core
