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
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides);

} // namespace mlx::core
