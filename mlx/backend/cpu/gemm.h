// Copyright © 2025 Apple Inc.

#pragma once
#include "mlx/array.h"

namespace mlx::core {

template <typename T>
void matmul(
    const array& a,
    const array& b,
    array& out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    float alpha,
    float beta);

} // namespace mlx::core
