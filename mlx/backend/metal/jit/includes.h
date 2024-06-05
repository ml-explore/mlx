// Copyright © 2023-2024 Apple Inc.

#pragma once

namespace mlx::core::metal {

const char* utils();
const char* binary_ops();
const char* unary_ops();
const char* ternary_ops();
const char* reduce_utils();
const char* gather();
const char* scatter();

const char* arange();
const char* unary();
const char* binary();
const char* binary_two();
const char* copy();
const char* fft();
const char* ternary();
const char* scan();
const char* softmax();
const char* sort();
const char* reduce();

const char* gemm();
const char* steel_gemm_fused();
const char* steel_gemm_masked();
const char* steel_gemm_splitk();
const char* conv();
const char* steel_conv();
const char* steel_conv_general();

} // namespace mlx::core::metal
