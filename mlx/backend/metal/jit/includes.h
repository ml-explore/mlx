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
const char* gather_axis();
const char* gather_front();
const char* hadamard();
const char* logsumexp();
const char* quantized_utils();
const char* quantized();
const char* fp4_quantized();
const char* ternary();
const char* scan();
const char* scatter_axis();
const char* softmax();
const char* sort();
const char* reduce();

const char* gemm();
const char* steel_gemm_fused();
const char* steel_gemm_masked();
const char* steel_gemm_splitk();
const char* steel_gemm_gather();
const char* steel_gemm_segmented();
const char* conv();
const char* steel_conv();
const char* steel_conv_general();
const char* gemv_masked();

} // namespace mlx::core::metal
