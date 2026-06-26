// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/types/half_types.h"

namespace mlx::core::fast {

void rms_norm_highway_float(
    const float* x,
    const float* weight,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight);

void rms_norm_highway_float16(
    const float16_t* x,
    const float16_t* weight,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight);

void rms_norm_highway_bfloat16(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight);

void layer_norm_highway_float(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias);

void layer_norm_highway_float16(
    const float16_t* x,
    const float16_t* weight,
    const float16_t* bias,
    float16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias);

void layer_norm_highway_bfloat16(
    const bfloat16_t* x,
    const bfloat16_t* weight,
    const bfloat16_t* bias,
    bfloat16_t* out,
    int width,
    int rows,
    float eps,
    bool has_weight,
    bool has_bias);

} // namespace mlx::core::fast
