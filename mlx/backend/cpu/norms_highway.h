// Copyright © 2026 Apple Inc.

#pragma once

namespace mlx::core::fast {

void rms_norm_highway_float(
    const float* x,
    const float* weight,
    float* out,
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

} // namespace mlx::core::fast
