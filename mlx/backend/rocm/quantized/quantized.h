// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core {

// Affine quantization functions
void affine_quantize(
    const array& w,
    array& wq,
    array& scales,
    array& biases,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

void affine_dequantize(
    const array& wq,
    const array& scales,
    const array& biases,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

// Floating-point quantization functions
void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    rocm::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
