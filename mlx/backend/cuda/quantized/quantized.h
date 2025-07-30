// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"

namespace mlx::core {

void affine_quantize(
    const array& w,
    array& wq,
    array& scales,
    array& biases,
    int group_size_,
    int bits_,
    cu::CommandEncoder& enc,
    const Stream& s);

void affine_dequantize(
    const array& wq,
    const array& scales,
    const array& biases,
    array& w,
    int group_size_,
    int bits_,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
