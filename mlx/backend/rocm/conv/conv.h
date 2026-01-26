// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core::rocm {

// Convolution using MIOpen (AMD's equivalent of cuDNN)
// Note: MIOpen integration is optional. If not available, convolution
// falls back to CPU implementation.

bool miopen_available();

void conv_forward(
    CommandEncoder& encoder,
    const array& input,
    const array& weight,
    array& output,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups);

void conv_backward_input(
    CommandEncoder& encoder,
    const array& grad_output,
    const array& weight,
    array& grad_input,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups);

void conv_backward_weight(
    CommandEncoder& encoder,
    const array& input,
    const array& grad_output,
    array& grad_weight,
    const std::vector<int>& padding,
    const std::vector<int>& stride,
    const std::vector<int>& dilation,
    int groups);

} // namespace mlx::core::rocm
