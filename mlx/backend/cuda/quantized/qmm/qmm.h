// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

#include <optional>

namespace mlx::core {

bool supports_qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);

bool supports_fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);

bool supports_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device);

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
