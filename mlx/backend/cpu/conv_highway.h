// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>

namespace mlx::core::fast {

enum class ConvHighwayDType : uint8_t {
  Float32,
  Float16,
  BFloat16,
};

void depthwise_conv1d_kernel4_highway(
    void* out,
    const void* in,
    const void* wt,
    ConvHighwayDType dtype,
    int N,
    int iH,
    int C,
    int oH);

} // namespace mlx::core::fast
