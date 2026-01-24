// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

struct Select {
  template <typename T>
  __device__ T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};

} // namespace mlx::core::rocm
