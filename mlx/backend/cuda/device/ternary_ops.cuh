// Copyright © 2025 Apple Inc.
#pragma once

namespace mlx::core::cu {

struct Select {
  template <typename T>
  __device__ T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};

} // namespace mlx::core::cu
