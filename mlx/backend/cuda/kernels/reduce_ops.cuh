// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/utils.cuh"

namespace mlx::core::mxcuda {

template <typename U = bool>
struct And {
  static constexpr bool init = true;

  __device__ bool operator()(bool a, bool b) {
    return a && b;
  }
};

template <typename U = bool>
struct Or {
  static constexpr bool init = false;

  __device__ bool operator()(bool a, bool b) {
    return a || b;
  }
};

template <typename U>
struct Sum {
  static constexpr U init = zero_value<U>();

  __device__ U operator()(U a, U b) {
    return a + b;
  }
};

template <typename U>
struct Prod {
  static constexpr U init = one_value<U>();

  __device__ U operator()(U a, U b) {
    return a * b;
  }
};

template <typename U>
struct Min {
  static constexpr U init = Limits<U>::max;

  __device__ U operator()(U a, U b) {
    return a < b ? a : b;
  }
};

template <typename U>
struct Max {
  static constexpr U init = Limits<U>::min;

  __device__ U operator()(U a, U b) {
    return a > b ? a : b;
  }
};

} // namespace mlx::core::mxcuda
