// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

struct Select {
  template <typename T>
  __device__ T operator()(bool condition, T x, T y) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
      // hip_bfloat16 may not work well with ternary operator
      if (condition) {
        return x;
      } else {
        return y;
      }
    } else if constexpr (std::is_same_v<T, __half>) {
      if (condition) {
        return x;
      } else {
        return y;
      }
    } else {
      return condition ? x : y;
    }
  }
};

} // namespace mlx::core::rocm
