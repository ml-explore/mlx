// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <typename To, typename From>
struct CastOp {
  __device__ To operator()(From x) const {
    return static_cast<To>(x);
  }
};

template <typename To, typename From>
__device__ inline To cast_op(From x) {
  return static_cast<To>(x);
}

} // namespace mlx::core::rocm