// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/atomic_ops.hpp"

namespace mlx::core::rocm {

struct ScatterAssign {
  template <typename T>
  __device__ void operator()(T* out, T val) const {
    *out = val;
  }
};

struct ScatterSum {
  template <typename T>
  __device__ void operator()(T* out, T val) const {
    atomic_add(out, val);
  }
};

struct ScatterProd {
  template <typename T>
  __device__ void operator()(T* out, T val) const {
    atomic_prod(out, val);
  }
};

struct ScatterMax {
  template <typename T>
  __device__ void operator()(T* out, T val) const {
    atomic_max(out, val);
  }
};

struct ScatterMin {
  template <typename T>
  __device__ void operator()(T* out, T val) const {
    atomic_min(out, val);
  }
};

} // namespace mlx::core::rocm
