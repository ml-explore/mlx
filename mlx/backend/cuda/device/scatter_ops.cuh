// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/atomic_ops.cuh"

namespace mlx::core::cu {

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

} // namespace mlx::core::cu
