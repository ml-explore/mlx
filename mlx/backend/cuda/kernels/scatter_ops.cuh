// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/atomic_ops.cuh"

namespace mlx::core::cu {

template <typename T>
struct ScatterAssign {
  __device__ void operator()(T* out, T val) const {
    *out = val;
  }
};

template <typename T>
struct ScatterSum {
  __device__ void operator()(T* out, T val) const {
    atomic_add(out, val);
  }
};

template <typename T>
struct ScatterProd {
  __device__ void operator()(T* out, T val) const {
    atomic_prod(out, val);
  }
};

template <typename T>
struct ScatterMax {
  __device__ void operator()(T* out, T val) const {
    atomic_max(out, val);
  }
};

template <typename T>
struct ScatterMin {
  __device__ void operator()(T* out, T val) const {
    atomic_min(out, val);
  }
};

} // namespace mlx::core::cu
