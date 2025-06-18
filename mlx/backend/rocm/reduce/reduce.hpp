// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <cstddef>

namespace mlx::core::rocm {

// Reduction operation types
template <typename Op, typename T>
struct ReduceInit {
  static constexpr T value();
};

template <typename T>
struct ReduceInit<struct Sum, T> {
  static constexpr T value() {
    return T(0);
  }
};

template <typename T>
struct ReduceInit<struct Max, T> {
  static constexpr T value() {
    return -std::numeric_limits<T>::infinity();
  }
};

template <typename T>
struct ReduceInit<struct Min, T> {
  static constexpr T value() {
    return std::numeric_limits<T>::infinity();
  }
};

// Reduction operations
struct Sum {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a + b;
  }
};

struct Max {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return fmax(a, b);
  }
};

struct Min {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return fmin(a, b);
  }
};

struct Prod {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

// Utility functions for reductions
template <typename T>
__device__ T warp_reduce(T val, T (*op)(T, T)) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = op(val, __shfl_down(val, offset));
  }
  return val;
}

template <typename T>
__device__ T block_reduce(T val, T (*op)(T, T)) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warp_reduce(val, op);

  if (lane == 0)
    shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  if (wid == 0)
    val = warp_reduce(val, op);

  return val;
}

// Column reduction arguments
struct ColReduceArgs {
  size_t reduction_size;
  int64_t reduction_stride;
  int* shape;
  size_t* strides;
  int ndim;
  int* reduce_shape;
  size_t* reduce_strides;
  int reduce_ndim;
  size_t non_col_reductions;
};

// Row reduction arguments
struct RowReduceArgs {
  size_t reduction_size;
  int64_t reduction_stride;
  int* shape;
  size_t* strides;
  int ndim;
  int* reduce_shape;
  size_t* reduce_strides;
  int reduce_ndim;
};

} // namespace mlx::core::rocm