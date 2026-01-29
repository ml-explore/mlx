// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/atomic_ops.hpp"
#include "mlx/backend/rocm/device/cast_op.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Reduce ops with atomic_update for col_reduce

struct And {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a && b;
  }

  template <typename T>
  __device__ static constexpr T init() {
    return true;
  }

  __device__ void atomic_update(bool* x, bool y) {
    atomic_reduce<bool, And>(x, y);
  }
};

struct Or {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a || b;
  }

  template <typename T>
  __device__ static constexpr T init() {
    return false;
  }

  __device__ void atomic_update(bool* x, bool y) {
    atomic_reduce<bool, Or>(x, y);
  }
};

struct Sum {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }

  template <typename T>
  __device__ static constexpr T init() {
    return T(0);
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Sum>(x, y);
  }

  __device__ void atomic_update(float* x, float y) {
    atomicAdd(x, y);
  }

  __device__ void atomic_update(int* x, int y) {
    atomicAdd(x, y);
  }
};

struct Prod {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a * b;
  }

  template <typename T>
  __device__ static constexpr T init() {
    return T(1);
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Prod>(x, y);
  }
};

struct Max {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    // Handle NaN for floating point
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(a) || isnan(b)) {
        return a > b ? a : b;  // Propagate NaN
      }
    }
    return a > b ? a : b;
  }

  template <typename T>
  __device__ static constexpr T init() {
    return numeric_limits<T>::lowest();
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Max>(x, y);
  }
};

struct Min {
  template <typename T>
  __device__ __forceinline__ T operator()(T a, T b) const {
    // Handle NaN for floating point
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(a) || isnan(b)) {
        return a < b ? a : b;  // Propagate NaN
      }
    }
    return a < b ? a : b;
  }

  template <typename T>
  __device__ static constexpr T init() {
    return numeric_limits<T>::max();
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Min>(x, y);
  }
};

// Traits to get the result type of reduce op.
template <typename Op, typename T>
struct ReduceResult {
  using type = T;
};

template <typename T>
struct ReduceResult<And, T> {
  using type = bool;
};

template <typename T>
struct ReduceResult<Or, T> {
  using type = bool;
};

template <typename T>
struct ReduceResult<Sum, T> {
  using type = std::conditional_t<(std::is_integral_v<T> && sizeof(T) <= 4), int32_t, T>;
};

template <typename T>
struct ReduceResult<Prod, T> {
  using type = std::conditional_t<(std::is_integral_v<T> && sizeof(T) <= 4), int32_t, T>;
};

// Traits to get the init value of reduce op.
template <typename Op, typename T>
struct ReduceInit {
  __device__ static T value() {
    return Op::template init<T>();
  }
};

template <typename T>
struct ReduceInit<Sum, T> {
  __device__ static auto value() {
    return typename ReduceResult<Sum, T>::type(0);
  }
};

template <typename T>
struct ReduceInit<Prod, T> {
  __device__ static auto value() {
    return typename ReduceResult<Prod, T>::type(1);
  }
};

template <typename T>
struct ReduceInit<Max, T> {
  __device__ static T value() {
    return numeric_limits<T>::lowest();
  }
};

template <typename T>
struct ReduceInit<Min, T> {
  __device__ static T value() {
    return numeric_limits<T>::max();
  }
};

template <typename T>
struct ReduceInit<And, T> {
  __device__ static bool value() {
    return true;
  }
};

template <typename T>
struct ReduceInit<Or, T> {
  __device__ static bool value() {
    return false;
  }
};

} // namespace mlx::core::rocm
