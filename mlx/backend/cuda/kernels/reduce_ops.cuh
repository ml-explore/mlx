// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/utils.cuh"

namespace mlx::core::cu {

// Reduce ops.
struct And {
  __device__ bool operator()(bool a, bool b) {
    return a && b;
  }
};

struct Or {
  __device__ bool operator()(bool a, bool b) {
    return a || b;
  }
};

struct Sum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a + b;
  }
};

struct Prod {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a * b;
  }
};

struct Min {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a < b ? a : b;
  }
};

struct Max {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a > b ? a : b;
  }
};

// Traits to get the result type of reduce op.
template <typename Op, typename T>
struct ReduceResult;

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
  using type = cuda::std::conditional_t<
      (cuda::std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

template <typename T>
struct ReduceResult<Prod, T> {
  using type = cuda::std::conditional_t<
      (cuda::std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

template <typename T>
struct ReduceResult<Min, T> {
  using type = T;
};

template <typename T>
struct ReduceResult<Max, T> {
  using type = T;
};

// Traits to get the init value of reduce op.
template <typename Op, typename T>
struct ReduceInit;

template <typename T>
struct ReduceInit<And, T> {
  static constexpr bool value = true;
};

template <typename T>
struct ReduceInit<Or, T> {
  static constexpr bool value = false;
};

template <typename T>
struct ReduceInit<Sum, T> {
  static constexpr typename ReduceResult<Sum, T>::type value = zero_value<T>();
};

template <typename T>
struct ReduceInit<Prod, T> {
  static constexpr typename ReduceResult<Prod, T>::type value = one_value<T>();
};

template <typename T>
struct ReduceInit<Min, T> {
  static constexpr T value = Limits<T>::max;
};

template <typename T>
struct ReduceInit<Max, T> {
  static constexpr T value = Limits<T>::min;
};

} // namespace mlx::core::cu
