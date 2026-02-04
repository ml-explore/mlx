// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Reduce operations for ROCm

// And and Or only work with bool
struct And {
  __device__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

struct Or {
  __device__ bool operator()(bool a, bool b) const {
    return a || b;
  }
};

struct Sum {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a + b;
  }
};

struct Prod {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
};

struct Max {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    // Handle NaN for floating point types
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(a) || isnan(b)) {
        return numeric_limits<float>::quiet_NaN();
      }
    }
    return a > b ? a : b;
  }
};

struct Min {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    // Handle NaN for floating point types
    if constexpr (std::is_floating_point_v<T>) {
      if (isnan(a) || isnan(b)) {
        return numeric_limits<float>::quiet_NaN();
      }
    }
    return a < b ? a : b;
  }
};

// Reduce result type mapping
template <typename Op, typename T>
struct ReduceResult {
  using type = T;
};

// And and Or always return bool
template <typename T>
struct ReduceResult<And, T> {
  using type = bool;
};

template <typename T>
struct ReduceResult<Or, T> {
  using type = bool;
};

// Sum and Prod promote small integers to int32_t
template <typename T>
struct ReduceResult<Sum, T> {
  using type = std::conditional_t<
      (std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

template <typename T>
struct ReduceResult<Prod, T> {
  using type = std::conditional_t<
      (std::is_integral_v<T> && sizeof(T) <= 4),
      int32_t,
      T>;
};

// Reduce init value
template <typename Op, typename T>
struct ReduceInit;

template <typename T>
struct ReduceInit<And, T> {
  static __device__ bool value() {
    return true;
  }
};

template <typename T>
struct ReduceInit<Or, T> {
  static __device__ bool value() {
    return false;
  }
};

template <typename T>
struct ReduceInit<Sum, T> {
  static __device__ auto value() {
    using ResultT = typename ReduceResult<Sum, T>::type;
    return ResultT(0);
  }
};

template <typename T>
struct ReduceInit<Prod, T> {
  static __device__ auto value() {
    using ResultT = typename ReduceResult<Prod, T>::type;
    return ResultT(1);
  }
};

template <typename T>
struct ReduceInit<Max, T> {
  static __device__ T value() {
    return Limits<T>::min();
  }
};

template <typename T>
struct ReduceInit<Min, T> {
  static __device__ T value() {
    return Limits<T>::max();
  }
};

} // namespace rocm

// Column reduction function declarations
void col_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

void all_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type);

void row_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan);

void init_reduce(
    rocm::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type);

} // namespace mlx::core
