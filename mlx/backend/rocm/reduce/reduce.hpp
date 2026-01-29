// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

// Reduce operations for ROCm
struct And {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a && b;
  }
  template <typename T>
  __device__ static constexpr T init() {
    return true;
  }
};

struct Or {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a || b;
  }
  template <typename T>
  __device__ static constexpr T init() {
    return false;
  }
};

struct Sum {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a + b;
  }
  template <typename T>
  __device__ static constexpr T init() {
    return T(0);
  }
};

struct Prod {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a * b;
  }
  template <typename T>
  __device__ static constexpr T init() {
    return T(1);
  }
};

struct Max {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a > b ? a : b;
  }
  template <typename T>
  __device__ static constexpr T init() {
    return numeric_limits<T>::lowest();
  }
};

struct Min {
  template <typename T>
  __device__ T operator()(T a, T b) const {
    return a < b ? a : b;
  }
  template <typename T>
  __device__ static constexpr T init() {
    return numeric_limits<T>::max();
  }
};

// Reduce result type mapping
template <typename Op, typename T>
struct ReduceResult {
  using type = T;
};

// Specialization for Sum with bool - result is int32_t
template <>
struct ReduceResult<Sum, bool> {
  using type = int32_t;
};

// Reduce init value
template <typename Op, typename T>
struct ReduceInit {
  static __device__ T value() {
    return Op::template init<T>();
  }
};

template <typename T>
struct ReduceInit<Sum, T> {
  static __device__ T value() {
    return T(0);
  }
};

template <typename T>
struct ReduceInit<Prod, T> {
  static __device__ T value() {
    return T(1);
  }
};

template <typename T>
struct ReduceInit<Max, T> {
  static __device__ T value() {
    return numeric_limits<T>::lowest();
  }
};

template <typename T>
struct ReduceInit<Min, T> {
  static __device__ T value() {
    return numeric_limits<T>::max();
  }
};

template <typename T>
struct ReduceInit<And, T> {
  static __device__ T value() {
    return true;
  }
};

template <typename T>
struct ReduceInit<Or, T> {
  static __device__ T value() {
    return false;
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
