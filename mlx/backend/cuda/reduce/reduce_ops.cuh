// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/utils.cuh"

namespace mlx::core::cu {

template <size_t N>
struct uint_by_size;
template <>
struct uint_by_size<2> {
  using type = uint16_t;
};
template <>
struct uint_by_size<4> {
  using type = uint32_t;
};
template <>
struct uint_by_size<8> {
  using type = unsigned long long int;
};

template <typename T, typename Op>
__device__ void atomic_reduce(T* x, T y) {
  if constexpr (sizeof(T) == 1) {
    using U = uint16_t;
    U* x_int = (U*)((char*)x - ((size_t)x % 2));
    int shift = ((char*)x - (char*)x_int) * 8;
    int mask = 0xff << shift;
    U old_val, new_val;
    do {
      old_val = *x_int;
      T result = Op{}(static_cast<T>((old_val >> shift) & 0xff), y);
      new_val = (old_val & ~mask) | (result << shift);
    } while (atomicCAS(x_int, old_val, new_val) != old_val);
  } else {
    using U = typename uint_by_size<sizeof(T)>::type;
    U* x_int = (U*)(x);
    U old_val, new_val;
    do {
      old_val = *x_int;
      T result = Op{}(*((T*)&old_val), y);
      new_val = *((U*)&result);
    } while (atomicCAS(x_int, old_val, new_val) != old_val);
  }
}

// Reduce ops.
struct And {
  __device__ bool operator()(bool a, bool b) {
    return a && b;
  }

  __device__ void atomic_update(bool* x, bool y) {
    atomic_reduce<bool, And>(x, y);
  }
};

struct Or {
  __device__ bool operator()(bool a, bool b) {
    return a || b;
  }

  __device__ void atomic_update(bool* x, bool y) {
    atomic_reduce<bool, Or>(x, y);
  }
};

struct Sum {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a + b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Sum>(x, y);
  }

  __device__ void atomic_update(__nv_bfloat16* x, __nv_bfloat16 y) {
    atomicAdd(x, y);
  }

  __device__ void atomic_update(int* x, int y) {
    atomicAdd(x, y);
  }

  __device__ void atomic_update(float* x, float y) {
    atomicAdd(x, y);
  }
};

struct Prod {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a * b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Prod>(x, y);
  }
};

struct Min {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a < b ? a : b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Min>(x, y);
  }
};

struct Max {
  template <typename T>
  __device__ T operator()(T a, T b) {
    return a > b ? a : b;
  }

  template <typename T>
  __device__ void atomic_update(T* x, T y) {
    atomic_reduce<T, Max>(x, y);
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
  static constexpr __host__ __device__ bool value() {
    return true;
  }
};

template <typename T>
struct ReduceInit<Or, T> {
  static constexpr __host__ __device__ bool value() {
    return false;
  }
};

template <typename T>
struct ReduceInit<Sum, T> {
  static constexpr __host__ __device__ auto value() {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return T{0, 0};
    } else {
      return typename ReduceResult<Sum, T>::type{0};
    }
  }
};

template <typename T>
struct ReduceInit<Prod, T> {
  static constexpr __host__ __device__ auto value() {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return T{1, 0};
    } else {
      return typename ReduceResult<Prod, T>::type{1};
    }
  }
};

template <typename T>
struct ReduceInit<Min, T> {
  static constexpr __host__ __device__ T value() {
    return Limits<T>::max();
  }
};

template <typename T>
struct ReduceInit<Max, T> {
  static constexpr __host__ __device__ T value() {
    return Limits<T>::min();
  }
};

} // namespace mlx::core::cu
