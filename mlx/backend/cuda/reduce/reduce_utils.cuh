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

// TODO: Should make a custom complex type
template <typename U, typename T>
inline __device__ U __cast(T x) {
  return static_cast<U>(x);
}

template <>
inline __device__ bool __cast<bool, cuComplex>(cuComplex x) {
  return x.x != 0 && x.y != 0;
}

template <>
inline __device__ cuComplex __cast<cuComplex, bool>(bool x) {
  return x ? make_cuFloatComplex(1, 1) : make_cuFloatComplex(0, 0);
}

} // namespace mlx::core::cu
