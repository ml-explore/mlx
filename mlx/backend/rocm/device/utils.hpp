// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>

#include <cstdint>
#include <type_traits>

namespace mlx::core::rocm {

// Type traits for complex types
template <typename T>
struct is_complex : std::false_type {};

template <>
struct is_complex<hipFloatComplex> : std::true_type {};

template <typename T>
inline constexpr bool is_complex_v = is_complex<T>::value;

// Complex type alias
template <typename T>
using complex_t = hipFloatComplex;

// Numeric limits for device code
template <typename T>
struct numeric_limits;

template <>
struct numeric_limits<float> {
  __device__ static constexpr float infinity() { return __int_as_float(0x7f800000); }
  __device__ static constexpr float quiet_NaN() { return __int_as_float(0x7fc00000); }
  __device__ static constexpr float lowest() { return -3.402823466e+38f; }
  __device__ static constexpr float max() { return 3.402823466e+38f; }
};

template <>
struct numeric_limits<double> {
  __device__ static constexpr double infinity() { return __longlong_as_double(0x7ff0000000000000LL); }
  __device__ static constexpr double quiet_NaN() { return __longlong_as_double(0x7ff8000000000000LL); }
  __device__ static constexpr double lowest() { return -1.7976931348623158e+308; }
  __device__ static constexpr double max() { return 1.7976931348623158e+308; }
};

template <>
struct numeric_limits<__half> {
  __device__ static __half infinity() { return __ushort_as_half(0x7c00); }
  __device__ static __half quiet_NaN() { return __ushort_as_half(0x7e00); }
  __device__ static __half lowest() { return __ushort_as_half(0xfbff); }
  __device__ static __half max() { return __ushort_as_half(0x7bff); }
};

template <>
struct numeric_limits<__hip_bfloat16> {
  __device__ static __hip_bfloat16 infinity() { return __ushort_as_bfloat16(0x7f80); }
  __device__ static __hip_bfloat16 quiet_NaN() { return __ushort_as_bfloat16(0x7fc0); }
  __device__ static __hip_bfloat16 lowest() { return __ushort_as_bfloat16(0xff7f); }
  __device__ static __hip_bfloat16 max() { return __ushort_as_bfloat16(0x7f7f); }
};

template <>
struct numeric_limits<int32_t> {
  __device__ static constexpr int32_t lowest() { return INT32_MIN; }
  __device__ static constexpr int32_t max() { return INT32_MAX; }
};

template <>
struct numeric_limits<int64_t> {
  __device__ static constexpr int64_t lowest() { return INT64_MIN; }
  __device__ static constexpr int64_t max() { return INT64_MAX; }
};

template <>
struct numeric_limits<uint32_t> {
  __device__ static constexpr uint32_t lowest() { return 0; }
  __device__ static constexpr uint32_t max() { return UINT32_MAX; }
};

template <>
struct numeric_limits<uint64_t> {
  __device__ static constexpr uint64_t lowest() { return 0; }
  __device__ static constexpr uint64_t max() { return UINT64_MAX; }
};

// Strides type
using Strides = int64_t[8];

// HIP array type (similar to cuda::std::array)
template <typename T, int N>
struct hip_array {
  T data_[N];
  
  __host__ __device__ T& operator[](int i) { return data_[i]; }
  __host__ __device__ const T& operator[](int i) const { return data_[i]; }
  __host__ __device__ constexpr int size() const { return N; }
};

// Ceil division
template <typename T>
__host__ __device__ T ceildiv(T a, T b) {
  return (a + b - 1) / b;
}

// Elem to loc conversion
template <typename IdxT = int64_t>
__device__ IdxT elem_to_loc(
    IdxT elem,
    const int* shape,
    const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    loc += (elem % shape[i]) * strides[i];
    elem /= shape[i];
  }
  return loc;
}

// Get the thread index in the block
__device__ inline int thread_index() {
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

// Get the block index in the grid
__device__ inline int block_index() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

// Get the global thread index
__device__ inline int global_thread_index() {
  return thread_index() + block_index() * (blockDim.x * blockDim.y * blockDim.z);
}

} // namespace mlx::core::rocm
