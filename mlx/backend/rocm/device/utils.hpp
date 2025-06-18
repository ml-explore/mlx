// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// HIP/ROCm type definitions
using hip_complex = hipFloatComplex;

// Utility functions for HIP device code
template <typename T>
struct hip_type {
  using type = T;
};

template <>
struct hip_type<bool> {
  using type = bool;
};

template <>
struct hip_type<int8_t> {
  using type = int8_t;
};

template <>
struct hip_type<uint8_t> {
  using type = uint8_t;
};

template <>
struct hip_type<int16_t> {
  using type = int16_t;
};

template <>
struct hip_type<uint16_t> {
  using type = uint16_t;
};

template <>
struct hip_type<int32_t> {
  using type = int32_t;
};

template <>
struct hip_type<uint32_t> {
  using type = uint32_t;
};

template <>
struct hip_type<int64_t> {
  using type = int64_t;
};

template <>
struct hip_type<uint64_t> {
  using type = uint64_t;
};

template <>
struct hip_type<float> {
  using type = float;
};

template <>
struct hip_type<double> {
  using type = double;
};

#ifdef __HIP_PLATFORM_HCC__
template <>
struct hip_type<__half> {
  using type = __half;
};

template <>
struct hip_type<__hip_bfloat16> {
  using type = __hip_bfloat16;
};
#endif

template <typename T>
using hip_type_t = typename hip_type<T>::type;

// Element-wise operations support
template <typename T>
constexpr bool is_floating_point_v = std::is_floating_point_v<T>;

template <typename T>
constexpr bool is_integral_v = std::is_integral_v<T>;

template <typename T>
constexpr bool is_signed_v = std::is_signed_v<T>;

template <typename T>
constexpr bool is_unsigned_v = std::is_unsigned_v<T>;

// Complex number helper functions
inline __device__ hipFloatComplex make_complex(float real, float imag) {
  return make_hipFloatComplex(real, imag);
}

inline __device__ float hip_real(hipFloatComplex z) {
  return hipCrealf(z);
}

inline __device__ float hip_imag(hipFloatComplex z) {
  return hipCimagf(z);
}

inline __device__ hipFloatComplex hip_conj(hipFloatComplex z) {
  return make_hipFloatComplex(hipCrealf(z), -hipCimagf(z));
}

inline __device__ float hip_abs(hipFloatComplex z) {
  return sqrtf(hipCrealf(z) * hipCrealf(z) + hipCimagf(z) * hipCimagf(z));
}

// Memory access utilities
template <typename T>
inline __device__ T hip_load_global(const T* ptr) {
  return *ptr;
}

template <typename T>
inline __device__ void hip_store_global(T* ptr, T value) {
  *ptr = value;
}

// Grid and block utilities
inline __device__ int hip_thread_idx() {
  return threadIdx.x;
}

inline __device__ int hip_block_idx() {
  return blockIdx.x;
}

inline __device__ int hip_block_dim() {
  return blockDim.x;
}

inline __device__ int hip_grid_dim() {
  return gridDim.x;
}

inline __device__ int hip_global_thread_idx() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// Synchronization
inline __device__ void hip_sync_threads() {
  __syncthreads();
}

// Math constants for HIP (equivalent to CUDA's math_constants.h)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_LN2
#define M_LN2 0.693147180559945309417
#endif

#ifndef M_LN10
#define M_LN10 2.302585092994045684018
#endif

} // namespace mlx::core::rocm