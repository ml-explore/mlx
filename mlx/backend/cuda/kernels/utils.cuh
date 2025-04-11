// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/cucomplex_math.cuh"
#include "mlx/backend/cuda/kernels/fp16_math.cuh"

#include <cuda/std/array>
#include <cuda/std/limits>

namespace mlx::core {

// The clang-format is not friendly with "#pragma unroll" so use a macro.
#define CUDA_UNROLL _Pragma("unroll")

// All existing NVIDIA hardware has fixed 32 warp size. Though there is warpSize
// built-in variable, using it would prevent compile-time optimizations.
#define WARP_SIZE 32

// The maximum maxThreadsPerBlock value. Some code use it to define the size of
// shared memory.
// TODO: Kernels ported from Metal assume this number to be <= 1024, we should
// figure out if this number could be larger in CUDA - if so some kernels need
// rewritten.
#define MAX_BLOCK_DIM 1024

// (a + b - 1) / b
using ::cuda::ceil_div;

constexpr __host__ __device__ dim3 ceil_div(dim3 a, dim3 b) {
  return {ceil_div(a.x, b.x), ceil_div(a.y, b.y), ceil_div(a.z, b.z)};
}

namespace mxcuda {

///////////////////////////////////////////////////////////////////////////////
// CUDA kernel utils
///////////////////////////////////////////////////////////////////////////////

// To pass shape/strides to kernels via constant memory, their size must be
// known at compile time.
#define MAX_NDIM 8

using Shape = cuda::std::array<int32_t, MAX_NDIM>;
using Strides = cuda::std::array<int64_t, MAX_NDIM>;

// Utility to copy data from vector to array in host.
template <typename T>
inline cuda::std::array<T, MAX_NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error("ndim can not be larger than 8.");
  }
  cuda::std::array<T, MAX_NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

// Helper macros for dispatch macros (see below).
#define MLX_INTERNAL_IF_CASE(DIM, THREADS, BLOCK_DIM, ...) \
  if (THREADS <= DIM) {                                    \
    constexpr uint32_t BLOCK_DIM = DIM;                    \
    __VA_ARGS__;                                           \
  }

#define MLX_INTERNAL_IF_CASE_DIMS(...)   \
  MLX_INTERNAL_IF_CASE(32, __VA_ARGS__)  \
  MLX_INTERNAL_IF_CASE(64, __VA_ARGS__)  \
  MLX_INTERNAL_IF_CASE(128, __VA_ARGS__) \
  MLX_INTERNAL_IF_CASE(256, __VA_ARGS__) \
  MLX_INTERNAL_IF_CASE(512, __VA_ARGS__) \
  MLX_INTERNAL_IF_CASE(1024, __VA_ARGS__)

// Some kernels use CUB which requires block_dim to be known at compile-time,
// use this to dispatch fixed block_dim from dynamic total_threads.
#define MLX_GET_BLOCK_DIM(THREADS, BLOCK_DIM, ...) \
  MLX_INTERNAL_IF_CASE_DIMS(THREADS, BLOCK_DIM, __VA_ARGS__)

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static constexpr U max = cuda::std::numeric_limits<U>::max();
  static constexpr U min = cuda::std::numeric_limits<U>::min();
  static constexpr U finite_max = cuda::std::numeric_limits<U>::max();
  static constexpr U finite_min = cuda::std::numeric_limits<U>::min();
};

template <>
struct Limits<bool> {
  static constexpr bool max = true;
  static constexpr bool min = false;
};

template <>
struct Limits<cuComplex> {
  static constexpr cuComplex max = {
      cuda::std::numeric_limits<float>::infinity(),
      cuda::std::numeric_limits<float>::infinity()};
  static constexpr cuComplex min = {
      -cuda::std::numeric_limits<float>::infinity(),
      -cuda::std::numeric_limits<float>::infinity()};
};

// Like MLX_FORALL_FLOAT_TYPES but use CUDA types.
#define MLX_FORALL_CUDA_FLOAT_TYPES(_) \
  _(float, float32)                    \
  _(double, float64)                   \
  _(__half, float16)                   \
  _(__nv_bfloat16, bfloat16)

// Some CCCL/CUDA combinations do not provide constexpr limits for half types.
#define SPECIALIZE_FloatLimits(CPP_TYPE, DTYPE)                          \
  template <>                                                            \
  struct Limits<CPP_TYPE> {                                              \
    static constexpr CPP_TYPE max = infinite_value<CPP_TYPE>();          \
    static constexpr CPP_TYPE min = negative_infinite_value<CPP_TYPE>(); \
    static constexpr CPP_TYPE finite_max = finite_max_value<CPP_TYPE>(); \
    static constexpr CPP_TYPE finite_min = finite_min_value<CPP_TYPE>(); \
  };

MLX_FORALL_CUDA_FLOAT_TYPES(SPECIALIZE_FloatLimits)

#undef SPECIALIZE_FloatLimits

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(IdxT elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
inline __host__ __device__ IdxT
elem_to_loc(uint3 elem, const int* shape, const int64_t* strides, int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT elem_to_loc_1(uint elem, int64_t stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT elem_to_loc_2(uint2 elem, int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
inline __host__ __device__ IdxT elem_to_loc_3(uint3 elem, int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[0]);
}

} // namespace mxcuda

} // namespace mlx::core
