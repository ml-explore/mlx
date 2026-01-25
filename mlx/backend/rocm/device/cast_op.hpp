// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

// Cast operation for type conversion
template <typename From, typename To>
struct Cast {
  __device__ To operator()(From x) {
    return static_cast<To>(x);
  }
};

// Specializations for half types
template <typename To>
struct Cast<__half, To> {
  __device__ To operator()(__half x) {
    return static_cast<To>(__half2float(x));
  }
};

template <typename From>
struct Cast<From, __half> {
  __device__ __half operator()(From x) {
    return __float2half(static_cast<float>(x));
  }
};

template <>
struct Cast<__half, __half> {
  __device__ __half operator()(__half x) {
    return x;
  }
};

// Specializations for bfloat16 types
template <typename To>
struct Cast<hip_bfloat16, To> {
  __device__ To operator()(hip_bfloat16 x) {
    return static_cast<To>(static_cast<float>(x));
  }
};

template <typename From>
struct Cast<From, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(From x) {
    return hip_bfloat16(static_cast<float>(x));
  }
};

template <>
struct Cast<hip_bfloat16, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(hip_bfloat16 x) {
    return x;
  }
};

// Conversion between half and bfloat16
template <>
struct Cast<__half, hip_bfloat16> {
  __device__ hip_bfloat16 operator()(__half x) {
    return hip_bfloat16(__half2float(x));
  }
};

template <>
struct Cast<hip_bfloat16, __half> {
  __device__ __half operator()(hip_bfloat16 x) {
    return __float2half(static_cast<float>(x));
  }
};

} // namespace mlx::core::rocm
