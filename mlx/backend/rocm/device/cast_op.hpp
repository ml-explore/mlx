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
struct Cast<__hip_bfloat16, To> {
  __device__ To operator()(__hip_bfloat16 x) {
    return static_cast<To>(__bfloat162float(x));
  }
};

template <typename From>
struct Cast<From, __hip_bfloat16> {
  __device__ __hip_bfloat16 operator()(From x) {
    return __float2bfloat16(static_cast<float>(x));
  }
};

template <>
struct Cast<__hip_bfloat16, __hip_bfloat16> {
  __device__ __hip_bfloat16 operator()(__hip_bfloat16 x) {
    return x;
  }
};

// Conversion between half and bfloat16
template <>
struct Cast<__half, __hip_bfloat16> {
  __device__ __hip_bfloat16 operator()(__half x) {
    return __float2bfloat16(__half2float(x));
  }
};

template <>
struct Cast<__hip_bfloat16, __half> {
  __device__ __half operator()(__hip_bfloat16 x) {
    return __float2half(__bfloat162float(x));
  }
};

} // namespace mlx::core::rocm
