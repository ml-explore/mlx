// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// HIP/ROCm equivalents of CUDA half precision math functions
inline __device__ __half2 h2sin(__half2 x) {
  return __half2{hsin(x.x), hsin(x.y)};
}

inline __device__ __half2 h2cos(__half2 x) {
  return __half2{hcos(x.x), hcos(x.y)};
}

inline __device__ __half2 h2exp(__half2 x) {
  return __half2{hexp(x.x), hexp(x.y)};
}

inline __device__ __half2 h2log(__half2 x) {
  return __half2{hlog(x.x), hlog(x.y)};
}

inline __device__ __half2 h2sqrt(__half2 x) {
  return __half2{hsqrt(x.x), hsqrt(x.y)};
}

inline __device__ __half2 h2rsqrt(__half2 x) {
  return __half2{hrsqrt(x.x), hrsqrt(x.y)};
}

inline __device__ __half2 h2ceil(__half2 x) {
  return __half2{hceil(x.x), hceil(x.y)};
}

inline __device__ __half2 h2floor(__half2 x) {
  return __half2{hfloor(x.x), hfloor(x.y)};
}

inline __device__ __half2 h2rint(__half2 x) {
  return __half2{hrint(x.x), hrint(x.y)};
}

inline __device__ __half2 h2trunc(__half2 x) {
  return __half2{htrunc(x.x), htrunc(x.y)};
}

// Additional math functions for half precision
inline __device__ __half habs(__half x) {
  return __half{fabsf(__half2float(x))};
}

inline __device__ __half2 h2abs(__half2 x) {
  return __half2{habs(x.x), habs(x.y)};
}

inline __device__ __half hneg(__half x) {
  return __half{-__half2float(x)};
}

inline __device__ __half2 h2neg(__half2 x) {
  return __half2{hneg(x.x), hneg(x.y)};
}

// BFloat16 support functions
#ifdef __HIP_BFLOAT16__
inline __device__ __hip_bfloat16 habs(__hip_bfloat16 x) {
  return __hip_bfloat16{fabsf(__bfloat162float(x))};
}

inline __device__ __hip_bfloat162 h2abs(__hip_bfloat162 x) {
  return __hip_bfloat162{habs(x.x), habs(x.y)};
}

inline __device__ __hip_bfloat16 hneg(__hip_bfloat16 x) {
  return __hip_bfloat16{-__bfloat162float(x)};
}

inline __device__ __hip_bfloat162 h2neg(__hip_bfloat162 x) {
  return __hip_bfloat162{hneg(x.x), hneg(x.y)};
}
#endif

} // namespace mlx::core::rocm