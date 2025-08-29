// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Atomic operations for HIP
__device__ inline float atomicAddFloat(float* address, float val) {
  return atomicAdd(address, val);
}

__device__ inline double atomicAddDouble(double* address, double val) {
  return atomicAdd(address, val);
}

__device__ inline int atomicAddInt(int* address, int val) {
  return atomicAdd(address, val);
}

__device__ inline unsigned int atomicAddUInt(
    unsigned int* address,
    unsigned int val) {
  return atomicAdd(address, val);
}

__device__ inline float atomicMaxFloat(float* address, float val) {
  return atomicMax(address, val);
}

__device__ inline float atomicMinFloat(float* address, float val) {
  return atomicMin(address, val);
}

} // namespace mlx::core::rocm