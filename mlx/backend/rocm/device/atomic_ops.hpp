// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Atomic add for various types
template <typename T>
__device__ void atomic_add(T* addr, T val) {
  atomicAdd(addr, val);
}

// Specialization for float
template <>
__device__ inline void atomic_add<float>(float* addr, float val) {
  atomicAdd(addr, val);
}

// Specialization for double
template <>
__device__ inline void atomic_add<double>(double* addr, double val) {
  atomicAdd(addr, val);
}

// Specialization for int
template <>
__device__ inline void atomic_add<int>(int* addr, int val) {
  atomicAdd(addr, val);
}

// Specialization for unsigned int
template <>
__device__ inline void atomic_add<unsigned int>(unsigned int* addr, unsigned int val) {
  atomicAdd(addr, val);
}

// Specialization for unsigned long long
template <>
__device__ inline void atomic_add<unsigned long long>(unsigned long long* addr, unsigned long long val) {
  atomicAdd(addr, val);
}

// Atomic max for various types
template <typename T>
__device__ void atomic_max(T* addr, T val) {
  atomicMax(addr, val);
}

// Atomic min for various types
template <typename T>
__device__ void atomic_min(T* addr, T val) {
  atomicMin(addr, val);
}

// Atomic CAS (Compare-And-Swap)
template <typename T>
__device__ T atomic_cas(T* addr, T compare, T val) {
  return atomicCAS(addr, compare, val);
}

// Atomic exchange
template <typename T>
__device__ T atomic_exchange(T* addr, T val) {
  return atomicExch(addr, val);
}

} // namespace mlx::core::rocm
