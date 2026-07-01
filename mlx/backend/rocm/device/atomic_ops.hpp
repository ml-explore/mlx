// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Generic atomic reduce using CAS loop
template <typename T, typename Op>
__device__ void atomic_reduce(T* addr, T val) {
  Op op;
  T old = *addr;
  T assumed;
  do {
    assumed = old;
    T new_val = op(assumed, val);
    old = atomicCAS(addr, assumed, new_val);
  } while (old != assumed);
}

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
__device__ inline void atomic_add<unsigned int>(
    unsigned int* addr,
    unsigned int val) {
  atomicAdd(addr, val);
}

// Specialization for unsigned long long
template <>
__device__ inline void atomic_add<unsigned long long>(
    unsigned long long* addr,
    unsigned long long val) {
  atomicAdd(addr, val);
}

// Specialization for int64_t (maps to long long on most platforms)
template <>
__device__ inline void atomic_add<long long>(long long* addr, long long val) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(addr),
      static_cast<unsigned long long>(val));
}

// CAS-based atomic add for unsupported types
template <typename T>
__device__ void atomic_add_general(T* addr, T val) {
  // Use CAS loop for types without native atomic support
  T old = *addr;
  T assumed;
  do {
    assumed = old;
    T new_val = assumed + val;
    // Reinterpret as unsigned int for CAS
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    unsigned int old_as_uint =
        __float_as_uint(*reinterpret_cast<float*>(&assumed));
    unsigned int new_as_uint =
        __float_as_uint(*reinterpret_cast<float*>(&new_val));
    unsigned int result = atomicCAS(addr_as_uint, old_as_uint, new_as_uint);
    old = *reinterpret_cast<T*>(&result);
  } while (old != assumed);
}

// Specialization for __half using CAS
template <>
__device__ inline void atomic_add<__half>(__half* addr, __half val) {
  // Use 32-bit CAS for half precision
  unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(
      reinterpret_cast<size_t>(addr) & ~size_t(0x3));
  unsigned int shift = (reinterpret_cast<size_t>(addr) & 0x2) ? 16 : 0;

  unsigned int old = *addr_as_uint;
  unsigned int assumed;
  do {
    assumed = old;
    __half old_half = __ushort_as_half((assumed >> shift) & 0xFFFF);
    __half new_half = __hadd(old_half, val);
    unsigned int new_val =
        (assumed & ~(0xFFFF << shift)) | (__half_as_ushort(new_half) << shift);
    old = atomicCAS(addr_as_uint, assumed, new_val);
  } while (old != assumed);
}

// Specialization for hip_bfloat16 using CAS
template <>
__device__ inline void atomic_add<hip_bfloat16>(
    hip_bfloat16* addr,
    hip_bfloat16 val) {
  // Use 32-bit CAS for bfloat16
  unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(
      reinterpret_cast<size_t>(addr) & ~size_t(0x3));
  unsigned int shift = (reinterpret_cast<size_t>(addr) & 0x2) ? 16 : 0;

  unsigned int old = *addr_as_uint;
  unsigned int assumed;
  do {
    assumed = old;
    hip_bfloat16 old_bf16;
    old_bf16.data = (assumed >> shift) & 0xFFFF;
    hip_bfloat16 new_bf16 =
        hip_bfloat16(static_cast<float>(old_bf16) + static_cast<float>(val));
    unsigned int new_val =
        (assumed & ~(0xFFFF << shift)) | (new_bf16.data << shift);
    old = atomicCAS(addr_as_uint, assumed, new_val);
  } while (old != assumed);
}

// Specialization for hipFloatComplex using CAS
template <>
__device__ inline void atomic_add<hipFloatComplex>(
    hipFloatComplex* addr,
    hipFloatComplex val) {
  // Atomic add for real and imaginary parts separately
  atomic_add(&(addr->x), val.x);
  atomic_add(&(addr->y), val.y);
}

// Atomic product using CAS loop
template <typename T>
__device__ void atomic_prod(T* addr, T val) {
  T old = *addr;
  T assumed;
  do {
    assumed = old;
    T new_val = assumed * val;
    old = atomicCAS(addr, assumed, new_val);
  } while (old != assumed);
}

// Specialization for float
template <>
__device__ inline void atomic_prod<float>(float* addr, float val) {
  unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
  unsigned int old = *addr_as_uint;
  unsigned int assumed;
  do {
    assumed = old;
    float old_float = __uint_as_float(assumed);
    float new_float = old_float * val;
    old = atomicCAS(addr_as_uint, assumed, __float_as_uint(new_float));
  } while (old != assumed);
}

// Specialization for double
template <>
__device__ inline void atomic_prod<double>(double* addr, double val) {
  unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    double old_double = __longlong_as_double(assumed);
    double new_double = old_double * val;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(new_double));
  } while (old != assumed);
}

// Atomic max for various types
template <typename T>
__device__ void atomic_max(T* addr, T val) {
  atomicMax(addr, val);
}

// Specialization for float using CAS
template <>
__device__ inline void atomic_max<float>(float* addr, float val) {
  if (val < 0.0f) {
    // For negative values, use integer atomicMin on the bit representation
    int* addr_as_int = reinterpret_cast<int*>(addr);
    atomicMin(addr_as_int, __float_as_int(val));
  } else {
    // For non-negative values, use integer atomicMax
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    atomicMax(addr_as_uint, __float_as_uint(val));
  }
}

// Specialization for double using CAS
template <>
__device__ inline void atomic_max<double>(double* addr, double val) {
  unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    double old_double = __longlong_as_double(assumed);
    double new_double = (old_double > val) ? old_double : val;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(new_double));
  } while (old != assumed && __longlong_as_double(old) < val);
}

// Atomic min for various types
template <typename T>
__device__ void atomic_min(T* addr, T val) {
  atomicMin(addr, val);
}

// Specialization for float using CAS
template <>
__device__ inline void atomic_min<float>(float* addr, float val) {
  if (val < 0.0f) {
    // For negative values, use integer atomicMax on the bit representation
    int* addr_as_int = reinterpret_cast<int*>(addr);
    atomicMax(addr_as_int, __float_as_int(val));
  } else {
    // For non-negative values, use integer atomicMin
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(addr);
    atomicMin(addr_as_uint, __float_as_uint(val));
  }
}

// Specialization for double using CAS
template <>
__device__ inline void atomic_min<double>(double* addr, double val) {
  unsigned long long* addr_as_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_as_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    double old_double = __longlong_as_double(assumed);
    double new_double = (old_double < val) ? old_double : val;
    old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(new_double));
  } while (old != assumed && __longlong_as_double(old) > val);
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

// Atomic and
template <typename T>
__device__ void atomic_and(T* addr, T val) {
  atomicAnd(addr, val);
}

// Atomic or
template <typename T>
__device__ void atomic_or(T* addr, T val) {
  atomicOr(addr, val);
}

// Specialization for bool
template <>
__device__ inline void atomic_and<bool>(bool* addr, bool val) {
  if (!val) {
    // If val is false, set to false
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~size_t(0x3));
    unsigned int shift = (reinterpret_cast<size_t>(addr) & 0x3) * 8;
    atomicAnd(addr_as_uint, ~(0xFF << shift));
  }
}

template <>
__device__ inline void atomic_or<bool>(bool* addr, bool val) {
  if (val) {
    // If val is true, set to true
    unsigned int* addr_as_uint = reinterpret_cast<unsigned int*>(
        reinterpret_cast<size_t>(addr) & ~size_t(0x3));
    unsigned int shift = (reinterpret_cast<size_t>(addr) & 0x3) * 8;
    atomicOr(addr_as_uint, 0x01 << shift);
  }
}

} // namespace mlx::core::rocm
