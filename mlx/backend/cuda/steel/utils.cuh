// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/steel/defines.cuh"

namespace mlx::core::cu {

/**
 * Copy bytes from the global memory address pointed to by x to the smem
 * address pointed to by row_address.
 *
 * A simple wrapper over the PTX.
 */
template <int N, typename T>
__device__ inline void cp_async(uint32_t row_address, const T* x) {
  static_assert(
      N == 16 || N == 8 || N == 4,
      "cp.async is only supported for N in {4, 8, 16}.");
#if defined(MLX_CUDA_SM_80_ENABLED)
  if constexpr (N == 16) {
    asm volatile(
        "cp.async.ca.shared::cta.global [%0], [%1], 16;\n" ::"r"(row_address),
        "l"(reinterpret_cast<const int4*>(x)));
  } else if constexpr (N == 8) {
    asm volatile(
        "cp.async.ca.shared::cta.global [%0], [%1], 8;\n" ::"r"(row_address),
        "l"(reinterpret_cast<const int2*>(x)));
  } else if constexpr (N == 4) {
    asm volatile(
        "cp.async.ca.shared::cta.global [%0], [%1], 4;\n" ::"r"(row_address),
        "l"(reinterpret_cast<const int*>(x)));
  }
#endif
}

/**
 * Submit all the previous async copies to be executed.
 */
__device__ inline void cp_async_commit() {
#if defined(MLX_CUDA_SM_80_ENABLED)
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/**
 * Wait for all but N of the async copies to finish.
 */
template <int N>
__device__ inline void cp_async_wait() {
#if defined(MLX_CUDA_SM_80_ENABLED)
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
#endif
}

/**
 * Wait for all the async copies to finish.
 */
__device__ inline void cp_async_wait_all() {
  cp_async_wait<0>();
}

/**
 * Extract ``bits`` bits from the 32 bit value.
 *
 * Single instruction shift and mask.
 */
template <int bits>
__device__ inline uint32_t extract_bits(uint32_t value, int start_bit) {
  static_assert(
      bits == 2 || bits == 4 || bits == 8,
      "extract_bits only supports 2, 4, 8 for now.");
  uint32_t result;
  if constexpr (bits == 2) {
    asm("bfe.u32 %0, %1, %2, 2;" : "=r"(result) : "r"(value), "r"(start_bit));
  } else if constexpr (bits == 4) {
    asm("bfe.u32 %0, %1, %2, 4;" : "=r"(result) : "r"(value), "r"(start_bit));
  } else if constexpr (bits == 8) {
    asm("bfe.u32 %0, %1, %2, 8;" : "=r"(result) : "r"(value), "r"(start_bit));
  }
  return result;
}

} // namespace mlx::core::cu
