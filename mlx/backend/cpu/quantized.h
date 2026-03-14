// Copyright © 2026 Apple Inc.
//
// Shared declarations for quantized matmul and its ISA-specific headers.

#pragma once

#include <atomic>
#include <type_traits>

#include "mlx/backend/cpu/simd/simd.h"

namespace mlx::core {

// Load S elements of type T from memory and return as Simd<float, S>.
// For float, this is a direct SIMD load. For float16/bfloat16, uses
// hardware-accelerated conversion when available (F16C for float16,
// SIMD bit manipulation for bfloat16). Falls back to scalar conversion.
//
// Uses a helper struct to allow partial specialization by ISA without
// issues with if-constexpr instantiating non-existent Simd types.
template <typename T, int S>
struct LoadAsFloat {
  static inline simd::Simd<float, S> apply(const T* ptr) {
    if constexpr (std::is_same_v<T, float>) {
      return simd::load<float, S>(ptr);
    } else {
      alignas(64) float tmp[S];
      for (int i = 0; i < S; i++) {
        tmp[i] = static_cast<float>(ptr[i]);
      }
      return simd::load<float, S>(tmp);
    }
  }
};

// Convenience wrapper: load S elements of T, convert to Simd<float, S>.
// SIMD specializations (float16, bfloat16 with S=8) are provided by
// the arch-specific header (e.g. quantized_avx2.h).
template <typename T, int S>
inline simd::Simd<float, S> load_as_float(const T* ptr) {
  return LoadAsFloat<T, S>::apply(ptr);
}

// Pre-quantized activation data, shared across threads for M==1 optimization.
// When passed to _qmm_t_simd_row, skips per-thread redundant quantization.
struct PreqAct {
  const void* x_q; // int8_t* (4-bit) or int16_t* (8-bit)
  const float* x_scales;
  const float* x_group_sums;
};

// Forward declaration -- defined in quantized.cpp.
template <typename T, int bits, int group_size>
void _qmm_t_simd_row(
    T* result,
    const T* x,
    const uint32_t* w,
    const T* scales,
    const T* biases,
    int n_start,
    int n_end,
    int K,
    const PreqAct* preq = nullptr);

} // namespace mlx::core
