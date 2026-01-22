// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/fp4.h"
#include "mlx/backend/metal/kernels/fp8.h"

enum class QuantMode { Affine, Mxfp4, Mxfp8, Nvfp4 };

template <QuantMode mode>
struct QuantTraits;

// Affine quantization: scale * val + bias
template <>
struct QuantTraits<QuantMode::Affine> {
  static constant constexpr int default_group_size = 64;
  static constant constexpr int default_bits = 4;
  static constant constexpr int group_size = default_group_size;
  static constant constexpr int bits = default_bits;
  static constant constexpr bool has_bias = true;

  template <typename T>
  static inline T dequantize_scale(T s) {
    return s;
  }

  template <typename T>
  static inline T dequantize_value(uint8_t v, T scale, T bias) {
    return fma(scale, T(v), bias);
  }

  template <typename T>
  static inline T dequantize(uint8_t v, T scale, T bias) {
    return fma(scale, T(v), bias);
  }
};

// MXFP4: fp4_e2m1 data, fp8_e8m0 scale (power-of-2), group_size=32
template <>
struct QuantTraits<QuantMode::Mxfp4> {
  static constant constexpr int group_size = 32;
  static constant constexpr int bits = 4;
  static constant constexpr bool has_bias = false;

  template <typename T>
  static inline T dequantize_scale(uint8_t s) {
    return T(*(thread fp8_e8m0*)(&s));
  }

  template <typename T>
  static inline T dequantize_value(uint8_t v) {
    return T(*(thread fp4_e2m1*)(&v));
  }

  template <typename T>
  static inline T dequantize(uint8_t v, T scale, T /*bias*/) {
    return scale * dequantize_value<T>(v);
  }
};

// NVFP4: fp4_e2m1 data, fp8_e4m3 scale (with mantissa), group_size=16
template <>
struct QuantTraits<QuantMode::Nvfp4> {
  static constant constexpr int group_size = 16;
  static constant constexpr int bits = 4;
  static constant constexpr bool has_bias = false;

  template <typename T>
  static inline T dequantize_scale(uint8_t s) {
    return T(*(thread fp8_e4m3*)(&s));
  }

  template <typename T>
  static inline T dequantize_value(uint8_t v) {
    return T(*(thread fp4_e2m1*)(&v));
  }

  template <typename T>
  static inline T dequantize(uint8_t v, T scale, T /*bias*/) {
    return scale * dequantize_value<T>(v);
  }
};

// MXFP8: fp8_e4m3 data, fp8_e8m0 scale, group_size=32
template <>
struct QuantTraits<QuantMode::Mxfp8> {
  static constant constexpr int group_size = 32;
  static constant constexpr int bits = 8;
  static constant constexpr bool has_bias = false;

  template <typename T>
  static inline T dequantize_scale(uint8_t s) {
    return T(*(thread fp8_e8m0*)(&s));
  }

  template <typename T>
  static inline T dequantize_value(uint8_t v) {
    return T(*(thread fp8_e4m3*)(&v));
  }

  template <typename T>
  static inline T dequantize(uint8_t v, T scale, T /*bias*/) {
    return scale * dequantize_value<T>(v);
  }
};

// Compile-time LoadType selector by bit-width
template <int bits>
struct LoadType {
  using type = uint32_t;
};

template <>
struct LoadType<4> {
  using type = uint16_t;
};

// Helpers to fetch mode-specific defaults (affine uses default_* values)
template <QuantMode mode>
constexpr int get_group_size() {
  if constexpr (mode == QuantMode::Affine) {
    return QuantTraits<mode>::default_group_size;
  } else {
    return QuantTraits<mode>::group_size;
  }
}

template <QuantMode mode>
constexpr int get_bits() {
  if constexpr (mode == QuantMode::Affine) {
    return QuantTraits<mode>::default_bits;
  } else {
    return QuantTraits<mode>::bits;
  }
}

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_aligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    loader_a.load_unsafe();
    loader_b.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <
    bool rows_aligned,
    bool cols_aligned,
    bool transpose,
    typename T,
    typename mma_t,
    typename loader_a_t,
    typename loader_b_t>
METAL_FUNC void gemm_loop_unaligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations,
    const short tgp_bm,
    const short tgp_bn,
    const short tgp_bk) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    if (rows_aligned) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(short2(tgp_bk, tgp_bm));
    }
    if (cols_aligned) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(
          transpose ? short2(tgp_bk, tgp_bn) : short2(tgp_bn, tgp_bk));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_finalize(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const short2 tile_a,
    const short2 tile_b) {
  loader_a.load_safe(tile_a);
  loader_b.load_safe(tile_b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.mma(As, Bs);
}
