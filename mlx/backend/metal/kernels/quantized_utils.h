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
  using scale_type = T;

  template <typename T>
  static inline T dequantize_scale(T s) {
    return s;
  }

  // Single-arg version returns raw value (for use in dot_key where
  // dequantization is applied separately)
  template <typename T>
  static inline T dequantize_value(uint8_t v) {
    return T(v);
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
  using scale_type = uint8_t;

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
  using scale_type = uint8_t;

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
  using scale_type = uint8_t;

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

// Pack metadata and unpackers for arbitrary bit-widths (wsize fixed at 32 bits)
template <int bits>
struct PackInfo {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "PackInfo only supports bits in {2,3,4,5,6,8}");

  static constant constexpr int pack_factor =
      (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : 32 / bits);
  static constant constexpr int bytes_per_pack =
      ((bits & (bits - 1)) == 0) ? 4 : (bits == 5 ? 5 : 3);
};

template <int bits>
struct PackReader;

template <>
struct PackReader<2> {
  static constant constexpr int pack_factor = PackInfo<2>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<2>::bytes_per_pack;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p, thread uint8_t (&out)[pack_factor]) {
    uint32_t v = *(reinterpret_cast<const device uint32_t*>(p));
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; ++i) {
      out[i] = (v >> (2 * i)) & 0x03;
    }
  }
};

template <>
struct PackReader<3> {
  static constant constexpr int pack_factor = PackInfo<3>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<3>::bytes_per_pack;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p, thread uint8_t (&out)[pack_factor]) {
    uint8_t w0 = p[0];
    uint8_t w1 = p[1];
    uint8_t w2 = p[2];
    out[0] = w0 & 0x07;
    out[1] = (w0 >> 3) & 0x07;
    out[2] = ((w0 >> 6) | ((w1 & 0x01) << 2)) & 0x07;
    out[3] = (w1 >> 1) & 0x07;
    out[4] = (w1 >> 4) & 0x07;
    out[5] = ((w1 >> 7) | ((w2 & 0x03) << 1)) & 0x07;
    out[6] = (w2 >> 2) & 0x07;
    out[7] = (w2 >> 5) & 0x07;
  }
};

template <>
struct PackReader<4> {
  static constant constexpr int pack_factor = PackInfo<4>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<4>::bytes_per_pack;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p, thread uint8_t (&out)[pack_factor]) {
    uint32_t v = *(reinterpret_cast<const device uint32_t*>(p));
#pragma clang loop unroll(full)
    for (int i = 0; i < pack_factor; ++i) {
      out[i] = (v >> (4 * i)) & 0x0f;
    }
  }
};

template <>
struct PackReader<5> {
  static constant constexpr int pack_factor = PackInfo<5>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<5>::bytes_per_pack;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p, thread uint8_t (&out)[pack_factor]) {
    uint8_t w0 = p[0];
    uint8_t w1 = p[1];
    uint8_t w2 = p[2];
    uint8_t w3 = p[3];
    uint8_t w4 = p[4];
    out[0] = w0 & 0x1f;
    out[1] = ((w0 >> 5) | ((w1 & 0x03) << 3)) & 0x1f;
    out[2] = (w1 >> 2) & 0x1f;
    out[3] = ((w1 >> 7) | ((w2 & 0x0f) << 1)) & 0x1f;
    out[4] = ((w2 >> 4) | ((w3 & 0x01) << 4)) & 0x1f;
    out[5] = (w3 >> 1) & 0x1f;
    out[6] = ((w3 >> 6) | ((w4 & 0x07) << 2)) & 0x1f;
    out[7] = (w4 >> 3) & 0x1f;
  }
};

template <>
struct PackReader<6> {
  static constant constexpr int pack_factor = PackInfo<6>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<6>::bytes_per_pack;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p, thread uint8_t (&out)[pack_factor]) {
    uint8_t w0 = p[0];
    uint8_t w1 = p[1];
    uint8_t w2 = p[2];
    out[0] = w0 & 0x3f;
    out[1] = ((w0 >> 6) | ((w1 & 0x0f) << 2)) & 0x3f;
    out[2] = ((w1 >> 4) | ((w2 & 0x03) << 4)) & 0x3f;
    out[3] = (w2 >> 2) & 0x3f;
  }
};

template <>
struct PackReader<8> {
  static constant constexpr int pack_factor = PackInfo<8>::pack_factor;
  static constant constexpr int bytes_per_pack = PackInfo<8>::bytes_per_pack;

  [[gnu::always_inline]] static void load(
      const device uint8_t* p, thread uint8_t (&out)[pack_factor]) {
    uint32_t v = *(reinterpret_cast<const device uint32_t*>(p));
    out[0] = v & 0xff;
    out[1] = (v >> 8) & 0xff;
    out[2] = (v >> 16) & 0xff;
    out[3] = (v >> 24) & 0xff;
  }
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
