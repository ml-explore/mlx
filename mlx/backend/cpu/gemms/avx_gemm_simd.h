// Copyright © 2025 Apple Inc.
#pragma once

#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "mlx/backend/cpu/simd/base_simd.h"

// GEMM-private AVX2 SIMD helpers for fp16/bf16 matmul
// Note: This header requires -mavx2 -mfma -mf16c
namespace mlx::core::detail {

// Forward declarations
template <typename T, int N>
struct Simd;
template <typename T, int N>
inline Simd<T, N> load(const T* ptr);
template <typename T, int N>
inline void store(T* ptr, Simd<T, N> x);
template <typename T, int N>
inline Simd<T, N> broadcast(const T* ptr);
template <typename T, int N>
inline Simd<T, N> fma(Simd<T, N> a, Simd<T, N> b, Simd<T, N> c);

template <>
struct Simd<float, 8> {
  static constexpr int size = 8;
  __m256 value;

  Simd() : value(_mm256_setzero_ps()) {}
  Simd(float v) : value(_mm256_set1_ps(v)) {}
  explicit Simd(__m256 v) : value(v) {}
  Simd(const Simd& other) = default;
  Simd& operator=(const Simd& other) = default;
  operator __m256() const {
    return value;
  }
};

// --- Load/Store (float) ---
template <>
inline Simd<float, 8> load<float, 8>(const float* x) {
  return Simd<float, 8>(_mm256_loadu_ps(x));
}
template <>
inline void store<float, 8>(float* dst, Simd<float, 8> x) {
  _mm256_storeu_ps(dst, x.value);
}
template <>
inline Simd<float, 8> broadcast<float, 8>(const float* x) {
  return Simd<float, 8>(_mm256_broadcast_ss(x));
}

// --- Arithmetic ---
inline Simd<float, 8> operator+(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>(_mm256_add_ps(a, b));
}
inline Simd<float, 8> operator-(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>(_mm256_sub_ps(a, b));
}
inline Simd<float, 8> operator*(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>(_mm256_mul_ps(a, b));
}
inline Simd<float, 8> operator/(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<float, 8>(_mm256_div_ps(a, b));
}

// --- FMA ---
template <>
inline Simd<float, 8>
fma<float, 8>(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
  return Simd<float, 8>(_mm256_fmadd_ps(a, b, c));
}

// --- Horizontal Sum ---
inline float sum(Simd<float, 8> x) {
  __m256 val = x.value;
  __m128 vlow = _mm256_castps256_ps128(val);
  __m128 vhigh = _mm256_extractf128_ps(val, 1); // high 128
  vlow = _mm_add_ps(vlow, vhigh); // add the low 128
  __m128 shuf = _mm_movehdup_ps(vlow); // broadcast elements 3,1 to 2,0
  __m128 sums = _mm_add_ps(vlow, shuf);
  shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

// 8x8 block transpose with fp16/bf16 → fp32 conversion.
// Loads 8 rows of 8 half-precision values, converts and transposes to fp32.
template <typename T>
inline void
transpose_8x8_block(const T* src, float* dst, int src_stride, int dst_stride) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "transpose_8x8_block requires float16_t or bfloat16_t input");

  if constexpr (std::is_same_v<T, float16_t>) {
    // Load 8 rows of 8 float16 values, convert to fp32 via F16C
    __m128i row0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    __m128i row1 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + src_stride));
    __m128i row2 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 2 * src_stride));
    __m128i row3 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 3 * src_stride));
    __m128i row4 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 4 * src_stride));
    __m128i row5 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 5 * src_stride));
    __m128i row6 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 6 * src_stride));
    __m128i row7 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + 7 * src_stride));

    // Convert to fp32 (vcvtph2ps: 1/cycle throughput, 3 cycle latency)
    __m256 frow0 = _mm256_cvtph_ps(row0);
    __m256 frow1 = _mm256_cvtph_ps(row1);
    __m256 frow2 = _mm256_cvtph_ps(row2);
    __m256 frow3 = _mm256_cvtph_ps(row3);
    __m256 frow4 = _mm256_cvtph_ps(row4);
    __m256 frow5 = _mm256_cvtph_ps(row5);
    __m256 frow6 = _mm256_cvtph_ps(row6);
    __m256 frow7 = _mm256_cvtph_ps(row7);

    // Transpose via unpack / shuffle / permute
    __m256 t0 = _mm256_unpacklo_ps(frow0, frow1);
    __m256 t1 = _mm256_unpackhi_ps(frow0, frow1);
    __m256 t2 = _mm256_unpacklo_ps(frow2, frow3);
    __m256 t3 = _mm256_unpackhi_ps(frow2, frow3);
    __m256 t4 = _mm256_unpacklo_ps(frow4, frow5);
    __m256 t5 = _mm256_unpackhi_ps(frow4, frow5);
    __m256 t6 = _mm256_unpacklo_ps(frow6, frow7);
    __m256 t7 = _mm256_unpackhi_ps(frow6, frow7);

    __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    __m256 r0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    __m256 r1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    __m256 r2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    __m256 r3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    __m256 r4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    __m256 r5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    __m256 r6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    __m256 r7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

    _mm256_storeu_ps(dst + 0 * dst_stride, r0);
    _mm256_storeu_ps(dst + 1 * dst_stride, r1);
    _mm256_storeu_ps(dst + 2 * dst_stride, r2);
    _mm256_storeu_ps(dst + 3 * dst_stride, r3);
    _mm256_storeu_ps(dst + 4 * dst_stride, r4);
    _mm256_storeu_ps(dst + 5 * dst_stride, r5);
    _mm256_storeu_ps(dst + 6 * dst_stride, r6);
    _mm256_storeu_ps(dst + 7 * dst_stride, r7);
  } else { // bfloat16_t
    // bf16 → fp32: zero-extend to 32-bit, shift left 16
    __m256 rows[8];
    for (int i = 0; i < 8; i++) {
      __m128i bf16_vals_u16 = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(src + i * src_stride));
      __m256i bf16_vals_u32 = _mm256_cvtepu16_epi32(bf16_vals_u16);
      __m256i fp32_bits = _mm256_slli_epi32(bf16_vals_u32, 16);
      rows[i] = _mm256_castsi256_ps(fp32_bits);
    }

    // Transpose the 8 rows using AVX shuffles
    __m256 t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    __m256 t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    __m256 t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    __m256 t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    __m256 t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    __m256 t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    __m256 t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    __m256 t7 = _mm256_unpackhi_ps(rows[6], rows[7]);

    __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
    __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
    __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
    __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
    __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    __m256 r0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    __m256 r1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    __m256 r2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    __m256 r3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    __m256 r4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    __m256 r5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    __m256 r6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    __m256 r7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);

    _mm256_storeu_ps(dst + 0 * dst_stride, r0);
    _mm256_storeu_ps(dst + 1 * dst_stride, r1);
    _mm256_storeu_ps(dst + 2 * dst_stride, r2);
    _mm256_storeu_ps(dst + 3 * dst_stride, r3);
    _mm256_storeu_ps(dst + 4 * dst_stride, r4);
    _mm256_storeu_ps(dst + 5 * dst_stride, r5);
    _mm256_storeu_ps(dst + 6 * dst_stride, r6);
    _mm256_storeu_ps(dst + 7 * dst_stride, r7);
  }
}

// ==========================================================================
// Conversion and Combined Operations (T -> float -> T)
// T = float16_t or bfloat16_t
// ==========================================================================

// Load 8 half-precision values, convert to float8.
template <typename T>
inline Simd<float, 8> load_convert_to_float(const T* src) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "load_convert_to_float requires float16_t or bfloat16_t input for this specialization.");
  static_assert(sizeof(T) == 2, "Input type T must be 2 bytes.");

  if constexpr (std::is_same_v<T, float16_t>) {
    __m128i f16_vals = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    return Simd<float, 8>(_mm256_cvtph_ps(f16_vals));
  } else { // bfloat16_t
    // bf16 → fp32: zero-extend to 32-bit then shift left 16
    __m128i bf16_vals_u16 =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src));
    __m256i bf16_vals_u32 = _mm256_cvtepu16_epi32(bf16_vals_u16);
    __m256i fp32_bits = _mm256_slli_epi32(bf16_vals_u32, 16);
    return Simd<float, 8>(_mm256_castsi256_ps(fp32_bits));
  }
}

// fp32 → bf16 with round-to-nearest-even (AVX2).
inline __m128i convert_float_to_bfloat16_rne_avx2(__m256 src) {
  __m256i val_int = _mm256_castps_si256(src);
  __m256i bias = _mm256_set1_epi32(0x7FFF);
  __m256i rounded_val = _mm256_add_epi32(val_int, bias);
  __m256i bf16_bits_32 = _mm256_srli_epi32(rounded_val, 16);
  __m128i bf16_bits_low = _mm256_castsi256_si128(bf16_bits_32);
  __m128i bf16_bits_high = _mm256_extracti128_si256(bf16_bits_32, 1);
  // Use signed pack to preserve negative values
  return _mm_packs_epi32(bf16_bits_low, bf16_bits_high);
}

// Store float8, converting back to 8 half-precision values.
template <typename T>
inline void store_convert_from_float(T* dst, Simd<float, 8> src) {
  static_assert(
      std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>,
      "store_convert_from_float requires float16_t or bfloat16_t output for this specialization.");
  static_assert(sizeof(T) == 2, "Output type T must be 2 bytes.");

  if constexpr (std::is_same_v<T, float16_t>) {
    __m128i f16_result = _mm256_cvtps_ph(
        src.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), f16_result);
  } else { // bfloat16_t
    __m128i bf16_result = convert_float_to_bfloat16_rne_avx2(src.value);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), bf16_result);
  }
}

// 6×16 AVX2 microkernel: C[6][16] += A[6][kc] * B[kc][16]
// Uses 12 YMM accumulators + 2 B loads + 1 A broadcast = 15 registers.
template <int MR = 6, int NR = 16>
inline void micro_kernel_6x16(
    const float* __restrict A_panel,
    const float* __restrict B_panel,
    float* __restrict C_block,
    int ldc,
    int kc,
    int a_stride,
    int b_stride) {
  static_assert(MR == 6, "This kernel requires MR=6");
  static_assert(NR == 16, "This kernel requires NR=16");

  // 12 accumulators + 2 B loads + 1 A broadcast = 15 YMM registers
  __m256 c00 = _mm256_loadu_ps(C_block + 0 * ldc);
  __m256 c01 = _mm256_loadu_ps(C_block + 0 * ldc + 8);
  __m256 c10 = _mm256_loadu_ps(C_block + 1 * ldc);
  __m256 c11 = _mm256_loadu_ps(C_block + 1 * ldc + 8);
  __m256 c20 = _mm256_loadu_ps(C_block + 2 * ldc);
  __m256 c21 = _mm256_loadu_ps(C_block + 2 * ldc + 8);
  __m256 c30 = _mm256_loadu_ps(C_block + 3 * ldc);
  __m256 c31 = _mm256_loadu_ps(C_block + 3 * ldc + 8);
  __m256 c40 = _mm256_loadu_ps(C_block + 4 * ldc);
  __m256 c41 = _mm256_loadu_ps(C_block + 4 * ldc + 8);
  __m256 c50 = _mm256_loadu_ps(C_block + 5 * ldc);
  __m256 c51 = _mm256_loadu_ps(C_block + 5 * ldc + 8);

  // Prefetch B and A data 8 iterations ahead into L1
  constexpr int PF_DIST = 8;

  for (int k = 0; k < kc; ++k) {
    const float* b_ptr = B_panel + k * b_stride;
    const float* a_ptr = A_panel + k * a_stride;

    // Prefetch next B and A rows into L1
    if (k + PF_DIST < kc) {
      _mm_prefetch(
          reinterpret_cast<const char*>(B_panel + (k + PF_DIST) * b_stride),
          _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(B_panel + (k + PF_DIST) * b_stride + 8),
          _MM_HINT_T0);
      _mm_prefetch(
          reinterpret_cast<const char*>(A_panel + (k + PF_DIST) * a_stride),
          _MM_HINT_T0);
    }

    __m256 b0 = _mm256_loadu_ps(b_ptr);
    __m256 b1 = _mm256_loadu_ps(b_ptr + 8);

    __m256 a;
    a = _mm256_broadcast_ss(a_ptr + 0);
    c00 = _mm256_fmadd_ps(a, b0, c00);
    c01 = _mm256_fmadd_ps(a, b1, c01);

    a = _mm256_broadcast_ss(a_ptr + 1);
    c10 = _mm256_fmadd_ps(a, b0, c10);
    c11 = _mm256_fmadd_ps(a, b1, c11);

    a = _mm256_broadcast_ss(a_ptr + 2);
    c20 = _mm256_fmadd_ps(a, b0, c20);
    c21 = _mm256_fmadd_ps(a, b1, c21);

    a = _mm256_broadcast_ss(a_ptr + 3);
    c30 = _mm256_fmadd_ps(a, b0, c30);
    c31 = _mm256_fmadd_ps(a, b1, c31);

    a = _mm256_broadcast_ss(a_ptr + 4);
    c40 = _mm256_fmadd_ps(a, b0, c40);
    c41 = _mm256_fmadd_ps(a, b1, c41);

    a = _mm256_broadcast_ss(a_ptr + 5);
    c50 = _mm256_fmadd_ps(a, b0, c50);
    c51 = _mm256_fmadd_ps(a, b1, c51);
  }

  _mm256_storeu_ps(C_block + 0 * ldc, c00);
  _mm256_storeu_ps(C_block + 0 * ldc + 8, c01);
  _mm256_storeu_ps(C_block + 1 * ldc, c10);
  _mm256_storeu_ps(C_block + 1 * ldc + 8, c11);
  _mm256_storeu_ps(C_block + 2 * ldc, c20);
  _mm256_storeu_ps(C_block + 2 * ldc + 8, c21);
  _mm256_storeu_ps(C_block + 3 * ldc, c30);
  _mm256_storeu_ps(C_block + 3 * ldc + 8, c31);
  _mm256_storeu_ps(C_block + 4 * ldc, c40);
  _mm256_storeu_ps(C_block + 4 * ldc + 8, c41);
  _mm256_storeu_ps(C_block + 5 * ldc, c50);
  _mm256_storeu_ps(C_block + 5 * ldc + 8, c51);
}

} // namespace mlx::core::detail
