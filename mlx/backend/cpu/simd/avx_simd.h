// Copyright © 2026 Apple Inc.

#pragma once

#ifdef __AVX2__

// AVX2 provides 256-bit SIMD for both floating-point and integers.
// AVX1 without AVX2 is too limited (no 256-bit integer ops), so we require
// AVX2 as the minimum x86 SIMD tier.
#include "mlx/backend/cpu/simd/base_simd.h"
#include "mlx/backend/cpu/simd/x86_simd_macros.h"
#include "mlx/types/half_types.h"

#include <immintrin.h> // AVX
#include <stdint.h>
#include <cmath>
#include <cstring>

namespace mlx::core::simd {

// AVX2: 256-bit SIMD for floats, doubles, and integers
template <>
inline constexpr int max_size<float> = 8;
template <>
inline constexpr int max_size<double> = 4;
template <>
inline constexpr int max_size<int32_t> = 8;
template <>
inline constexpr int max_size<uint32_t> = 8;
template <>
inline constexpr int max_size<uint8_t> = 8;
template <>
inline constexpr int max_size<float16_t> = 8;
template <>
inline constexpr int max_size<bfloat16_t> = 8;

// ============================================================================
// int64_t / uint64_t with size 8 (pair of 256-bit registers)
// These are needed for JIT compiled kernels where _S = max_size<float> = 8,
// but an input is int64/uint64. Only broadcast and convert-to-float are needed.
// ============================================================================

template <>
struct Simd<uint64_t, 8> {
  static constexpr int size = 8;
  __m256i lo, hi; // lo holds elements 0-3, hi holds elements 4-7

  Simd() : lo(_mm256_setzero_si256()), hi(_mm256_setzero_si256()) {}
  Simd(uint64_t v)
      : lo(_mm256_set1_epi64x(static_cast<int64_t>(v))),
        hi(_mm256_set1_epi64x(static_cast<int64_t>(v))) {}

  uint64_t operator[](int idx) const {
    alignas(32) uint64_t tmp[4];
    if (idx < 4) {
      _mm256_store_si256((__m256i*)tmp, lo);
      return tmp[idx];
    } else {
      _mm256_store_si256((__m256i*)tmp, hi);
      return tmp[idx - 4];
    }
  }
};

template <>
struct Simd<int64_t, 8> {
  static constexpr int size = 8;
  __m256i lo, hi;

  Simd() : lo(_mm256_setzero_si256()), hi(_mm256_setzero_si256()) {}
  Simd(int64_t v) : lo(_mm256_set1_epi64x(v)), hi(_mm256_set1_epi64x(v)) {}

  int64_t operator[](int idx) const {
    alignas(32) int64_t tmp[4];
    if (idx < 4) {
      _mm256_store_si256((__m256i*)tmp, lo);
      return tmp[idx];
    } else {
      _mm256_store_si256((__m256i*)tmp, hi);
      return tmp[idx - 4];
    }
  }
};

// ============================================================================
// Bool specializations
// ============================================================================

// Bool for 8 floats or 8 int32s (8x32-bit masks)
template <>
struct Simd<bool, 8> {
  static constexpr int size = 8;
  __m256i value;

  Simd() : value(_mm256_setzero_si256()) {}
  Simd(__m256i v) : value(v) {}
  Simd(bool v) : value(_mm256_set1_epi32(v ? -1 : 0)) {}
  Simd(Simd<uint8_t, 8> v);
};

// Bool for 4 doubles (4x64-bit masks)
template <>
struct Simd<bool, 4> {
  static constexpr int size = 4;
  __m256i value;

  Simd() : value(_mm256_setzero_si256()) {}
  Simd(__m256i v) : value(v) {}
  Simd(bool v) : value(_mm256_set1_epi64x(v ? -1LL : 0)) {}
};

DEFINE_X86_BOOL_OPS_VECTOR(8, si256, _mm256, 32)

// Bool operators for Simd<bool, 4> (64-bit element width)
inline Simd<bool, 4> operator!(Simd<bool, 4> a) {
  return Simd<bool, 4>(_mm256_xor_si256(a.value, _mm256_set1_epi64x(-1)));
}
inline Simd<bool, 4> operator||(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm256_or_si256(a.value, b.value));
}
inline Simd<bool, 4> operator&&(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm256_and_si256(a.value, b.value));
}
inline Simd<bool, 4> operator&(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm256_and_si256(a.value, b.value));
}
inline Simd<bool, 4> operator|(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm256_or_si256(a.value, b.value));
}
inline Simd<bool, 4> operator^(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm256_xor_si256(a.value, b.value));
}
inline Simd<bool, 4> operator==(Simd<bool, 4> a, Simd<bool, 4> b) {
  return !Simd<bool, 4>(_mm256_xor_si256(a.value, b.value));
}
inline Simd<bool, 4> operator!=(Simd<bool, 4> a, Simd<bool, 4> b) {
  return !(a == b);
}

// Load/Store bool types
template <>
inline Simd<bool, 8> load<bool, 8>(const bool* ptr) {
  return Simd<bool, 8>(_mm256_set_epi32(
      ptr[7] ? -1 : 0,
      ptr[6] ? -1 : 0,
      ptr[5] ? -1 : 0,
      ptr[4] ? -1 : 0,
      ptr[3] ? -1 : 0,
      ptr[2] ? -1 : 0,
      ptr[1] ? -1 : 0,
      ptr[0] ? -1 : 0));
}

template <>
inline void store<bool, 8>(bool* ptr, Simd<bool, 8> v) {
  int mask = _mm256_movemask_ps(_mm256_castsi256_ps(v.value));
  for (int i = 0; i < 8; i++) {
    ptr[i] = (mask & (1 << i)) != 0;
  }
}

template <>
inline Simd<bool, 4> load<bool, 4>(const bool* ptr) {
  return Simd<bool, 4>(_mm256_set_epi64x(
      ptr[3] ? -1LL : 0,
      ptr[2] ? -1LL : 0,
      ptr[1] ? -1LL : 0,
      ptr[0] ? -1LL : 0));
}

template <>
inline void store<bool, 4>(bool* ptr, Simd<bool, 4> v) {
  int mask = _mm256_movemask_pd(_mm256_castsi256_pd(v.value));
  ptr[0] = (mask & 1) != 0;
  ptr[1] = (mask & 2) != 0;
  ptr[2] = (mask & 4) != 0;
  ptr[3] = (mask & 8) != 0;
}

// ============================================================================
// float32x8 (8 floats in 256 bits)
// ============================================================================

template <>
struct Simd<float, 8> {
  static constexpr int size = 8;
  __m256 value;

  Simd() : value(_mm256_setzero_ps()) {}
  Simd(__m256 v) : value(v) {}
  Simd(float v) : value(_mm256_set1_ps(v)) {}
  Simd(Simd<bool, 8> v);
  Simd(Simd<int32_t, 8> v);
  Simd(Simd<uint32_t, 8> v);
  Simd(Simd<float16_t, 8> v);
  Simd(Simd<bfloat16_t, 8> v);
  Simd(Simd<uint64_t, 8> v);
  Simd(Simd<int64_t, 8> v);

  float operator[](int idx) const {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, value);
    return tmp[idx];
  }
};

// Load/Store float32x8
template <>
inline Simd<float, 8> load<float, 8>(const float* ptr) {
  return _mm256_loadu_ps(ptr);
}

template <>
inline void store<float, 8>(float* ptr, Simd<float, 8> v) {
  _mm256_storeu_ps(ptr, v.value);
}

// Arithmetic float32x8
DEFINE_X86_BINARY_OP(+, float, 8, _mm256_add_ps)
DEFINE_X86_BINARY_OP(-, float, 8, _mm256_sub_ps)
DEFINE_X86_BINARY_OP(*, float, 8, _mm256_mul_ps)
DEFINE_X86_BINARY_OP(/, float, 8, _mm256_div_ps)
inline Simd<float, 8> operator-(Simd<float, 8> a) {
  return _mm256_sub_ps(_mm256_setzero_ps(), a.value);
}

// Comparisons float32x8
DEFINE_X86_COMPARISONS_CMP(float, 8, _mm256, ps, _mm256_castps_si256)

inline Simd<bool, 8> operator!(Simd<float, 8> a) {
  return a == Simd<float, 8>(0.0f);
}

inline Simd<bool, 8> isnan(Simd<float, 8> a) {
  return Simd<bool, 8>(
      _mm256_castps_si256(_mm256_cmp_ps(a.value, a.value, _CMP_UNORD_Q)));
}

// Select (blend) float32x8
inline Simd<float, 8>
select(Simd<bool, 8> mask, Simd<float, 8> x, Simd<float, 8> y) {
  return _mm256_blendv_ps(y.value, x.value, _mm256_castsi256_ps(mask.value));
}

// Math functions float32x8
inline Simd<float, 8> abs(Simd<float, 8> a) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return _mm256_and_ps(a.value, mask);
}

DEFINE_X86_UNARY_OP(sqrt, float, 8, _mm256_sqrt_ps)

// rsqrt: 1/sqrt(x). Use sqrt+div for correct handling of special values
// (NR refinement of _mm256_rsqrt_ps produces NaN for inf/zero inputs).
inline Simd<float, 8> rsqrt(Simd<float, 8> a) {
  return Simd<float, 8>{
      _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_sqrt_ps(a.value))};
}

// recip with Newton-Raphson refinement for full float precision.
// _mm256_rcp_ps gives ~12-bit precision; one NR step brings it to ~24 bits.
// y1 = y0 * (2 - x * y0)
// Note: NR refinement produces NaN for inf inputs (inf*0=NaN), so we use
// plain division which correctly handles all special values.
inline Simd<float, 8> recip(Simd<float, 8> a) {
  return Simd<float, 8>{_mm256_div_ps(_mm256_set1_ps(1.0f), a.value)};
}
DEFINE_X86_UNARY_OP(floor, float, 8, _mm256_floor_ps)
DEFINE_X86_UNARY_OP(ceil, float, 8, _mm256_ceil_ps)

inline Simd<float, 8> rint(Simd<float, 8> a) {
  return _mm256_round_ps(
      a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline Simd<float, 8> maximum(Simd<float, 8> a, Simd<float, 8> b) {
  auto out = Simd<float, 8>(_mm256_max_ps(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<float, 8> minimum(Simd<float, 8> a, Simd<float, 8> b) {
  auto out = Simd<float, 8>(_mm256_min_ps(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<float, 8>
clamp(Simd<float, 8> v, Simd<float, 8> min_val, Simd<float, 8> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

// Transcendental functions float32x8 (scalar fallback)
DEFINE_X86_TRANSCENDENTAL(exp, float, 8, 32, std::exp)
DEFINE_X86_TRANSCENDENTAL(expm1, float, 8, 32, std::expm1)
DEFINE_X86_TRANSCENDENTAL(log, float, 8, 32, std::log)
DEFINE_X86_TRANSCENDENTAL(log1p, float, 8, 32, std::log1p)
DEFINE_X86_TRANSCENDENTAL(log2, float, 8, 32, std::log2)
DEFINE_X86_TRANSCENDENTAL(log10, float, 8, 32, std::log10)
DEFINE_X86_TRANSCENDENTAL(sin, float, 8, 32, std::sin)
DEFINE_X86_TRANSCENDENTAL(cos, float, 8, 32, std::cos)
DEFINE_X86_TRANSCENDENTAL(tan, float, 8, 32, std::tan)
DEFINE_X86_TRANSCENDENTAL(asin, float, 8, 32, std::asin)
DEFINE_X86_TRANSCENDENTAL(acos, float, 8, 32, std::acos)
DEFINE_X86_TRANSCENDENTAL(atan, float, 8, 32, std::atan)
DEFINE_X86_TRANSCENDENTAL(sinh, float, 8, 32, std::sinh)
DEFINE_X86_TRANSCENDENTAL(cosh, float, 8, 32, std::cosh)
DEFINE_X86_TRANSCENDENTAL(tanh, float, 8, 32, std::tanh)
DEFINE_X86_TRANSCENDENTAL(asinh, float, 8, 32, std::asinh)
DEFINE_X86_TRANSCENDENTAL(acosh, float, 8, 32, std::acosh)
DEFINE_X86_TRANSCENDENTAL(atanh, float, 8, 32, std::atanh)

DEFINE_X86_BINARY_TRANSCENDENTAL(atan2, float, 8, 32, std::atan2)
DEFINE_X86_BINARY_TRANSCENDENTAL(pow, float, 8, 32, std::pow)
// Floored remainder (Python/numpy semantics): fmod + sign correction
inline Simd<float, 8> remainder(Simd<float, 8> a, Simd<float, 8> b) {
  alignas(32) float tmp_a[8], tmp_b[8];
  store<float, 8>(tmp_a, a);
  store<float, 8>(tmp_b, b);
  for (int i = 0; i < 8; ++i) {
    float r = std::fmod(tmp_a[i], tmp_b[i]);
    if (r != 0 && (std::signbit(r) != std::signbit(tmp_b[i]))) {
      r += tmp_b[i];
    }
    tmp_a[i] = r;
  }
  return load<float, 8>(tmp_a);
}

// Logical operators float32x8
inline Simd<float, 8> operator&&(Simd<float, 8> a, Simd<float, 8> b) {
  __m256 zero = _mm256_setzero_ps();
  __m256 mask_a = _mm256_cmp_ps(a.value, zero, _CMP_NEQ_OQ);
  __m256 mask_b = _mm256_cmp_ps(b.value, zero, _CMP_NEQ_OQ);
  return _mm256_and_ps(mask_a, mask_b);
}

inline Simd<float, 8> operator||(Simd<float, 8> a, Simd<float, 8> b) {
  __m256 zero = _mm256_setzero_ps();
  __m256 mask_a = _mm256_cmp_ps(a.value, zero, _CMP_NEQ_OQ);
  __m256 mask_b = _mm256_cmp_ps(b.value, zero, _CMP_NEQ_OQ);
  return _mm256_or_ps(mask_a, mask_b);
}

// FMA float32x8
#if defined(__FMA__)
inline Simd<float, 8>
fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
  return _mm256_fmadd_ps(a.value, b.value, c.value);
}
#else
inline Simd<float, 8>
fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
  return _mm256_add_ps(_mm256_mul_ps(a.value, b.value), c.value);
}
#endif

// Reductions float32x8
inline float sum(Simd<float, 8> v) {
  __m128 hi = _mm256_extractf128_ps(v.value, 1);
  __m128 lo = _mm256_castps256_ps128(v.value);
  __m128 sum128 = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(sum128);
  __m128 sums = _mm_add_ps(sum128, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

inline float max(Simd<float, 8> v) {
  __m128 hi = _mm256_extractf128_ps(v.value, 1);
  __m128 lo = _mm256_castps256_ps128(v.value);
  __m128 max128 = _mm_max_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(max128);
  __m128 maxs = _mm_max_ps(max128, shuf);
  shuf = _mm_movehl_ps(shuf, maxs);
  maxs = _mm_max_ss(maxs, shuf);
  return _mm_cvtss_f32(maxs);
}

inline float min(Simd<float, 8> v) {
  __m128 hi = _mm256_extractf128_ps(v.value, 1);
  __m128 lo = _mm256_castps256_ps128(v.value);
  __m128 min128 = _mm_min_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(min128);
  __m128 mins = _mm_min_ps(min128, shuf);
  shuf = _mm_movehl_ps(shuf, mins);
  mins = _mm_min_ss(mins, shuf);
  return _mm_cvtss_f32(mins);
}

inline float prod(Simd<float, 8> v) {
  __m128 hi = _mm256_extractf128_ps(v.value, 1);
  __m128 lo = _mm256_castps256_ps128(v.value);
  __m128 prod128 = _mm_mul_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(prod128);
  __m128 prods = _mm_mul_ps(prod128, shuf);
  shuf = _mm_movehl_ps(shuf, prods);
  prods = _mm_mul_ss(prods, shuf);
  return _mm_cvtss_f32(prods);
}

// ============================================================================
// double64x4 (4 doubles in 256 bits)
// ============================================================================

template <>
struct Simd<double, 4> {
  static constexpr int size = 4;
  __m256d value;

  Simd() : value(_mm256_setzero_pd()) {}
  Simd(__m256d v) : value(v) {}
  Simd(double v) : value(_mm256_set1_pd(v)) {}
  Simd(Simd<bool, 4> v);

  template <typename U>
  Simd(Simd<U, 4> v);

  double operator[](int idx) const {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, value);
    return tmp[idx];
  }
};

// Load/Store double64x4
template <>
inline Simd<double, 4> load<double, 4>(const double* ptr) {
  return _mm256_loadu_pd(ptr);
}

template <>
inline void store<double, 4>(double* ptr, Simd<double, 4> v) {
  _mm256_storeu_pd(ptr, v.value);
}

// Arithmetic double64x4
DEFINE_X86_BINARY_OP(+, double, 4, _mm256_add_pd)
DEFINE_X86_BINARY_OP(-, double, 4, _mm256_sub_pd)
DEFINE_X86_BINARY_OP(*, double, 4, _mm256_mul_pd)
DEFINE_X86_BINARY_OP(/, double, 4, _mm256_div_pd)
inline Simd<double, 4> operator-(Simd<double, 4> a) {
  return _mm256_sub_pd(_mm256_setzero_pd(), a.value);
}

// Bitwise double64x4
inline Simd<double, 4> operator&(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_and_pd(a.value, b.value);
}
inline Simd<double, 4> operator|(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_or_pd(a.value, b.value);
}
inline Simd<double, 4> operator^(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_xor_pd(a.value, b.value);
}

// Comparisons double64x4
DEFINE_X86_COMPARISONS_CMP(double, 4, _mm256, pd, _mm256_castpd_si256)

inline Simd<bool, 4> operator!(Simd<double, 4> a) {
  return a == Simd<double, 4>(0.0);
}

// Select (blend) double64x4
inline Simd<double, 4>
select(Simd<bool, 4> mask, Simd<double, 4> x, Simd<double, 4> y) {
  return _mm256_blendv_pd(y.value, x.value, _mm256_castsi256_pd(mask.value));
}

inline Simd<bool, 4> isnan(Simd<double, 4> a) {
  return Simd<bool, 4>(
      _mm256_castpd_si256(_mm256_cmp_pd(a.value, a.value, _CMP_UNORD_Q)));
}

// Math functions double64x4
inline Simd<double, 4> abs(Simd<double, 4> a) {
  __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
  return _mm256_and_pd(a.value, mask);
}

DEFINE_X86_UNARY_OP(sqrt, double, 4, _mm256_sqrt_pd)

inline Simd<double, 4> rsqrt(Simd<double, 4> a) {
  return Simd<double, 4>(1.0) / sqrt(a);
}

inline Simd<double, 4> recip(Simd<double, 4> a) {
  return Simd<double, 4>(1.0) / a;
}

DEFINE_X86_UNARY_OP(floor, double, 4, _mm256_floor_pd)
DEFINE_X86_UNARY_OP(ceil, double, 4, _mm256_ceil_pd)

inline Simd<double, 4> rint(Simd<double, 4> a) {
  return _mm256_round_pd(
      a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline Simd<double, 4> maximum(Simd<double, 4> a, Simd<double, 4> b) {
  auto out = Simd<double, 4>(_mm256_max_pd(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 4> minimum(Simd<double, 4> a, Simd<double, 4> b) {
  auto out = Simd<double, 4>(_mm256_min_pd(a.value, b.value));
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 4>
clamp(Simd<double, 4> v, Simd<double, 4> min_val, Simd<double, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

// Transcendental functions double64x4 (scalar fallback)
DEFINE_X86_TRANSCENDENTAL(exp, double, 4, 32, std::exp)
DEFINE_X86_TRANSCENDENTAL(expm1, double, 4, 32, std::expm1)
DEFINE_X86_TRANSCENDENTAL(log, double, 4, 32, std::log)
DEFINE_X86_TRANSCENDENTAL(log1p, double, 4, 32, std::log1p)
DEFINE_X86_TRANSCENDENTAL(log2, double, 4, 32, std::log2)
DEFINE_X86_TRANSCENDENTAL(log10, double, 4, 32, std::log10)
DEFINE_X86_TRANSCENDENTAL(sin, double, 4, 32, std::sin)
DEFINE_X86_TRANSCENDENTAL(cos, double, 4, 32, std::cos)
DEFINE_X86_TRANSCENDENTAL(tan, double, 4, 32, std::tan)
DEFINE_X86_TRANSCENDENTAL(asin, double, 4, 32, std::asin)
DEFINE_X86_TRANSCENDENTAL(acos, double, 4, 32, std::acos)
DEFINE_X86_TRANSCENDENTAL(atan, double, 4, 32, std::atan)
DEFINE_X86_TRANSCENDENTAL(sinh, double, 4, 32, std::sinh)
DEFINE_X86_TRANSCENDENTAL(cosh, double, 4, 32, std::cosh)
DEFINE_X86_TRANSCENDENTAL(tanh, double, 4, 32, std::tanh)
DEFINE_X86_TRANSCENDENTAL(asinh, double, 4, 32, std::asinh)
DEFINE_X86_TRANSCENDENTAL(acosh, double, 4, 32, std::acosh)
DEFINE_X86_TRANSCENDENTAL(atanh, double, 4, 32, std::atanh)

DEFINE_X86_BINARY_TRANSCENDENTAL(atan2, double, 4, 32, std::atan2)
DEFINE_X86_BINARY_TRANSCENDENTAL(pow, double, 4, 32, std::pow)
// Floored remainder (Python/numpy semantics): fmod + sign correction
inline Simd<double, 4> remainder(Simd<double, 4> a, Simd<double, 4> b) {
  alignas(32) double tmp_a[4], tmp_b[4];
  store<double, 4>(tmp_a, a);
  store<double, 4>(tmp_b, b);
  for (int i = 0; i < 4; ++i) {
    double r = std::fmod(tmp_a[i], tmp_b[i]);
    if (r != 0 && (std::signbit(r) != std::signbit(tmp_b[i]))) {
      r += tmp_b[i];
    }
    tmp_a[i] = r;
  }
  return load<double, 4>(tmp_a);
}

// Logical operators double64x4
inline Simd<double, 4> operator&&(Simd<double, 4> a, Simd<double, 4> b) {
  __m256d zero = _mm256_setzero_pd();
  __m256d mask_a = _mm256_cmp_pd(a.value, zero, _CMP_NEQ_OQ);
  __m256d mask_b = _mm256_cmp_pd(b.value, zero, _CMP_NEQ_OQ);
  return _mm256_and_pd(mask_a, mask_b);
}

inline Simd<double, 4> operator||(Simd<double, 4> a, Simd<double, 4> b) {
  __m256d zero = _mm256_setzero_pd();
  __m256d mask_a = _mm256_cmp_pd(a.value, zero, _CMP_NEQ_OQ);
  __m256d mask_b = _mm256_cmp_pd(b.value, zero, _CMP_NEQ_OQ);
  return _mm256_or_pd(mask_a, mask_b);
}

// FMA double64x4
inline Simd<double, 4>
fma(Simd<double, 4> a, Simd<double, 4> b, Simd<double, 4> c) {
  return _mm256_add_pd(_mm256_mul_pd(a.value, b.value), c.value);
}

// Reductions double64x4
inline double sum(Simd<double, 4> v) {
  __m128d hi = _mm256_extractf128_pd(v.value, 1);
  __m128d lo = _mm256_castpd256_pd128(v.value);
  __m128d sum128 = _mm_add_pd(lo, hi);
  __m128d shuf = _mm_shuffle_pd(sum128, sum128, 1);
  sum128 = _mm_add_sd(sum128, shuf);
  return _mm_cvtsd_f64(sum128);
}

inline double max(Simd<double, 4> v) {
  __m128d hi = _mm256_extractf128_pd(v.value, 1);
  __m128d lo = _mm256_castpd256_pd128(v.value);
  __m128d max128 = _mm_max_pd(lo, hi);
  __m128d shuf = _mm_shuffle_pd(max128, max128, 1);
  max128 = _mm_max_sd(max128, shuf);
  return _mm_cvtsd_f64(max128);
}

inline double min(Simd<double, 4> v) {
  __m128d hi = _mm256_extractf128_pd(v.value, 1);
  __m128d lo = _mm256_castpd256_pd128(v.value);
  __m128d min128 = _mm_min_pd(lo, hi);
  __m128d shuf = _mm_shuffle_pd(min128, min128, 1);
  min128 = _mm_min_sd(min128, shuf);
  return _mm_cvtsd_f64(min128);
}

inline double prod(Simd<double, 4> v) {
  __m128d hi = _mm256_extractf128_pd(v.value, 1);
  __m128d lo = _mm256_castpd256_pd128(v.value);
  __m128d prod128 = _mm_mul_pd(lo, hi);
  __m128d shuf = _mm_shuffle_pd(prod128, prod128, 1);
  prod128 = _mm_mul_sd(prod128, shuf);
  return _mm_cvtsd_f64(prod128);
}

// ============================================================================
// Forward declarations needed by half-type transcendental macros
// ============================================================================

template <typename T, int N>
Simd<T, N> sigmoid(Simd<T, N> x);

// ============================================================================
// float16x8 (8 float16s stored as 128-bit, computed via float32)
// ============================================================================

template <>
struct Simd<float16_t, 8> {
  static constexpr int size = 8;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(float16_t v) {
    uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    value = _mm_set1_epi16(static_cast<short>(bits));
  }
  Simd(int v) : Simd(static_cast<float16_t>(static_cast<float>(v))) {}
  Simd(float v) : Simd(static_cast<float16_t>(v)) {}
  Simd(double v) : Simd(static_cast<float16_t>(static_cast<float>(v))) {}
  explicit Simd(Simd<float, 8> v);

  float16_t operator[](int idx) const {
    alignas(16) uint16_t tmp[8];
    _mm_store_si128((__m128i*)tmp, value);
    return *reinterpret_cast<float16_t*>(&tmp[idx]);
  }
};

// F16C hardware conversions
inline Simd<float, 8>::Simd(Simd<float16_t, 8> v) {
  value = _mm256_cvtph_ps(v.value);
}

inline Simd<float16_t, 8>::Simd(Simd<float, 8> v) {
  value = _mm256_cvtps_ph(v.value, _MM_FROUND_TO_NEAREST_INT);
}

// Load/store float16
template <>
inline Simd<float16_t, 8> load<float16_t, 8>(const float16_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <>
inline void store<float16_t, 8>(float16_t* ptr, Simd<float16_t, 8> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// All float16 operations via shared macros
DEFINE_X86_HALF_SCALAR_MUL(float16_t)
DEFINE_X86_HALF_BINARY_ARITHMETIC(float16_t)
DEFINE_X86_HALF_COMPARISONS_VIA_FLOAT(float16_t)
DEFINE_X86_HALF_SELECT_FMA_ISNAN(float16_t)
DEFINE_X86_HALF_TRANSCENDENTALS(float16_t)
DEFINE_X86_HALF_LOGICAL_OPS(float16_t)

// ============================================================================
// bfloat16x8 (8 bfloat16s stored as 128-bit, computed via float32)
// BF16 is a simple truncation of float32 (top 16 bits), so conversion
// is just bit shifting rather than F16C hardware instructions.
// ============================================================================

template <>
struct Simd<bfloat16_t, 8> {
  static constexpr int size = 8;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(bfloat16_t v) {
    uint16_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    value = _mm_set1_epi16(static_cast<short>(bits));
  }
  Simd(int v) : Simd(static_cast<bfloat16_t>(static_cast<float>(v))) {}
  Simd(float v) : Simd(static_cast<bfloat16_t>(v)) {}
  Simd(double v) : Simd(static_cast<bfloat16_t>(static_cast<float>(v))) {}
  explicit Simd(Simd<float, 8> v);

  bfloat16_t operator[](int idx) const {
    alignas(16) uint16_t tmp[8];
    _mm_store_si128((__m128i*)tmp, value);
    return *reinterpret_cast<bfloat16_t*>(&tmp[idx]);
  }
};

// BF16 <-> float32 conversions (bit manipulation, no hardware instruction)
inline Simd<float, 8>::Simd(Simd<bfloat16_t, 8> v) {
  __m256i i32 = _mm256_cvtepu16_epi32(v.value);
  value = _mm256_castsi256_ps(_mm256_slli_epi32(i32, 16));
}

inline Simd<bfloat16_t, 8>::Simd(Simd<float, 8> v) {
  // Round to nearest even
  __m256i bits = _mm256_castps_si256(v.value);
  __m256i rounding_bias = _mm256_add_epi32(
      _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)),
      _mm256_set1_epi32(0x7FFF));
  bits = _mm256_add_epi32(bits, rounding_bias);
  bits = _mm256_srli_epi32(bits, 16);
  __m128i lo = _mm256_castsi256_si128(bits);
  __m128i hi = _mm256_extracti128_si256(bits, 1);
  value = _mm_packus_epi32(lo, hi);
}

// Load/store bfloat16
template <>
inline Simd<bfloat16_t, 8> load<bfloat16_t, 8>(const bfloat16_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <>
inline void store<bfloat16_t, 8>(bfloat16_t* ptr, Simd<bfloat16_t, 8> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// All bfloat16 operations via shared macros
DEFINE_X86_HALF_SCALAR_MUL(bfloat16_t)
DEFINE_X86_HALF_BINARY_ARITHMETIC(bfloat16_t)
DEFINE_X86_HALF_COMPARISONS_VIA_FLOAT(bfloat16_t)
DEFINE_X86_HALF_SELECT_FMA_ISNAN(bfloat16_t)
DEFINE_X86_HALF_TRANSCENDENTALS(bfloat16_t)
DEFINE_X86_HALF_LOGICAL_OPS(bfloat16_t)

// ============================================================================
// int32x8 and uint32x8 (8 int32s in 256 bits)
// ============================================================================

template <>
struct Simd<int32_t, 8> {
  static constexpr int size = 8;
  __m256i value;

  Simd() : value(_mm256_setzero_si256()) {}
  Simd(__m256i v) : value(v) {}
  Simd(int32_t v) : value(_mm256_set1_epi32(v)) {}
  Simd(Simd<bool, 8> v);
  Simd(Simd<int32_t, 4> lo, Simd<int32_t, 4> hi);
  Simd(Simd<uint8_t, 8> v);
  Simd(Simd<uint16_t, 8> v);
  Simd(Simd<uint32_t, 8> v);

  int32_t operator[](int idx) const {
    alignas(32) int32_t tmp[8];
    _mm256_store_si256((__m256i*)tmp, value);
    return tmp[idx];
  }
};

template <>
struct Simd<uint32_t, 8> {
  static constexpr int size = 8;
  __m256i value;

  Simd() : value(_mm256_setzero_si256()) {}
  Simd(__m256i v) : value(v) {}
  Simd(uint32_t v) : value(_mm256_set1_epi32(v)) {}
  Simd(Simd<int32_t, 8> v);
  Simd(Simd<bool, 8> v);
  Simd(Simd<uint32_t, 4> lo, Simd<uint32_t, 4> hi);
  Simd(Simd<uint8_t, 8> v);
  Simd(Simd<uint16_t, 8> v);

  uint32_t operator[](int idx) const {
    alignas(32) uint32_t tmp[8];
    _mm256_store_si256((__m256i*)tmp, value);
    return tmp[idx];
  }
};

// uint16_t with size 8 (128-bit SSE register)
template <>
struct Simd<uint16_t, 8> {
  static constexpr int size = 8;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(uint16_t v) : value(_mm_set1_epi16(v)) {}
  Simd(Simd<uint8_t, 8> v);

  uint16_t operator[](int idx) const {
    alignas(16) uint16_t tmp[8];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

template <>
inline Simd<uint16_t, 8> load<uint16_t, 8>(const uint16_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <>
inline void store<uint16_t, 8>(uint16_t* ptr, Simd<uint16_t, 8> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

inline Simd<uint16_t, 8> operator<<(Simd<uint16_t, 8> a, int bits) {
  return _mm_slli_epi16(a.value, bits);
}

inline Simd<uint16_t, 8> operator&(Simd<uint16_t, 8> a, Simd<uint16_t, 8> b) {
  return _mm_and_si128(a.value, b.value);
}

// int32_t and uint32_t with size 4 (128-bit SSE registers)
template <>
struct Simd<int32_t, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(int32_t v) : value(_mm_set1_epi32(v)) {}
  Simd(uint32_t v);

  int32_t operator[](int idx) const {
    alignas(16) int32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

template <>
struct Simd<uint32_t, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(uint32_t v) : value(_mm_set1_epi32(v)) {}
  Simd(int32_t v);

  uint32_t operator[](int idx) const {
    alignas(16) uint32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

// Load/Store int32/uint32 size 4
template <>
inline Simd<int32_t, 4> load<int32_t, 4>(const int32_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <>
inline void store<int32_t, 4>(int32_t* ptr, Simd<int32_t, 4> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

template <>
inline Simd<uint32_t, 4> load<uint32_t, 4>(const uint32_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <>
inline void store<uint32_t, 4>(uint32_t* ptr, Simd<uint32_t, 4> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// Load/Store int32/uint32 size 8
template <>
inline Simd<int32_t, 8> load<int32_t, 8>(const int32_t* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

template <>
inline void store<int32_t, 8>(int32_t* ptr, Simd<int32_t, 8> v) {
  _mm256_storeu_si256((__m256i*)ptr, v.value);
}

template <>
inline Simd<uint32_t, 8> load<uint32_t, 8>(const uint32_t* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

template <>
inline void store<uint32_t, 8>(uint32_t* ptr, Simd<uint32_t, 8> v) {
  _mm256_storeu_si256((__m256i*)ptr, v.value);
}

// Arithmetic int32
inline Simd<int32_t, 8> operator+(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_add_epi32(a.value, b.value);
}
inline Simd<int32_t, 8> operator-(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_sub_epi32(a.value, b.value);
}
inline Simd<int32_t, 8> operator*(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_mullo_epi32(a.value, b.value);
}
inline Simd<int32_t, 8> operator-(Simd<int32_t, 8> a) {
  return _mm256_sub_epi32(_mm256_setzero_si256(), a.value);
}

// Arithmetic uint32
inline Simd<uint32_t, 8> operator+(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_add_epi32(a.value, b.value);
}
inline Simd<uint32_t, 8> operator-(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_sub_epi32(a.value, b.value);
}
inline Simd<uint32_t, 8> operator*(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_mullo_epi32(a.value, b.value);
}
inline Simd<uint32_t, 8> operator-(Simd<uint32_t, 8> a) {
  return _mm256_sub_epi32(_mm256_setzero_si256(), a.value);
}

// Shifts (scalar shift amount)
inline Simd<int32_t, 8> operator<<(Simd<int32_t, 8> a, int bits) {
  return _mm256_slli_epi32(a.value, bits);
}
inline Simd<int32_t, 8> operator>>(Simd<int32_t, 8> a, int bits) {
  return _mm256_srai_epi32(a.value, bits);
}
inline Simd<uint32_t, 8> operator<<(Simd<uint32_t, 8> a, int bits) {
  return _mm256_slli_epi32(a.value, bits);
}
inline Simd<uint32_t, 8> operator>>(Simd<uint32_t, 8> a, int bits) {
  return _mm256_srli_epi32(a.value, bits);
}

// Shifts (vector shift amount - AVX2 per-element variable shifts)
inline Simd<int32_t, 8> operator<<(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_sllv_epi32(a.value, b.value);
}
inline Simd<int32_t, 8> operator>>(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_srav_epi32(a.value, b.value);
}
inline Simd<uint32_t, 8> operator<<(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_sllv_epi32(a.value, b.value);
}
inline Simd<uint32_t, 8> operator>>(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_srlv_epi32(a.value, b.value);
}

// Bitwise int32
inline Simd<int32_t, 8> operator&(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_and_si256(a.value, b.value);
}
inline Simd<int32_t, 8> operator|(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_or_si256(a.value, b.value);
}
inline Simd<int32_t, 8> operator^(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_xor_si256(a.value, b.value);
}
inline Simd<int32_t, 8> operator~(Simd<int32_t, 8> a) {
  return _mm256_xor_si256(a.value, _mm256_set1_epi32(-1));
}

// Bitwise uint32
inline Simd<uint32_t, 8> operator&(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_and_si256(a.value, b.value);
}
inline Simd<uint32_t, 8> operator|(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_or_si256(a.value, b.value);
}
inline Simd<uint32_t, 8> operator^(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_xor_si256(a.value, b.value);
}
inline Simd<uint32_t, 8> operator~(Simd<uint32_t, 8> a) {
  return _mm256_xor_si256(a.value, _mm256_set1_epi32(-1));
}

// Logical not for integers
inline Simd<bool, 8> operator!(Simd<int32_t, 8> a) {
  return a == Simd<int32_t, 8>(0);
}

inline Simd<bool, 8> operator!(Simd<uint32_t, 8> a) {
  return a == Simd<uint32_t, 8>(0);
}

// Comparisons int32
inline Simd<bool, 8> operator<(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return Simd<bool, 8>(_mm256_cmpgt_epi32(b.value, a.value));
}
inline Simd<bool, 8> operator>(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return Simd<bool, 8>(_mm256_cmpgt_epi32(a.value, b.value));
}
inline Simd<bool, 8> operator==(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return Simd<bool, 8>(_mm256_cmpeq_epi32(a.value, b.value));
}
inline Simd<bool, 8> operator!=(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return !(a == b);
}
inline Simd<bool, 8> operator<=(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return !(a > b);
}
inline Simd<bool, 8> operator>=(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return !(a < b);
}

// Comparisons uint32
inline Simd<bool, 8> operator==(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return Simd<bool, 8>(_mm256_cmpeq_epi32(a.value, b.value));
}
inline Simd<bool, 8> operator!=(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return !(a == b);
}
inline Simd<bool, 8> operator<(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  auto minn = _mm256_min_epu32(a.value, b.value);
  return Simd<bool, 8>(_mm256_andnot_si256(
      _mm256_cmpeq_epi32(a.value, b.value), _mm256_cmpeq_epi32(minn, a.value)));
}
inline Simd<bool, 8> operator>(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return b < a;
}
inline Simd<bool, 8> operator<=(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return !(a > b);
}
inline Simd<bool, 8> operator>=(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return !(a < b);
}

// Min/max
inline Simd<int32_t, 8> maximum(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_max_epi32(a.value, b.value);
}
inline Simd<int32_t, 8> minimum(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return _mm256_min_epi32(a.value, b.value);
}
inline Simd<uint32_t, 8> maximum(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_max_epu32(a.value, b.value);
}
inline Simd<uint32_t, 8> minimum(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return _mm256_min_epu32(a.value, b.value);
}

inline Simd<int32_t, 8> abs(Simd<int32_t, 8> a) {
  return _mm256_abs_epi32(a.value);
}

// Select
inline Simd<int32_t, 8>
select(Simd<bool, 8> mask, Simd<int32_t, 8> x, Simd<int32_t, 8> y) {
  return _mm256_blendv_epi8(y.value, x.value, mask.value);
}
inline Simd<uint32_t, 8>
select(Simd<bool, 8> mask, Simd<uint32_t, 8> x, Simd<uint32_t, 8> y) {
  return _mm256_blendv_epi8(y.value, x.value, mask.value);
}

// clz (count leading zeros)
inline Simd<uint32_t, 8> clz(Simd<uint32_t, 8> x) {
  alignas(32) uint32_t tmp[8], res[8];
  _mm256_store_si256((__m256i*)tmp, x.value);
  for (int i = 0; i < 8; i++) {
#ifdef _MSC_VER
    unsigned long idx;
    res[i] = _BitScanReverse(&idx, tmp[i]) ? (31 - idx) : 32;
#else
    res[i] = tmp[i] == 0 ? 32 : __builtin_clz(tmp[i]);
#endif
  }
  return _mm256_load_si256((const __m256i*)res);
}

inline Simd<int32_t, 8> clz(Simd<int32_t, 8> x) {
  return Simd<int32_t, 8>(clz(Simd<uint32_t, 8>(x.value)).value);
}

// Logical operators for integers
inline Simd<int32_t, 8> operator&&(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  auto mask_a = (a != Simd<int32_t, 8>(0));
  auto mask_b = (b != Simd<int32_t, 8>(0));
  return Simd<int32_t, 8>(_mm256_and_si256(mask_a.value, mask_b.value));
}

inline Simd<int32_t, 8> operator||(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  auto mask_a = (a != Simd<int32_t, 8>(0));
  auto mask_b = (b != Simd<int32_t, 8>(0));
  return Simd<int32_t, 8>(_mm256_or_si256(mask_a.value, mask_b.value));
}

inline Simd<uint32_t, 8> operator&&(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  auto mask_a = (a != Simd<uint32_t, 8>(0));
  auto mask_b = (b != Simd<uint32_t, 8>(0));
  return Simd<uint32_t, 8>(_mm256_and_si256(mask_a.value, mask_b.value));
}

inline Simd<uint32_t, 8> operator||(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  auto mask_a = (a != Simd<uint32_t, 8>(0));
  auto mask_b = (b != Simd<uint32_t, 8>(0));
  return Simd<uint32_t, 8>(_mm256_or_si256(mask_a.value, mask_b.value));
}

// pow for integers (scalar fallback)
inline Simd<int32_t, 8> pow(Simd<int32_t, 8> base, Simd<int32_t, 8> exp) {
  alignas(32) int32_t tmp_base[8], tmp_exp[8], tmp_r[8];
  _mm256_store_si256((__m256i*)tmp_base, base.value);
  _mm256_store_si256((__m256i*)tmp_exp, exp.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = static_cast<int32_t>(std::pow(tmp_base[i], tmp_exp[i]));
  }
  return _mm256_load_si256((const __m256i*)tmp_r);
}

inline Simd<uint32_t, 8> pow(Simd<uint32_t, 8> base, Simd<uint32_t, 8> exp) {
  alignas(32) uint32_t tmp_base[8], tmp_exp[8], tmp_r[8];
  _mm256_store_si256((__m256i*)tmp_base, base.value);
  _mm256_store_si256((__m256i*)tmp_exp, exp.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = static_cast<uint32_t>(std::pow(tmp_base[i], tmp_exp[i]));
  }
  return _mm256_load_si256((const __m256i*)tmp_r);
}

// Division for integers (scalar fallback)
inline Simd<int32_t, 8> operator/(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  alignas(32) int32_t tmp_a[8], tmp_b[8], tmp_r[8];
  _mm256_store_si256((__m256i*)tmp_a, a.value);
  _mm256_store_si256((__m256i*)tmp_b, b.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return _mm256_load_si256((const __m256i*)tmp_r);
}

inline Simd<uint32_t, 8> operator/(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  alignas(32) uint32_t tmp_a[8], tmp_b[8], tmp_r[8];
  _mm256_store_si256((__m256i*)tmp_a, a.value);
  _mm256_store_si256((__m256i*)tmp_b, b.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return _mm256_load_si256((const __m256i*)tmp_r);
}

// Remainder for integers (Python/numpy floored semantics)
inline Simd<int32_t, 8> remainder(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  auto r = a - b * (a / b);
  // Sign correction: if r != 0 and sign(r) != sign(b), add b
  __m256i zero = _mm256_setzero_si256();
  __m256i r_nonzero = _mm256_xor_si256(
      _mm256_cmpeq_epi32(r.value, zero), _mm256_set1_epi32(-1));
  __m256i r_neg = _mm256_cmpgt_epi32(zero, r.value);
  __m256i b_neg = _mm256_cmpgt_epi32(zero, b.value);
  __m256i sign_diff = _mm256_xor_si256(r_neg, b_neg);
  __m256i needs_fix = _mm256_and_si256(r_nonzero, sign_diff);
  return Simd<int32_t, 8>{
      _mm256_add_epi32(r.value, _mm256_and_si256(needs_fix, b.value))};
}

inline Simd<uint32_t, 8> remainder(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return a - b * (a / b);
}

// Reductions int32
inline int32_t sum(Simd<int32_t, 8> v) {
  __m128i hi = _mm256_extracti128_si256(v.value, 1);
  __m128i lo = _mm256_castsi256_si128(v.value);
  __m128i sum128 = _mm_add_epi32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sums = _mm_add_epi32(sum128, shuf);
  shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));
  sums = _mm_add_epi32(sums, shuf);
  return _mm_cvtsi128_si32(sums);
}

inline uint32_t sum(Simd<uint32_t, 8> v) {
  return (uint32_t)sum(Simd<int32_t, 8>(v.value));
}

inline int32_t max(Simd<int32_t, 8> v) {
  __m128i hi = _mm256_extracti128_si256(v.value, 1);
  __m128i lo = _mm256_castsi256_si128(v.value);
  __m128i max128 = _mm_max_epi32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(max128, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i maxs = _mm_max_epi32(max128, shuf);
  shuf = _mm_shuffle_epi32(maxs, _MM_SHUFFLE(1, 0, 3, 2));
  maxs = _mm_max_epi32(maxs, shuf);
  return _mm_cvtsi128_si32(maxs);
}

inline uint32_t max(Simd<uint32_t, 8> v) {
  __m128i hi = _mm256_extracti128_si256(v.value, 1);
  __m128i lo = _mm256_castsi256_si128(v.value);
  __m128i max128 = _mm_max_epu32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(max128, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i maxs = _mm_max_epu32(max128, shuf);
  shuf = _mm_shuffle_epi32(maxs, _MM_SHUFFLE(1, 0, 3, 2));
  maxs = _mm_max_epu32(maxs, shuf);
  return _mm_cvtsi128_si32(maxs);
}

inline int32_t min(Simd<int32_t, 8> v) {
  __m128i hi = _mm256_extracti128_si256(v.value, 1);
  __m128i lo = _mm256_castsi256_si128(v.value);
  __m128i min128 = _mm_min_epi32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(min128, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i mins = _mm_min_epi32(min128, shuf);
  shuf = _mm_shuffle_epi32(mins, _MM_SHUFFLE(1, 0, 3, 2));
  mins = _mm_min_epi32(mins, shuf);
  return _mm_cvtsi128_si32(mins);
}

inline uint32_t min(Simd<uint32_t, 8> v) {
  __m128i hi = _mm256_extracti128_si256(v.value, 1);
  __m128i lo = _mm256_castsi256_si128(v.value);
  __m128i min128 = _mm_min_epu32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(min128, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i mins = _mm_min_epu32(min128, shuf);
  shuf = _mm_shuffle_epi32(mins, _MM_SHUFFLE(1, 0, 3, 2));
  mins = _mm_min_epu32(mins, shuf);
  return _mm_cvtsi128_si32(mins);
}

inline int32_t prod(Simd<int32_t, 8> v) {
  __m128i hi = _mm256_extracti128_si256(v.value, 1);
  __m128i lo = _mm256_castsi256_si128(v.value);
  __m128i prod128 = _mm_mullo_epi32(lo, hi);
  __m128i shuf = _mm_shuffle_epi32(prod128, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i prods = _mm_mullo_epi32(prod128, shuf);
  shuf = _mm_shuffle_epi32(prods, _MM_SHUFFLE(1, 0, 3, 2));
  prods = _mm_mullo_epi32(prods, shuf);
  return _mm_cvtsi128_si32(prods);
}

inline uint32_t prod(Simd<uint32_t, 8> v) {
  return (uint32_t)prod(Simd<int32_t, 8>(v.value));
}

// ============================================================================
// uint8x8 (8 uint8s packed in lower 64 bits of 128-bit register)
// ============================================================================

template <>
struct Simd<uint8_t, 8> {
  static constexpr int size = 8;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(uint8_t v) : value(_mm_set1_epi8(v)) {}
  Simd(Simd<uint32_t, 8> v);

  uint8_t operator[](int idx) const {
    alignas(16) uint8_t tmp[16];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

template <>
inline Simd<uint8_t, 8> load<uint8_t, 8>(const uint8_t* ptr) {
  return _mm_loadl_epi64((const __m128i*)ptr);
}

template <>
inline void store<uint8_t, 8>(uint8_t* ptr, Simd<uint8_t, 8> v) {
  _mm_storel_epi64((__m128i*)ptr, v.value);
}

inline Simd<uint8_t, 8> operator+(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_add_epi8(a.value, b.value);
}

inline Simd<uint8_t, 8> operator-(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_sub_epi8(a.value, b.value);
}

inline Simd<uint8_t, 8> operator*(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  alignas(16) uint8_t a_arr[16], b_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  _mm_store_si128((__m128i*)b_arr, b.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] * b_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> operator-(Simd<uint8_t, 8> a) {
  return _mm_sub_epi8(_mm_setzero_si128(), a.value);
}

inline Simd<uint8_t, 8> operator!(Simd<uint8_t, 8> a) {
  return _mm_cmpeq_epi8(a.value, _mm_setzero_si128());
}

inline Simd<uint8_t, 8> operator&(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_and_si128(a.value, b.value);
}

inline Simd<uint8_t, 8> operator|(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_or_si128(a.value, b.value);
}

inline Simd<uint8_t, 8> operator^(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_xor_si128(a.value, b.value);
}

inline Simd<uint8_t, 8> operator~(Simd<uint8_t, 8> a) {
  return _mm_xor_si128(a.value, _mm_set1_epi8(-1));
}

// Comparisons uint8
inline Simd<bool, 8> operator==(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto mask8 = _mm_cmpeq_epi8(a.value, b.value);
  return Simd<bool, 8>(_mm256_cvtepi8_epi32(mask8));
}

inline Simd<bool, 8> operator!=(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return !(a == b);
}

inline Simd<bool, 8> operator<(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto minn = _mm_min_epu8(a.value, b.value);
  auto eq = _mm_cmpeq_epi8(a.value, b.value);
  auto min_a = _mm_cmpeq_epi8(minn, a.value);
  auto mask8 = _mm_andnot_si128(eq, min_a);
  return Simd<bool, 8>(_mm256_cvtepi8_epi32(mask8));
}

inline Simd<bool, 8> operator>(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return b < a;
}

inline Simd<bool, 8> operator<=(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return !(a > b);
}

inline Simd<bool, 8> operator>=(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return !(a < b);
}

// Shift operators uint8
inline Simd<uint8_t, 8> operator<<(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  alignas(16) uint8_t a_arr[16], b_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  _mm_store_si128((__m128i*)b_arr, b.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] << b_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> operator>>(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  alignas(16) uint8_t a_arr[16], b_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  _mm_store_si128((__m128i*)b_arr, b.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] >> b_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> operator<<(Simd<uint8_t, 8> a, int bits) {
  if (bits >= 8)
    return Simd<uint8_t, 8>(_mm_setzero_si128());
  alignas(16) uint8_t a_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] << bits;
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> operator>>(Simd<uint8_t, 8> a, int bits) {
  if (bits >= 8)
    return Simd<uint8_t, 8>(_mm_setzero_si128());
  alignas(16) uint8_t a_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] >> bits;
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

// Math functions uint8
inline Simd<uint8_t, 8> minimum(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_min_epu8(a.value, b.value);
}

inline Simd<uint8_t, 8> maximum(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_max_epu8(a.value, b.value);
}

inline Simd<uint8_t, 8> pow(Simd<uint8_t, 8> base, Simd<uint8_t, 8> exp) {
  alignas(16) uint8_t base_arr[16], exp_arr[16], result[16];
  _mm_store_si128((__m128i*)base_arr, base.value);
  _mm_store_si128((__m128i*)exp_arr, exp.value);
  for (int i = 0; i < 8; i++) {
    result[i] = static_cast<uint8_t>(std::pow(base_arr[i], exp_arr[i]));
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

// select uint8
inline Simd<uint8_t, 8>
select(Simd<bool, 8> mask, Simd<uint8_t, 8> x, Simd<uint8_t, 8> y) {
  alignas(32) int32_t mask32[8];
  _mm256_store_si256((__m256i*)mask32, mask.value);
  alignas(16) uint8_t x_arr[16], y_arr[16], result[16];
  _mm_store_si128((__m128i*)x_arr, x.value);
  _mm_store_si128((__m128i*)y_arr, y.value);
  for (int i = 0; i < 8; i++) {
    result[i] = mask32[i] ? x_arr[i] : y_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

// Division and remainder uint8
inline Simd<uint8_t, 8> operator/(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  alignas(16) uint8_t a_arr[16], b_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  _mm_store_si128((__m128i*)b_arr, b.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] / b_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> remainder(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return a - b * (a / b);
}

// Reductions uint8
inline uint8_t sum(Simd<uint8_t, 8> v) {
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v.value);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
}

inline uint8_t min(Simd<uint8_t, 8> v) {
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v.value);
  uint8_t result = tmp[0];
  for (int i = 1; i < 8; i++) {
    if (tmp[i] < result)
      result = tmp[i];
  }
  return result;
}

inline uint8_t max(Simd<uint8_t, 8> v) {
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v.value);
  uint8_t result = tmp[0];
  for (int i = 1; i < 8; i++) {
    if (tmp[i] > result)
      result = tmp[i];
  }
  return result;
}

inline uint8_t prod(Simd<uint8_t, 8> v) {
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v.value);
  return tmp[0] * tmp[1] * tmp[2] * tmp[3] * tmp[4] * tmp[5] * tmp[6] * tmp[7];
}

// Logical operators uint8
inline Simd<uint8_t, 8> operator&&(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto mask_a = _mm_cmpeq_epi8(a.value, _mm_setzero_si128());
  auto mask_b = _mm_cmpeq_epi8(b.value, _mm_setzero_si128());
  auto not_a = _mm_xor_si128(mask_a, _mm_set1_epi8(-1));
  auto not_b = _mm_xor_si128(mask_b, _mm_set1_epi8(-1));
  return Simd<uint8_t, 8>(_mm_and_si128(not_a, not_b));
}

inline Simd<uint8_t, 8> operator||(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto mask_a = _mm_cmpeq_epi8(a.value, _mm_setzero_si128());
  auto mask_b = _mm_cmpeq_epi8(b.value, _mm_setzero_si128());
  auto not_a = _mm_xor_si128(mask_a, _mm_set1_epi8(-1));
  auto not_b = _mm_xor_si128(mask_b, _mm_set1_epi8(-1));
  return Simd<uint8_t, 8>(_mm_or_si128(not_a, not_b));
}

// ============================================================================
// Conversion constructor definitions (after all structs are complete)
// ============================================================================
inline Simd<float, 8>::Simd(Simd<bool, 8> v)
    : value(_mm256_castsi256_ps(v.value)) {}
inline Simd<float, 8>::Simd(Simd<int32_t, 8> v)
    : value(_mm256_cvtepi32_ps(v.value)) {}
inline Simd<float, 8>::Simd(Simd<uint32_t, 8> v)
    : value(_mm256_cvtepi32_ps(v.value)) {}
inline Simd<double, 4>::Simd(Simd<bool, 4> v)
    : value(_mm256_castsi256_pd(v.value)) {}

inline Simd<int32_t, 8>::Simd(Simd<uint32_t, 8> v) : value(v.value) {}
inline Simd<uint32_t, 8>::Simd(Simd<int32_t, 8> v) : value(v.value) {}
inline Simd<int32_t, 8>::Simd(Simd<bool, 8> v) : value(v.value) {}
inline Simd<uint32_t, 8>::Simd(Simd<bool, 8> v) : value(v.value) {}

inline Simd<int32_t, 4>::Simd(uint32_t v) : value(_mm_set1_epi32(v)) {}
inline Simd<uint32_t, 4>::Simd(int32_t v) : value(_mm_set1_epi32(v)) {}

inline Simd<int32_t, 8>::Simd(Simd<int32_t, 4> lo, Simd<int32_t, 4> hi)
    : value(_mm256_inserti128_si256(
          _mm256_castsi128_si256(lo.value),
          hi.value,
          1)) {}

inline Simd<uint32_t, 8>::Simd(Simd<uint32_t, 4> lo, Simd<uint32_t, 4> hi)
    : value(_mm256_inserti128_si256(
          _mm256_castsi128_si256(lo.value),
          hi.value,
          1)) {}

inline Simd<int32_t, 8>::Simd(Simd<uint8_t, 8> v)
    : value(_mm256_cvtepu8_epi32(v.value)) {}

inline Simd<uint32_t, 8>::Simd(Simd<uint8_t, 8> v)
    : value(_mm256_cvtepu8_epi32(v.value)) {}

inline Simd<uint8_t, 8>::Simd(Simd<uint32_t, 8> v) {
  alignas(32) uint32_t tmp32[8];
  _mm256_store_si256((__m256i*)tmp32, v.value);
  alignas(16) uint8_t tmp8[16] = {0};
  for (int i = 0; i < 8; i++) {
    tmp8[i] = static_cast<uint8_t>(tmp32[i]);
  }
  value = _mm_loadl_epi64((const __m128i*)tmp8);
}

inline Simd<int32_t, 8>::Simd(Simd<uint16_t, 8> v)
    : value(_mm256_cvtepu16_epi32(v.value)) {}

inline Simd<uint32_t, 8>::Simd(Simd<uint16_t, 8> v)
    : value(_mm256_cvtepu16_epi32(v.value)) {}

inline Simd<uint16_t, 8>::Simd(Simd<uint8_t, 8> v)
    : value(_mm_cvtepu8_epi16(v.value)) {}

inline Simd<bool, 8>::Simd(Simd<uint8_t, 8> v)
    : value(_mm256_cvtepi8_epi32(v.value)) {}

// int64/uint64 -> float (no AVX2 intrinsic, extract and convert element-wise)
inline Simd<float, 8>::Simd(Simd<uint64_t, 8> v) {
  alignas(32) uint64_t lo_tmp[4], hi_tmp[4];
  _mm256_store_si256((__m256i*)lo_tmp, v.lo);
  _mm256_store_si256((__m256i*)hi_tmp, v.hi);
  value = _mm256_setr_ps(
      static_cast<float>(lo_tmp[0]),
      static_cast<float>(lo_tmp[1]),
      static_cast<float>(lo_tmp[2]),
      static_cast<float>(lo_tmp[3]),
      static_cast<float>(hi_tmp[0]),
      static_cast<float>(hi_tmp[1]),
      static_cast<float>(hi_tmp[2]),
      static_cast<float>(hi_tmp[3]));
}

inline Simd<float, 8>::Simd(Simd<int64_t, 8> v) {
  alignas(32) int64_t lo_tmp[4], hi_tmp[4];
  _mm256_store_si256((__m256i*)lo_tmp, v.lo);
  _mm256_store_si256((__m256i*)hi_tmp, v.hi);
  value = _mm256_setr_ps(
      static_cast<float>(lo_tmp[0]),
      static_cast<float>(lo_tmp[1]),
      static_cast<float>(lo_tmp[2]),
      static_cast<float>(lo_tmp[3]),
      static_cast<float>(hi_tmp[0]),
      static_cast<float>(hi_tmp[1]),
      static_cast<float>(hi_tmp[2]),
      static_cast<float>(hi_tmp[3]));
}

// ============================================================================
// Bool reductions
// ============================================================================
inline bool all(Simd<bool, 8> x) {
  return _mm256_movemask_ps(_mm256_castsi256_ps(x.value)) == 0xFF;
}

inline bool any(Simd<bool, 8> x) {
  return _mm256_movemask_ps(_mm256_castsi256_ps(x.value)) != 0;
}

inline bool all(Simd<bool, 4> x) {
  return _mm256_movemask_pd(_mm256_castsi256_pd(x.value)) == 0xF;
}

inline bool any(Simd<bool, 4> x) {
  return _mm256_movemask_pd(_mm256_castsi256_pd(x.value)) != 0;
}

} // namespace mlx::core::simd

#endif // __AVX2__
