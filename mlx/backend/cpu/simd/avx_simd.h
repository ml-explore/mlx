#pragma once

#ifdef __AVX2__

// AVX2 provides 256-bit SIMD for both floating-point AND integers
// This combines what was originally Phase 2 (AVX) and Phase 3 (AVX2)
// Rationale: AVX1 without AVX2 is too limited (no 256-bit integer ops, not even for bool masks)
#include "mlx/backend/cpu/simd/base_simd.h"
#include "mlx/types/half_types.h"

#include <immintrin.h> // AVX
#include <cmath>
#include <stdint.h>

namespace mlx::core::simd {

// AVX2: 256-bit SIMD for floats, doubles, AND integers
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

// ============================================================================
// Bool specializations (forward declare early for comparison operators)
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

// Bool operators for Simd<bool, 8>
inline Simd<bool, 8> operator!(Simd<bool, 8> a) {
  return Simd<bool, 8>(_mm256_xor_si256(a.value, _mm256_set1_epi32(-1)));
}

inline Simd<bool, 8> operator||(Simd<bool, 8> a, Simd<bool, 8> b) {
  return Simd<bool, 8>(_mm256_or_si256(a.value, b.value));
}

inline Simd<bool, 8> operator&&(Simd<bool, 8> a, Simd<bool, 8> b) {
  return Simd<bool, 8>(_mm256_and_si256(a.value, b.value));
}

inline Simd<bool, 8> operator&(Simd<bool, 8> a, Simd<bool, 8> b) {
  return Simd<bool, 8>(_mm256_and_si256(a.value, b.value));
}

inline Simd<bool, 8> operator|(Simd<bool, 8> a, Simd<bool, 8> b) {
  return Simd<bool, 8>(_mm256_or_si256(a.value, b.value));
}

inline Simd<bool, 8> operator^(Simd<bool, 8> a, Simd<bool, 8> b) {
  return Simd<bool, 8>(_mm256_xor_si256(a.value, b.value));
}

inline Simd<bool, 8> operator==(Simd<bool, 8> a, Simd<bool, 8> b) {
  return !Simd<bool, 8>(_mm256_xor_si256(a.value, b.value));
}

inline Simd<bool, 8> operator!=(Simd<bool, 8> a, Simd<bool, 8> b) {
  return !(a == b);
}

// Bool operators for Simd<bool, 4> (AVX double version with __m256i)
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
    ptr[7] ? -1 : 0, ptr[6] ? -1 : 0, ptr[5] ? -1 : 0, ptr[4] ? -1 : 0,
    ptr[3] ? -1 : 0, ptr[2] ? -1 : 0, ptr[1] ? -1 : 0, ptr[0] ? -1 : 0
  ));
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
    ptr[0] ? -1LL : 0
  ));
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
inline Simd<float, 8> operator+(Simd<float, 8> a, Simd<float, 8> b) {
  return _mm256_add_ps(a.value, b.value);
}
inline Simd<float, 8> operator-(Simd<float, 8> a, Simd<float, 8> b) {
  return _mm256_sub_ps(a.value, b.value);
}
inline Simd<float, 8> operator*(Simd<float, 8> a, Simd<float, 8> b) {
  return _mm256_mul_ps(a.value, b.value);
}
inline Simd<float, 8> operator/(Simd<float, 8> a, Simd<float, 8> b) {
  return _mm256_div_ps(a.value, b.value);
}
inline Simd<float, 8> operator-(Simd<float, 8> a) {
  return _mm256_sub_ps(_mm256_setzero_ps(), a.value);
}

// Comparisons float32x8
inline Simd<bool, 8> operator<(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, b.value, _CMP_LT_OQ)));
}
inline Simd<bool, 8> operator>(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, b.value, _CMP_GT_OQ)));
}
inline Simd<bool, 8> operator<=(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, b.value, _CMP_LE_OQ)));
}
inline Simd<bool, 8> operator>=(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, b.value, _CMP_GE_OQ)));
}
inline Simd<bool, 8> operator==(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, b.value, _CMP_EQ_OQ)));
}
inline Simd<bool, 8> operator!=(Simd<float, 8> a, Simd<float, 8> b) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, b.value, _CMP_NEQ_OQ)));
}

inline Simd<bool, 8> operator!(Simd<float, 8> a) {
  return a == Simd<float, 8>(0.0f);
}

inline Simd<bool, 8> isnan(Simd<float, 8> a) {
  return Simd<bool, 8>(_mm256_castps_si256(_mm256_cmp_ps(a.value, a.value, _CMP_UNORD_Q)));
}

// Select (blend) float32x8
inline Simd<float, 8> select(Simd<bool, 8> mask, Simd<float, 8> x, Simd<float, 8> y) {
  return _mm256_blendv_ps(y.value, x.value, _mm256_castsi256_ps(mask.value));
}

// Math functions float32x8
inline Simd<float, 8> abs(Simd<float, 8> a) {
  __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  return _mm256_and_ps(a.value, mask);
}

inline Simd<float, 8> sqrt(Simd<float, 8> a) {
  return _mm256_sqrt_ps(a.value);
}

inline Simd<float, 8> rsqrt(Simd<float, 8> a) {
  return _mm256_rsqrt_ps(a.value);
}

inline Simd<float, 8> recip(Simd<float, 8> a) {
  return _mm256_rcp_ps(a.value);
}

inline Simd<float, 8> floor(Simd<float, 8> a) {
  return _mm256_floor_ps(a.value);
}

inline Simd<float, 8> ceil(Simd<float, 8> a) {
  return _mm256_ceil_ps(a.value);
}

inline Simd<float, 8> rint(Simd<float, 8> a) {
  return _mm256_round_ps(a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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

inline Simd<float, 8> clamp(Simd<float, 8> v, Simd<float, 8> min_val, Simd<float, 8> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

// Transcendental functions - scalar fallback
inline Simd<float, 8> atan2(Simd<float, 8> a, Simd<float, 8> b) {
  alignas(32) float tmp_a[8], tmp_b[8], tmp_r[8];
  _mm256_store_ps(tmp_a, a.value);
  _mm256_store_ps(tmp_b, b.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = std::atan2(tmp_a[i], tmp_b[i]);
  }
  return _mm256_load_ps(tmp_r);
}

inline Simd<float, 8> pow(Simd<float, 8> base, Simd<float, 8> exp) {
  alignas(32) float tmp_base[8], tmp_exp[8], tmp_r[8];
  _mm256_store_ps(tmp_base, base.value);
  _mm256_store_ps(tmp_exp, exp.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = std::pow(tmp_base[i], tmp_exp[i]);
  }
  return _mm256_load_ps(tmp_r);
}

inline Simd<float, 8> remainder(Simd<float, 8> a, Simd<float, 8> b) {
  alignas(32) float tmp_a[8], tmp_b[8], tmp_r[8];
  _mm256_store_ps(tmp_a, a.value);
  _mm256_store_ps(tmp_b, b.value);
  for (int i = 0; i < 8; i++) {
    tmp_r[i] = std::remainder(tmp_a[i], tmp_b[i]);
  }
  return _mm256_load_ps(tmp_r);
}

// Macro for transcendental functions
#define AVX2_TRANSCENDENTAL_FLOAT(name, func)                    \
  inline Simd<float, 8> name(Simd<float, 8> a) {                 \
    alignas(32) float tmp_a[8], tmp_r[8];                        \
    _mm256_store_ps(tmp_a, a.value);                             \
    for (int i = 0; i < 8; i++) {                                \
      tmp_r[i] = std::func(tmp_a[i]);                            \
    }                                                            \
    return _mm256_load_ps(tmp_r);                                \
  }

AVX2_TRANSCENDENTAL_FLOAT(exp, exp)
AVX2_TRANSCENDENTAL_FLOAT(expm1, expm1)
AVX2_TRANSCENDENTAL_FLOAT(log, log)
AVX2_TRANSCENDENTAL_FLOAT(log1p, log1p)
AVX2_TRANSCENDENTAL_FLOAT(log2, log2)
AVX2_TRANSCENDENTAL_FLOAT(log10, log10)
AVX2_TRANSCENDENTAL_FLOAT(sin, sin)
AVX2_TRANSCENDENTAL_FLOAT(cos, cos)
AVX2_TRANSCENDENTAL_FLOAT(tan, tan)
AVX2_TRANSCENDENTAL_FLOAT(asin, asin)
AVX2_TRANSCENDENTAL_FLOAT(acos, acos)
AVX2_TRANSCENDENTAL_FLOAT(atan, atan)
AVX2_TRANSCENDENTAL_FLOAT(sinh, sinh)
AVX2_TRANSCENDENTAL_FLOAT(cosh, cosh)
AVX2_TRANSCENDENTAL_FLOAT(tanh, tanh)
AVX2_TRANSCENDENTAL_FLOAT(asinh, asinh)
AVX2_TRANSCENDENTAL_FLOAT(acosh, acosh)
AVX2_TRANSCENDENTAL_FLOAT(atanh, atanh)

// Logical operators
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

// FMA
#if defined(__FMA__)
inline Simd<float, 8> fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
  return _mm256_fmadd_ps(a.value, b.value, c.value);
}
#else
inline Simd<float, 8> fma(Simd<float, 8> a, Simd<float, 8> b, Simd<float, 8> c) {
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
inline Simd<double, 4> operator+(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_add_pd(a.value, b.value);
}
inline Simd<double, 4> operator-(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_sub_pd(a.value, b.value);
}
inline Simd<double, 4> operator*(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_mul_pd(a.value, b.value);
}
inline Simd<double, 4> operator/(Simd<double, 4> a, Simd<double, 4> b) {
  return _mm256_div_pd(a.value, b.value);
}
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
inline Simd<bool, 4> operator<(Simd<double, 4> a, Simd<double, 4> b) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, b.value, _CMP_LT_OQ)));
}
inline Simd<bool, 4> operator>(Simd<double, 4> a, Simd<double, 4> b) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, b.value, _CMP_GT_OQ)));
}
inline Simd<bool, 4> operator<=(Simd<double, 4> a, Simd<double, 4> b) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, b.value, _CMP_LE_OQ)));
}
inline Simd<bool, 4> operator>=(Simd<double, 4> a, Simd<double, 4> b) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, b.value, _CMP_GE_OQ)));
}
inline Simd<bool, 4> operator==(Simd<double, 4> a, Simd<double, 4> b) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, b.value, _CMP_EQ_OQ)));
}
inline Simd<bool, 4> operator!=(Simd<double, 4> a, Simd<double, 4> b) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, b.value, _CMP_NEQ_OQ)));
}

// Logical NOT for double - returns bool mask where each element is true if input is zero
inline Simd<bool, 4> operator!(Simd<double, 4> a) {
  return a == Simd<double, 4>(0.0);
}

// Select (blend) double64x4 - defined early since it's used by maximum/minimum
inline Simd<double, 4> select(
    Simd<bool, 4> mask,
    Simd<double, 4> x,
    Simd<double, 4> y) {
  return _mm256_blendv_pd(y.value, x.value, _mm256_castsi256_pd(mask.value));
}

// isnan for double - use unordered comparison (specifically for NaN detection)
inline Simd<bool, 4> isnan(Simd<double, 4> a) {
  return Simd<bool, 4>(_mm256_castpd_si256(_mm256_cmp_pd(a.value, a.value, _CMP_UNORD_Q)));
}

// Math functions double64x4
inline Simd<double, 4> abs(Simd<double, 4> a) {
  __m256d mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
  return _mm256_and_pd(a.value, mask);
}

inline Simd<double, 4> sqrt(Simd<double, 4> a) {
  return _mm256_sqrt_pd(a.value);
}

inline Simd<double, 4> rsqrt(Simd<double, 4> a) {
  // No AVX rsqrt for double, use div
  return Simd<double, 4>(1.0) / sqrt(a);
}

inline Simd<double, 4> recip(Simd<double, 4> a) {
  // No AVX recip for double, use div
  return Simd<double, 4>(1.0) / a;
}

inline Simd<double, 4> floor(Simd<double, 4> a) {
  return _mm256_floor_pd(a.value);
}

inline Simd<double, 4> ceil(Simd<double, 4> a) {
  return _mm256_ceil_pd(a.value);
}

inline Simd<double, 4> rint(Simd<double, 4> a) {
  return _mm256_round_pd(a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline Simd<double, 4> maximum(Simd<double, 4> a, Simd<double, 4> b) {
  auto out = Simd<double, 4>(_mm256_max_pd(a.value, b.value));
  // NaN propagation: if either input is NaN, return that NaN
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 4> minimum(Simd<double, 4> a, Simd<double, 4> b) {
  auto out = Simd<double, 4>(_mm256_min_pd(a.value, b.value));
  // NaN propagation: if either input is NaN, return that NaN
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 4> clamp(Simd<double, 4> v, Simd<double, 4> min_val, Simd<double, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

inline Simd<double, 4> atan2(Simd<double, 4> a, Simd<double, 4> b) {
  // atan2 needs to be computed per element using scalar fallback
  alignas(32) double tmp_a[4], tmp_b[4], tmp_r[4];
  _mm256_store_pd(tmp_a, a.value);
  _mm256_store_pd(tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = std::atan2(tmp_a[i], tmp_b[i]);
  }
  return _mm256_load_pd(tmp_r);
}

inline Simd<double, 4> pow(Simd<double, 4> base, Simd<double, 4> exp) {
  // pow needs to be computed per element using scalar fallback
  alignas(32) double tmp_base[4], tmp_exp[4], tmp_r[4];
  _mm256_store_pd(tmp_base, base.value);
  _mm256_store_pd(tmp_exp, exp.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = std::pow(tmp_base[i], tmp_exp[i]);
  }
  return _mm256_load_pd(tmp_r);
}

// Transcendental functions - use scalar fallback for now
#define AVX_TRANSCENDENTAL_DOUBLE(name, func)                   \
  inline Simd<double, 4> name(Simd<double, 4> a) {              \
    alignas(32) double tmp_a[4], tmp_r[4];                      \
    _mm256_store_pd(tmp_a, a.value);                            \
    for (int i = 0; i < 4; i++) {                               \
      tmp_r[i] = std::func(tmp_a[i]);                           \
    }                                                           \
    return _mm256_load_pd(tmp_r);                               \
  }

AVX_TRANSCENDENTAL_DOUBLE(exp, exp)
AVX_TRANSCENDENTAL_DOUBLE(expm1, expm1)
AVX_TRANSCENDENTAL_DOUBLE(log, log)
AVX_TRANSCENDENTAL_DOUBLE(log1p, log1p)
AVX_TRANSCENDENTAL_DOUBLE(log2, log2)
AVX_TRANSCENDENTAL_DOUBLE(log10, log10)
AVX_TRANSCENDENTAL_DOUBLE(sin, sin)
AVX_TRANSCENDENTAL_DOUBLE(cos, cos)
AVX_TRANSCENDENTAL_DOUBLE(tan, tan)
AVX_TRANSCENDENTAL_DOUBLE(asin, asin)
AVX_TRANSCENDENTAL_DOUBLE(acos, acos)
AVX_TRANSCENDENTAL_DOUBLE(atan, atan)
AVX_TRANSCENDENTAL_DOUBLE(sinh, sinh)
AVX_TRANSCENDENTAL_DOUBLE(cosh, cosh)
AVX_TRANSCENDENTAL_DOUBLE(tanh, tanh)
AVX_TRANSCENDENTAL_DOUBLE(asinh, asinh)
AVX_TRANSCENDENTAL_DOUBLE(acosh, acosh)
AVX_TRANSCENDENTAL_DOUBLE(atanh, atanh)

inline Simd<double, 4> remainder(Simd<double, 4> a, Simd<double, 4> b) {
  alignas(32) double tmp_a[4], tmp_b[4], tmp_r[4];
  _mm256_store_pd(tmp_a, a.value);
  _mm256_store_pd(tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = std::remainder(tmp_a[i], tmp_b[i]);
  }
  return _mm256_load_pd(tmp_r);
}

// Logical operators for double64x4 (return double with bit pattern, not bool)
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

// FMA (emulated)
inline Simd<double, 4> fma(Simd<double, 4> a, Simd<double, 4> b, Simd<double, 4> c) {
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
// float16x8 (8 float16s stored as 128-bit, emulated via float32)
// Note: x86 doesn't have native FP16 until AVX-512, so we store as uint16
// and convert to float32 for operations
// ============================================================================

template <>
struct Simd<float16_t, 8> {
  static constexpr int size = 8;
  __m128i value;  // Stored as 8x16-bit values

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}

  float16_t operator[](int idx) const {
    alignas(16) uint16_t tmp[8];
    _mm_store_si128((__m128i*)tmp, value);
    return *reinterpret_cast<float16_t*>(&tmp[idx]);
  }
};

// Arithmetic via conversion to float32
inline Simd<float16_t, 8> operator*(Simd<float16_t, 8> a, float scalar) {
  // Convert to float32, multiply, convert back
  alignas(16) uint16_t a_bits[8];
  _mm_store_si128((__m128i*)a_bits, a.value);

  alignas(32) float tmp[8];
  for (int i = 0; i < 8; i++) {
    float16_t f16 = *reinterpret_cast<float16_t*>(&a_bits[i]);
    tmp[i] = static_cast<float>(f16) * scalar;
  }

  alignas(16) uint16_t result[8];
  for (int i = 0; i < 8; i++) {
    float16_t f16 = static_cast<float16_t>(tmp[i]);
    result[i] = *reinterpret_cast<uint16_t*>(&f16);
  }

  return _mm_load_si128((const __m128i*)result);
}

// Conversion to Simd<float, 8>
inline Simd<float, 8>::Simd(Simd<float16_t, 8> v) {
  alignas(16) uint16_t bits[8];
  _mm_store_si128((__m128i*)bits, v.value);

  alignas(32) float tmp[8];
  for (int i = 0; i < 8; i++) {
    float16_t f16 = *reinterpret_cast<float16_t*>(&bits[i]);
    tmp[i] = static_cast<float>(f16);
  }

  value = _mm256_load_ps(tmp);
}

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
  Simd(Simd<uint32_t, 8> v);
  Simd(Simd<bool, 8> v);
  Simd(Simd<int32_t, 4> lo, Simd<int32_t, 4> hi);
  Simd(Simd<uint8_t, 8> v);

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

  uint32_t operator[](int idx) const {
    alignas(32) uint32_t tmp[8];
    _mm256_store_si256((__m256i*)tmp, value);
    return tmp[idx];
  }
};

// uint16_t with size 8 (use 128-bit SSE register)
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

// Shift operator for uint16_t
inline Simd<uint16_t, 8> operator<<(Simd<uint16_t, 8> a, int bits) {
  return _mm_slli_epi16(a.value, bits);
}

inline Simd<uint16_t, 8> operator&(Simd<uint16_t, 8> a, Simd<uint16_t, 8> b) {
  return _mm_and_si128(a.value, b.value);
}

// int32_t and uint32_t with size 4 (use 128-bit SSE registers)
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

// Load/Store for int32_t/uint32_t size 4
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

// Load/Store for int32_t/uint32_t size 8
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

// Shifts (vector shift amount - AVX2 supports per-element variable shifts)
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

// Bitwise
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
  return Simd<bool, 8>(_mm256_andnot_si256(_mm256_cmpeq_epi32(a.value, b.value), _mm256_cmpeq_epi32(minn, a.value)));
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
inline Simd<int32_t, 8> select(Simd<bool, 8> mask, Simd<int32_t, 8> x, Simd<int32_t, 8> y) {
  return _mm256_blendv_epi8(y.value, x.value, mask.value);
}
inline Simd<uint32_t, 8> select(Simd<bool, 8> mask, Simd<uint32_t, 8> x, Simd<uint32_t, 8> y) {
  return _mm256_blendv_epi8(y.value, x.value, mask.value);
}

// clz (count leading zeros)
inline Simd<uint32_t, 8> clz(Simd<uint32_t, 8> x) {
  alignas(32) uint32_t tmp[8], res[8];
  _mm256_store_si256((__m256i*)tmp, x.value);
  for (int i = 0; i < 8; i++) {
    res[i] = tmp[i] == 0 ? 32 : __builtin_clz(tmp[i]);
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

// Remainder for integers
inline Simd<int32_t, 8> remainder(Simd<int32_t, 8> a, Simd<int32_t, 8> b) {
  return a - b * (a / b);
}

inline Simd<uint32_t, 8> remainder(Simd<uint32_t, 8> a, Simd<uint32_t, 8> b) {
  return a - b * (a / b);
}

// Reductions
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
// Note: AVX2 has 256-bit integer ops but quantized code only needs 8 bytes
// ============================================================================

template <>
struct Simd<uint8_t, 8> {
  static constexpr int size = 8;
  __m128i value;  // Only lower 64 bits used

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

// Simd<uint8_t, 8> operators
inline Simd<uint8_t, 8> operator+(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_add_epi8(a.value, b.value);
}

inline Simd<uint8_t, 8> operator-(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_sub_epi8(a.value, b.value);
}

inline Simd<uint8_t, 8> operator*(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  // SSE doesn't have 8-bit multiply, need to do it manually
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

// Comparison operators for uint8_t
// Need to convert 8-bit masks to 32-bit masks for Simd<bool, 8>
inline Simd<bool, 8> operator==(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto mask8 = _mm_cmpeq_epi8(a.value, b.value);
  // Extend 8x8-bit masks to 8x32-bit masks
  auto mask32 = _mm256_cvtepi8_epi32(mask8);
  return Simd<bool, 8>(mask32);
}

inline Simd<bool, 8> operator!=(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return !(a == b);
}

inline Simd<bool, 8> operator<(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  // Unsigned comparison: a < b if min(a,b) == a and a != b
  auto minn = _mm_min_epu8(a.value, b.value);
  auto eq = _mm_cmpeq_epi8(a.value, b.value);
  auto min_a = _mm_cmpeq_epi8(minn, a.value);
  auto mask8 = _mm_andnot_si128(eq, min_a);
  // Extend 8x8-bit masks to 8x32-bit masks
  auto mask32 = _mm256_cvtepi8_epi32(mask8);
  return Simd<bool, 8>(mask32);
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

// Shift operators for uint8_t
inline Simd<uint8_t, 8> operator<<(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  // No 8-bit shift in SSE, need scalar fallback
  alignas(16) uint8_t a_arr[16], b_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  _mm_store_si128((__m128i*)b_arr, b.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] << b_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> operator>>(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  // No 8-bit shift in SSE, need scalar fallback
  alignas(16) uint8_t a_arr[16], b_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  _mm_store_si128((__m128i*)b_arr, b.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] >> b_arr[i];
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

// Scalar versions
inline Simd<uint8_t, 8> operator<<(Simd<uint8_t, 8> a, int bits) {
  if (bits >= 8) return Simd<uint8_t, 8>(_mm_setzero_si128());
  // No 8-bit shift in SSE, need scalar fallback
  alignas(16) uint8_t a_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] << bits;
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

inline Simd<uint8_t, 8> operator>>(Simd<uint8_t, 8> a, int bits) {
  if (bits >= 8) return Simd<uint8_t, 8>(_mm_setzero_si128());
  // No 8-bit shift in SSE, need scalar fallback
  alignas(16) uint8_t a_arr[16], result[16];
  _mm_store_si128((__m128i*)a_arr, a.value);
  for (int i = 0; i < 8; i++) {
    result[i] = a_arr[i] >> bits;
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

// Math functions for uint8_t
inline Simd<uint8_t, 8> minimum(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_min_epu8(a.value, b.value);
}

inline Simd<uint8_t, 8> maximum(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  return _mm_max_epu8(a.value, b.value);
}

inline Simd<uint8_t, 8> pow(Simd<uint8_t, 8> base, Simd<uint8_t, 8> exp) {
  // Scalar fallback for pow
  alignas(16) uint8_t base_arr[16], exp_arr[16], result[16];
  _mm_store_si128((__m128i*)base_arr, base.value);
  _mm_store_si128((__m128i*)exp_arr, exp.value);
  for (int i = 0; i < 8; i++) {
    result[i] = static_cast<uint8_t>(std::pow(base_arr[i], exp_arr[i]));
  }
  return _mm_loadl_epi64((const __m128i*)result);
}

// select for uint8_t
inline Simd<uint8_t, 8> select(Simd<bool, 8> mask, Simd<uint8_t, 8> x, Simd<uint8_t, 8> y) {
  // Convert 32-bit masks back to 8-bit masks
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

// Division and remainder for uint8_t
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

// Reductions for uint8_t
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
    if (tmp[i] < result) result = tmp[i];
  }
  return result;
}

inline uint8_t max(Simd<uint8_t, 8> v) {
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v.value);
  uint8_t result = tmp[0];
  for (int i = 1; i < 8; i++) {
    if (tmp[i] > result) result = tmp[i];
  }
  return result;
}

inline uint8_t prod(Simd<uint8_t, 8> v) {
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i*)tmp, v.value);
  return tmp[0] * tmp[1] * tmp[2] * tmp[3] * tmp[4] * tmp[5] * tmp[6] * tmp[7];
}

// Logical operators for uint8_t
inline Simd<uint8_t, 8> operator&&(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto mask_a = _mm_cmpeq_epi8(a.value, _mm_setzero_si128());
  auto mask_b = _mm_cmpeq_epi8(b.value, _mm_setzero_si128());
  // Invert masks: value != 0
  auto not_a = _mm_xor_si128(mask_a, _mm_set1_epi8(-1));
  auto not_b = _mm_xor_si128(mask_b, _mm_set1_epi8(-1));
  return Simd<uint8_t, 8>(_mm_and_si128(not_a, not_b));
}

inline Simd<uint8_t, 8> operator||(Simd<uint8_t, 8> a, Simd<uint8_t, 8> b) {
  auto mask_a = _mm_cmpeq_epi8(a.value, _mm_setzero_si128());
  auto mask_b = _mm_cmpeq_epi8(b.value, _mm_setzero_si128());
  // Invert masks: value != 0
  auto not_a = _mm_xor_si128(mask_a, _mm_set1_epi8(-1));
  auto not_b = _mm_xor_si128(mask_b, _mm_set1_epi8(-1));
  return Simd<uint8_t, 8>(_mm_or_si128(not_a, not_b));
}

// ============================================================================
// Conversion constructor definitions (after all structs are complete)
// ============================================================================
inline Simd<float, 8>::Simd(Simd<bool, 8> v) : value(_mm256_castsi256_ps(v.value)) {}
inline Simd<float, 8>::Simd(Simd<int32_t, 8> v) : value(_mm256_cvtepi32_ps(v.value)) {}
inline Simd<float, 8>::Simd(Simd<uint32_t, 8> v) : value(_mm256_cvtepi32_ps(v.value)) {}
inline Simd<double, 4>::Simd(Simd<bool, 4> v) : value(_mm256_castsi256_pd(v.value)) {}

// int32_t and uint32_t conversions (size 8)
inline Simd<int32_t, 8>::Simd(Simd<uint32_t, 8> v) : value(v.value) {}
inline Simd<int32_t, 8>::Simd(Simd<bool, 8> v) : value(v.value) {}
inline Simd<uint32_t, 8>::Simd(Simd<int32_t, 8> v) : value(v.value) {}
inline Simd<uint32_t, 8>::Simd(Simd<bool, 8> v) : value(v.value) {}

// int32_t and uint32_t conversions (size 4)
inline Simd<int32_t, 4>::Simd(uint32_t v) : value(_mm_set1_epi32(v)) {}
inline Simd<uint32_t, 4>::Simd(int32_t v) : value(_mm_set1_epi32(v)) {}

// int32_t and uint32_t concatenation constructors (size 4 -> size 8)
inline Simd<int32_t, 8>::Simd(Simd<int32_t, 4> lo, Simd<int32_t, 4> hi)
  : value(_mm256_inserti128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1)) {}

inline Simd<uint32_t, 8>::Simd(Simd<uint32_t, 4> lo, Simd<uint32_t, 4> hi)
  : value(_mm256_inserti128_si256(_mm256_castsi128_si256(lo.value), hi.value, 1)) {}

// Type conversions between different element widths
// uint8 -> int32 (zero-extend)
inline Simd<int32_t, 8>::Simd(Simd<uint8_t, 8> v)
  : value(_mm256_cvtepu8_epi32(v.value)) {}

// uint8 -> uint32 (zero-extend)
inline Simd<uint32_t, 8>::Simd(Simd<uint8_t, 8> v)
  : value(_mm256_cvtepu8_epi32(v.value)) {}

// uint32 -> uint8 (truncate/pack)
inline Simd<uint8_t, 8>::Simd(Simd<uint32_t, 8> v) {
  // Pack 32-bit to 8-bit
  alignas(32) uint32_t tmp32[8];
  _mm256_store_si256((__m256i*)tmp32, v.value);
  alignas(16) uint8_t tmp8[16] = {0};
  for (int i = 0; i < 8; i++) {
    tmp8[i] = static_cast<uint8_t>(tmp32[i]);
  }
  value = _mm_loadl_epi64((const __m128i*)tmp8);
}

// uint8 -> uint16 (zero-extend using masks and shifts)
inline Simd<uint16_t, 8>::Simd(Simd<uint8_t, 8> v)
  : value(_mm_cvtepu8_epi16(v.value)) {}

// uint8 -> bool (extend to 32-bit masks)
inline Simd<bool, 8>::Simd(Simd<uint8_t, 8> v)
  : value(_mm256_cvtepi8_epi32(v.value)) {}

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
