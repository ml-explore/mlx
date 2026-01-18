#pragma once

#ifdef __SSE4_2__

#include <nmmintrin.h> // SSE4.2
#include <smmintrin.h> // SSE4.1
#include <cmath>
#include <stdint.h>

#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

// SSE: 128-bit SIMD
// Focus on float and double for Phase 1, let others use scalar fallback
template <>
inline constexpr int max_size<float> = 4;
template <>
inline constexpr int max_size<double> = 2;
// For now, only implement int32 as well since it's commonly used
template <>
inline constexpr int max_size<int32_t> = 4;
template <>
inline constexpr int max_size<uint32_t> = 4;
// Let other integer types use scalar fallback (max_size = 1 from base_simd.h)
// We can add these in later phases
// template <>
// inline constexpr int max_size<int64_t> = 2;
// template <>
// inline constexpr int max_size<uint64_t> = 2;
// template <>
// inline constexpr int max_size<int16_t> = 8;
// template <>
// inline constexpr int max_size<uint16_t> = 8;
// template <>
// inline constexpr int max_size<int8_t> = 16;
// template <>
// inline constexpr int max_size<uint8_t> = 16;

// ============================================================================
// Bool specializations (forward declare early for comparison operators)
// ============================================================================

template <>
struct Simd<bool, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(bool v) : value(_mm_set1_epi32(v ? -1 : 0)) {}
};

template <>
struct Simd<bool, 2> {
  static constexpr int size = 2;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(bool v) : value(_mm_set1_epi64x(v ? -1 : 0)) {}
};

// Bool operators (needed early for comparison operator implementations)
inline Simd<bool, 4> operator!(Simd<bool, 4> a) {
  return Simd<bool, 4>(_mm_xor_si128(a.value, _mm_set1_epi32(-1)));
}

inline Simd<bool, 2> operator!(Simd<bool, 2> a) {
  return Simd<bool, 2>(_mm_xor_si128(a.value, _mm_set1_epi64x(-1)));
}

inline Simd<bool, 4> operator||(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm_or_si128(a.value, b.value));
}

inline Simd<bool, 4> operator&&(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm_and_si128(a.value, b.value));
}

inline Simd<bool, 2> operator||(Simd<bool, 2> a, Simd<bool, 2> b) {
  return Simd<bool, 2>(_mm_or_si128(a.value, b.value));
}

inline Simd<bool, 2> operator&&(Simd<bool, 2> a, Simd<bool, 2> b) {
  return Simd<bool, 2>(_mm_and_si128(a.value, b.value));
}

// Bitwise operators for bool types (same as logical, but needed for generic code)
inline Simd<bool, 4> operator&(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm_and_si128(a.value, b.value));
}

inline Simd<bool, 4> operator|(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm_or_si128(a.value, b.value));
}

inline Simd<bool, 4> operator^(Simd<bool, 4> a, Simd<bool, 4> b) {
  return Simd<bool, 4>(_mm_xor_si128(a.value, b.value));
}

inline Simd<bool, 2> operator&(Simd<bool, 2> a, Simd<bool, 2> b) {
  return Simd<bool, 2>(_mm_and_si128(a.value, b.value));
}

inline Simd<bool, 2> operator|(Simd<bool, 2> a, Simd<bool, 2> b) {
  return Simd<bool, 2>(_mm_or_si128(a.value, b.value));
}

inline Simd<bool, 2> operator^(Simd<bool, 2> a, Simd<bool, 2> b) {
  return Simd<bool, 2>(_mm_xor_si128(a.value, b.value));
}

// Comparison operators for bool types
inline Simd<bool, 4> operator==(Simd<bool, 4> a, Simd<bool, 4> b) {
  // Bools are equal if XOR is zero (all bits match)
  return !Simd<bool, 4>(_mm_xor_si128(a.value, b.value));
}

inline Simd<bool, 4> operator!=(Simd<bool, 4> a, Simd<bool, 4> b) {
  return !(a == b);
}

inline Simd<bool, 2> operator==(Simd<bool, 2> a, Simd<bool, 2> b) {
  return !Simd<bool, 2>(_mm_xor_si128(a.value, b.value));
}

inline Simd<bool, 2> operator!=(Simd<bool, 2> a, Simd<bool, 2> b) {
  return !(a == b);
}

// Load/Store bool types (converting between 1-byte bool and 32/64-bit SIMD masks)
template <>
inline Simd<bool, 4> load<bool, 4>(const bool* ptr) {
  // Convert 4 bytes to 4x 32-bit masks
  return Simd<bool, 4>(_mm_set_epi32(
    ptr[3] ? -1 : 0,
    ptr[2] ? -1 : 0,
    ptr[1] ? -1 : 0,
    ptr[0] ? -1 : 0
  ));
}

template <>
inline void store<bool, 4>(bool* ptr, Simd<bool, 4> v) {
  // Convert 4x 32-bit masks to 4 bytes
  // Use movemask to extract the sign bits
  int mask = _mm_movemask_ps(_mm_castsi128_ps(v.value));
  ptr[0] = (mask & 1) != 0;
  ptr[1] = (mask & 2) != 0;
  ptr[2] = (mask & 4) != 0;
  ptr[3] = (mask & 8) != 0;
}

template <>
inline Simd<bool, 2> load<bool, 2>(const bool* ptr) {
  // Convert 2 bytes to 2x 64-bit masks
  return Simd<bool, 2>(_mm_set_epi64x(
    ptr[1] ? -1LL : 0,
    ptr[0] ? -1LL : 0
  ));
}

template <>
inline void store<bool, 2>(bool* ptr, Simd<bool, 2> v) {
  // Convert 2x 64-bit masks to 2 bytes
  // Use movemask to extract the sign bits
  int mask = _mm_movemask_pd(_mm_castsi128_pd(v.value));
  ptr[0] = (mask & 1) != 0;
  ptr[1] = (mask & 2) != 0;
}

// ============================================================================
// float32x4 (4 floats in 128 bits)
// ============================================================================

template <>
struct Simd<float, 4> {
  static constexpr int size = 4;
  __m128 value;

  Simd() : value(_mm_setzero_ps()) {}
  Simd(__m128 v) : value(v) {}
  Simd(float v) : value(_mm_set1_ps(v)) {}
  // Explicit conversions from other Simd types (defined after all structs)
  Simd(Simd<int32_t, 4> v);
  Simd(Simd<uint32_t, 4> v);
  Simd(Simd<bool, 4> v);

  template <typename U>
  Simd(Simd<U, 4> v);

  float operator[](int idx) const {
    alignas(16) float tmp[4];
    _mm_store_ps(tmp, value);
    return tmp[idx];
  }
};

// Load/Store float32x4
template <>
inline Simd<float, 4> load<float, 4>(const float* ptr) {
  return _mm_loadu_ps(ptr);
}

template <>
inline void store<float, 4>(float* ptr, Simd<float, 4> v) {
  _mm_storeu_ps(ptr, v.value);
}

// Arithmetic float32x4
inline Simd<float, 4> operator+(Simd<float, 4> a, Simd<float, 4> b) {
  return _mm_add_ps(a.value, b.value);
}
inline Simd<float, 4> operator-(Simd<float, 4> a, Simd<float, 4> b) {
  return _mm_sub_ps(a.value, b.value);
}
inline Simd<float, 4> operator*(Simd<float, 4> a, Simd<float, 4> b) {
  return _mm_mul_ps(a.value, b.value);
}
inline Simd<float, 4> operator/(Simd<float, 4> a, Simd<float, 4> b) {
  return _mm_div_ps(a.value, b.value);
}
inline Simd<float, 4> operator-(Simd<float, 4> a) {
  return _mm_sub_ps(_mm_setzero_ps(), a.value);
}

// Comparisons float32x4 (returns bool as int mask)
inline Simd<bool, 4> operator<(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmplt_ps(a.value, b.value)));
}
inline Simd<bool, 4> operator<=(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmple_ps(a.value, b.value)));
}
inline Simd<bool, 4> operator>(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmpgt_ps(a.value, b.value)));
}
inline Simd<bool, 4> operator>=(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmpge_ps(a.value, b.value)));
}
inline Simd<bool, 4> operator==(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmpeq_ps(a.value, b.value)));
}
inline Simd<bool, 4> operator!=(Simd<float, 4> a, Simd<float, 4> b) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmpneq_ps(a.value, b.value)));
}

// Logical NOT for float - returns bool mask where each element is true if input is zero
inline Simd<bool, 4> operator!(Simd<float, 4> a) {
  return a == Simd<float, 4>(0.0f);
}

// isnan for float - use unordered comparison (specifically for NaN detection)
inline Simd<bool, 4> isnan(Simd<float, 4> a) {
  return Simd<bool, 4>(_mm_castps_si128(_mm_cmpunord_ps(a.value, a.value)));
}

// Select (blend) float32x4 - defined early since it's used by maximum/minimum
inline Simd<float, 4> select(
    Simd<bool, 4> mask,
    Simd<float, 4> x,
    Simd<float, 4> y) {
  return _mm_blendv_ps(y.value, x.value, _mm_castsi128_ps(mask.value));
}

// Math functions float32x4
inline Simd<float, 4> abs(Simd<float, 4> a) {
  __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  return _mm_and_ps(a.value, mask);
}

inline Simd<float, 4> sqrt(Simd<float, 4> a) {
  return _mm_sqrt_ps(a.value);
}

inline Simd<float, 4> rsqrt(Simd<float, 4> a) {
  return _mm_rsqrt_ps(a.value);
}

inline Simd<float, 4> recip(Simd<float, 4> a) {
  return _mm_rcp_ps(a.value);
}

inline Simd<float, 4> floor(Simd<float, 4> a) {
  return _mm_floor_ps(a.value);
}

inline Simd<float, 4> ceil(Simd<float, 4> a) {
  return _mm_ceil_ps(a.value);
}

inline Simd<float, 4> rint(Simd<float, 4> a) {
  return _mm_round_ps(a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline Simd<float, 4> maximum(Simd<float, 4> a, Simd<float, 4> b) {
  auto out = Simd<float, 4>(_mm_max_ps(a.value, b.value));
  // NaN propagation: if either input is NaN, return that NaN
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<float, 4> minimum(Simd<float, 4> a, Simd<float, 4> b) {
  auto out = Simd<float, 4>(_mm_min_ps(a.value, b.value));
  // NaN propagation: if either input is NaN, return that NaN
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<float, 4> clamp(Simd<float, 4> v, Simd<float, 4> min_val, Simd<float, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

inline Simd<float, 4> atan2(Simd<float, 4> a, Simd<float, 4> b) {
  // atan2 needs to be computed per element using scalar fallback
  alignas(16) float tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_ps(tmp_a, a.value);
  _mm_store_ps(tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = std::atan2(tmp_a[i], tmp_b[i]);
  }
  return _mm_load_ps(tmp_r);
}

inline Simd<float, 4> pow(Simd<float, 4> base, Simd<float, 4> exp) {
  // pow needs to be computed per element using scalar fallback
  alignas(16) float tmp_base[4], tmp_exp[4], tmp_r[4];
  _mm_store_ps(tmp_base, base.value);
  _mm_store_ps(tmp_exp, exp.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = std::pow(tmp_base[i], tmp_exp[i]);
  }
  return _mm_load_ps(tmp_r);
}

// Transcendental functions - use scalar fallback for now
#define SSE_TRANSCENDENTAL_FLOAT(name, func)                    \
  inline Simd<float, 4> name(Simd<float, 4> a) {                \
    alignas(16) float tmp_a[4], tmp_r[4];                       \
    _mm_store_ps(tmp_a, a.value);                               \
    for (int i = 0; i < 4; i++) {                               \
      tmp_r[i] = std::func(tmp_a[i]);                           \
    }                                                           \
    return _mm_load_ps(tmp_r);                                  \
  }

SSE_TRANSCENDENTAL_FLOAT(exp, exp)
SSE_TRANSCENDENTAL_FLOAT(expm1, expm1)
SSE_TRANSCENDENTAL_FLOAT(log, log)
SSE_TRANSCENDENTAL_FLOAT(log1p, log1p)
SSE_TRANSCENDENTAL_FLOAT(log2, log2)
SSE_TRANSCENDENTAL_FLOAT(log10, log10)
SSE_TRANSCENDENTAL_FLOAT(sin, sin)
SSE_TRANSCENDENTAL_FLOAT(cos, cos)
SSE_TRANSCENDENTAL_FLOAT(tan, tan)
SSE_TRANSCENDENTAL_FLOAT(asin, asin)
SSE_TRANSCENDENTAL_FLOAT(acos, acos)
SSE_TRANSCENDENTAL_FLOAT(atan, atan)
SSE_TRANSCENDENTAL_FLOAT(sinh, sinh)
SSE_TRANSCENDENTAL_FLOAT(cosh, cosh)
SSE_TRANSCENDENTAL_FLOAT(tanh, tanh)
SSE_TRANSCENDENTAL_FLOAT(asinh, asinh)
SSE_TRANSCENDENTAL_FLOAT(acosh, acosh)
SSE_TRANSCENDENTAL_FLOAT(atanh, atanh)

inline Simd<float, 4> remainder(Simd<float, 4> a, Simd<float, 4> b) {
  alignas(16) float tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_ps(tmp_a, a.value);
  _mm_store_ps(tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = std::remainder(tmp_a[i], tmp_b[i]);
  }
  return _mm_load_ps(tmp_r);
}

// Logical operators for float32x4 (return float with bit pattern, not bool)
inline Simd<float, 4> operator&&(Simd<float, 4> a, Simd<float, 4> b) {
  __m128 zero = _mm_setzero_ps();
  __m128 mask_a = _mm_cmpneq_ps(a.value, zero);
  __m128 mask_b = _mm_cmpneq_ps(b.value, zero);
  return _mm_and_ps(mask_a, mask_b);
}

inline Simd<float, 4> operator||(Simd<float, 4> a, Simd<float, 4> b) {
  __m128 zero = _mm_setzero_ps();
  __m128 mask_a = _mm_cmpneq_ps(a.value, zero);
  __m128 mask_b = _mm_cmpneq_ps(b.value, zero);
  return _mm_or_ps(mask_a, mask_b);
}

// FMA (emulated for SSE4.2, will be native in AVX2)
inline Simd<float, 4> fma(Simd<float, 4> a, Simd<float, 4> b, Simd<float, 4> c) {
  return _mm_add_ps(_mm_mul_ps(a.value, b.value), c.value);
}

// Reductions float32x4
inline float sum(Simd<float, 4> v) {
  __m128 shuf = _mm_movehdup_ps(v.value);   // [1,1,3,3]
  __m128 sums = _mm_add_ps(v.value, shuf);  // [0+1,1+1,2+3,3+3]
  shuf = _mm_movehl_ps(shuf, sums);         // [2+3,3+3,...]
  sums = _mm_add_ss(sums, shuf);            // [0+1+2+3,...]
  return _mm_cvtss_f32(sums);
}

inline float max(Simd<float, 4> v) {
  __m128 shuf = _mm_shuffle_ps(v.value, v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 maxs = _mm_max_ps(v.value, shuf);
  shuf = _mm_movehl_ps(shuf, maxs);
  maxs = _mm_max_ss(maxs, shuf);
  return _mm_cvtss_f32(maxs);
}

inline float min(Simd<float, 4> v) {
  __m128 shuf = _mm_shuffle_ps(v.value, v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 mins = _mm_min_ps(v.value, shuf);
  shuf = _mm_movehl_ps(shuf, mins);
  mins = _mm_min_ss(mins, shuf);
  return _mm_cvtss_f32(mins);
}

inline float prod(Simd<float, 4> v) {
  __m128 shuf = _mm_shuffle_ps(v.value, v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128 prods = _mm_mul_ps(v.value, shuf);
  shuf = _mm_movehl_ps(shuf, prods);
  prods = _mm_mul_ss(prods, shuf);
  return _mm_cvtss_f32(prods);
}

// ============================================================================
// double64x2 (2 doubles in 128 bits)
// ============================================================================

template <>
struct Simd<double, 2> {
  static constexpr int size = 2;
  __m128d value;

  Simd() : value(_mm_setzero_pd()) {}
  Simd(__m128d v) : value(v) {}
  Simd(double v) : value(_mm_set1_pd(v)) {}
  // Explicit conversions from other Simd types (defined after all structs)
  Simd(Simd<bool, 2> v);

  template <typename U>
  Simd(Simd<U, 2> v);

  double operator[](int idx) const {
    alignas(16) double tmp[2];
    _mm_store_pd(tmp, value);
    return tmp[idx];
  }
};

// Load/Store double64x2
template <>
inline Simd<double, 2> load<double, 2>(const double* ptr) {
  return _mm_loadu_pd(ptr);
}

template <>
inline void store<double, 2>(double* ptr, Simd<double, 2> v) {
  _mm_storeu_pd(ptr, v.value);
}

// Arithmetic double64x2
inline Simd<double, 2> operator+(Simd<double, 2> a, Simd<double, 2> b) {
  return _mm_add_pd(a.value, b.value);
}
inline Simd<double, 2> operator-(Simd<double, 2> a, Simd<double, 2> b) {
  return _mm_sub_pd(a.value, b.value);
}
inline Simd<double, 2> operator*(Simd<double, 2> a, Simd<double, 2> b) {
  return _mm_mul_pd(a.value, b.value);
}
inline Simd<double, 2> operator/(Simd<double, 2> a, Simd<double, 2> b) {
  return _mm_div_pd(a.value, b.value);
}
inline Simd<double, 2> operator-(Simd<double, 2> a) {
  return _mm_sub_pd(_mm_setzero_pd(), a.value);
}

// Comparisons double64x2
inline Simd<bool, 2> operator<(Simd<double, 2> a, Simd<double, 2> b) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmplt_pd(a.value, b.value)));
}
inline Simd<bool, 2> operator<=(Simd<double, 2> a, Simd<double, 2> b) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmple_pd(a.value, b.value)));
}
inline Simd<bool, 2> operator>(Simd<double, 2> a, Simd<double, 2> b) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmpgt_pd(a.value, b.value)));
}
inline Simd<bool, 2> operator>=(Simd<double, 2> a, Simd<double, 2> b) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmpge_pd(a.value, b.value)));
}
inline Simd<bool, 2> operator==(Simd<double, 2> a, Simd<double, 2> b) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmpeq_pd(a.value, b.value)));
}
inline Simd<bool, 2> operator!=(Simd<double, 2> a, Simd<double, 2> b) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmpneq_pd(a.value, b.value)));
}

// Logical NOT for double - returns bool mask where each element is true if input is zero
inline Simd<bool, 2> operator!(Simd<double, 2> a) {
  return a == Simd<double, 2>(0.0);
}

// Select (blend) double64x2 - defined early since it's used by maximum/minimum
inline Simd<double, 2> select(
    Simd<bool, 2> mask,
    Simd<double, 2> x,
    Simd<double, 2> y) {
  return _mm_blendv_pd(y.value, x.value, _mm_castsi128_pd(mask.value));
}

// isnan for double - use unordered comparison (specifically for NaN detection)
inline Simd<bool, 2> isnan(Simd<double, 2> a) {
  return Simd<bool, 2>(_mm_castpd_si128(_mm_cmpunord_pd(a.value, a.value)));
}

// Math functions double64x2
inline Simd<double, 2> abs(Simd<double, 2> a) {
  __m128d mask = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFFLL));
  return _mm_and_pd(a.value, mask);
}

inline Simd<double, 2> sqrt(Simd<double, 2> a) {
  return _mm_sqrt_pd(a.value);
}

inline Simd<double, 2> rsqrt(Simd<double, 2> a) {
  // No SSE rsqrt for double, use div
  return Simd<double, 2>(1.0) / sqrt(a);
}

inline Simd<double, 2> recip(Simd<double, 2> a) {
  // No SSE recip for double, use div
  return Simd<double, 2>(1.0) / a;
}

inline Simd<double, 2> floor(Simd<double, 2> a) {
  return _mm_floor_pd(a.value);
}

inline Simd<double, 2> ceil(Simd<double, 2> a) {
  return _mm_ceil_pd(a.value);
}

inline Simd<double, 2> rint(Simd<double, 2> a) {
  return _mm_round_pd(a.value, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

inline Simd<double, 2> maximum(Simd<double, 2> a, Simd<double, 2> b) {
  auto out = Simd<double, 2>(_mm_max_pd(a.value, b.value));
  // NaN propagation: if either input is NaN, return that NaN
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 2> minimum(Simd<double, 2> a, Simd<double, 2> b) {
  auto out = Simd<double, 2>(_mm_min_pd(a.value, b.value));
  // NaN propagation: if either input is NaN, return that NaN
  out = select(isnan(b), b, select(isnan(a), a, out));
  return out;
}

inline Simd<double, 2> clamp(Simd<double, 2> v, Simd<double, 2> min_val, Simd<double, 2> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

inline Simd<double, 2> atan2(Simd<double, 2> a, Simd<double, 2> b) {
  // atan2 needs to be computed per element using scalar fallback
  alignas(16) double tmp_a[2], tmp_b[2], tmp_r[2];
  _mm_store_pd(tmp_a, a.value);
  _mm_store_pd(tmp_b, b.value);
  for (int i = 0; i < 2; i++) {
    tmp_r[i] = std::atan2(tmp_a[i], tmp_b[i]);
  }
  return _mm_load_pd(tmp_r);
}

inline Simd<double, 2> pow(Simd<double, 2> base, Simd<double, 2> exp) {
  // pow needs to be computed per element using scalar fallback
  alignas(16) double tmp_base[2], tmp_exp[2], tmp_r[2];
  _mm_store_pd(tmp_base, base.value);
  _mm_store_pd(tmp_exp, exp.value);
  for (int i = 0; i < 2; i++) {
    tmp_r[i] = std::pow(tmp_base[i], tmp_exp[i]);
  }
  return _mm_load_pd(tmp_r);
}

// Transcendental functions - use scalar fallback for now
#define SSE_TRANSCENDENTAL_DOUBLE(name, func)                   \
  inline Simd<double, 2> name(Simd<double, 2> a) {              \
    alignas(16) double tmp_a[2], tmp_r[2];                      \
    _mm_store_pd(tmp_a, a.value);                               \
    for (int i = 0; i < 2; i++) {                               \
      tmp_r[i] = std::func(tmp_a[i]);                           \
    }                                                           \
    return _mm_load_pd(tmp_r);                                  \
  }

SSE_TRANSCENDENTAL_DOUBLE(exp, exp)
SSE_TRANSCENDENTAL_DOUBLE(expm1, expm1)
SSE_TRANSCENDENTAL_DOUBLE(log, log)
SSE_TRANSCENDENTAL_DOUBLE(log1p, log1p)
SSE_TRANSCENDENTAL_DOUBLE(log2, log2)
SSE_TRANSCENDENTAL_DOUBLE(log10, log10)
SSE_TRANSCENDENTAL_DOUBLE(sin, sin)
SSE_TRANSCENDENTAL_DOUBLE(cos, cos)
SSE_TRANSCENDENTAL_DOUBLE(tan, tan)
SSE_TRANSCENDENTAL_DOUBLE(asin, asin)
SSE_TRANSCENDENTAL_DOUBLE(acos, acos)
SSE_TRANSCENDENTAL_DOUBLE(atan, atan)
SSE_TRANSCENDENTAL_DOUBLE(sinh, sinh)
SSE_TRANSCENDENTAL_DOUBLE(cosh, cosh)
SSE_TRANSCENDENTAL_DOUBLE(tanh, tanh)
SSE_TRANSCENDENTAL_DOUBLE(asinh, asinh)
SSE_TRANSCENDENTAL_DOUBLE(acosh, acosh)
SSE_TRANSCENDENTAL_DOUBLE(atanh, atanh)

inline Simd<double, 2> remainder(Simd<double, 2> a, Simd<double, 2> b) {
  alignas(16) double tmp_a[2], tmp_b[2], tmp_r[2];
  _mm_store_pd(tmp_a, a.value);
  _mm_store_pd(tmp_b, b.value);
  for (int i = 0; i < 2; i++) {
    tmp_r[i] = std::remainder(tmp_a[i], tmp_b[i]);
  }
  return _mm_load_pd(tmp_r);
}

// Logical operators for double64x2 (return double with bit pattern, not bool)
inline Simd<double, 2> operator&&(Simd<double, 2> a, Simd<double, 2> b) {
  __m128d zero = _mm_setzero_pd();
  __m128d mask_a = _mm_cmpneq_pd(a.value, zero);
  __m128d mask_b = _mm_cmpneq_pd(b.value, zero);
  return _mm_and_pd(mask_a, mask_b);
}

inline Simd<double, 2> operator||(Simd<double, 2> a, Simd<double, 2> b) {
  __m128d zero = _mm_setzero_pd();
  __m128d mask_a = _mm_cmpneq_pd(a.value, zero);
  __m128d mask_b = _mm_cmpneq_pd(b.value, zero);
  return _mm_or_pd(mask_a, mask_b);
}

// FMA (emulated)
inline Simd<double, 2> fma(Simd<double, 2> a, Simd<double, 2> b, Simd<double, 2> c) {
  return _mm_add_pd(_mm_mul_pd(a.value, b.value), c.value);
}

// Reductions double64x2
inline double sum(Simd<double, 2> v) {
  __m128d shuf = _mm_shuffle_pd(v.value, v.value, 1);
  __m128d sums = _mm_add_sd(v.value, shuf);
  return _mm_cvtsd_f64(sums);
}

inline double max(Simd<double, 2> v) {
  __m128d shuf = _mm_shuffle_pd(v.value, v.value, 1);
  __m128d maxs = _mm_max_sd(v.value, shuf);
  return _mm_cvtsd_f64(maxs);
}

inline double min(Simd<double, 2> v) {
  __m128d shuf = _mm_shuffle_pd(v.value, v.value, 1);
  __m128d mins = _mm_min_sd(v.value, shuf);
  return _mm_cvtsd_f64(mins);
}

inline double prod(Simd<double, 2> v) {
  __m128d shuf = _mm_shuffle_pd(v.value, v.value, 1);
  __m128d prods = _mm_mul_sd(v.value, shuf);
  return _mm_cvtsd_f64(prods);
}

// ============================================================================
// int32x4 (4 int32s in 128 bits)
// ============================================================================

template <>
struct Simd<int32_t, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(int32_t v) : value(_mm_set1_epi32(v)) {}
  // Allow conversion from other Simd<U, 4> where U is integer-like
  template <typename U>
  Simd(Simd<U, 4> v) : value(v.value) {}

  int32_t operator[](int idx) const {
    alignas(16) int32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

// uint32x4 (same as int32x4, just different signedness)
template <>
struct Simd<uint32_t, 4> {
  static constexpr int size = 4;
  __m128i value;

  Simd() : value(_mm_setzero_si128()) {}
  Simd(__m128i v) : value(v) {}
  Simd(uint32_t v) : value(_mm_set1_epi32(v)) {}
  // Allow conversion from other Simd<U, 4> where U is integer-like
  template <typename U>
  Simd(Simd<U, 4> v) : value(v.value) {}

  uint32_t operator[](int idx) const {
    alignas(16) uint32_t tmp[4];
    _mm_store_si128((__m128i*)tmp, value);
    return tmp[idx];
  }
};

// Conversion constructor definitions (after all structs are complete)
inline Simd<float, 4>::Simd(Simd<int32_t, 4> v) : value(_mm_cvtepi32_ps(v.value)) {}
inline Simd<float, 4>::Simd(Simd<uint32_t, 4> v) : value(_mm_cvtepi32_ps(v.value)) {}
inline Simd<float, 4>::Simd(Simd<bool, 4> v) : value(_mm_castsi128_ps(v.value)) {}
inline Simd<double, 2>::Simd(Simd<bool, 2> v) : value(_mm_castsi128_pd(v.value)) {}

// Load/Store int32x4
template <>
inline Simd<int32_t, 4> load<int32_t, 4>(const int32_t* ptr) {
  return _mm_loadu_si128((__m128i*)ptr);
}

template <>
inline void store<int32_t, 4>(int32_t* ptr, Simd<int32_t, 4> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// Load/Store uint32x4
template <>
inline Simd<uint32_t, 4> load<uint32_t, 4>(const uint32_t* ptr) {
  return _mm_loadu_si128((__m128i*)ptr);
}

template <>
inline void store<uint32_t, 4>(uint32_t* ptr, Simd<uint32_t, 4> v) {
  _mm_storeu_si128((__m128i*)ptr, v.value);
}

// Arithmetic uint32x4 (same as int32 but unsigned)
inline Simd<uint32_t, 4> operator+(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_add_epi32(a.value, b.value);
}
inline Simd<uint32_t, 4> operator-(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_sub_epi32(a.value, b.value);
}
inline Simd<uint32_t, 4> operator*(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_mullo_epi32(a.value, b.value);
}
// Unary minus for uint32
inline Simd<uint32_t, 4> operator-(Simd<uint32_t, 4> a) {
  return _mm_sub_epi32(_mm_setzero_si128(), a.value);
}
inline Simd<uint32_t, 4> operator/(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  // No SIMD division, use scalar
  alignas(16) uint32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Bitwise uint32x4
inline Simd<uint32_t, 4> operator&(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_and_si128(a.value, b.value);
}
inline Simd<uint32_t, 4> operator|(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_or_si128(a.value, b.value);
}
inline Simd<uint32_t, 4> operator^(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_xor_si128(a.value, b.value);
}
inline Simd<uint32_t, 4> operator~(Simd<uint32_t, 4> a) {
  return _mm_xor_si128(a.value, _mm_set1_epi32(-1));
}

// Shifts uint32x4
inline Simd<uint32_t, 4> operator<<(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  alignas(16) uint32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] << tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline Simd<uint32_t, 4> operator>>(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  alignas(16) uint32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] >> tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Additional uint32 operations (use scalar fallback or cast to int32)
inline Simd<uint32_t, 4> maximum(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_max_epu32(a.value, b.value);
}

inline Simd<uint32_t, 4> minimum(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return _mm_min_epu32(a.value, b.value);
}

inline Simd<uint32_t, 4> clamp(Simd<uint32_t, 4> v, Simd<uint32_t, 4> min_val, Simd<uint32_t, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

// Comparisons for uint32 (defined early since they're used by other operators)
inline Simd<bool, 4> operator==(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmpeq_epi32(a.value, b.value));
}

inline Simd<bool, 4> operator!=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return !(a == b);
}

// Logical NOT for uint32 - returns bool mask
inline Simd<bool, 4> operator!(Simd<uint32_t, 4> a) {
  return a == Simd<uint32_t, 4>(0);
}

// isnan for uint32 - integers are never NaN
inline Simd<bool, 4> isnan(Simd<uint32_t, 4> a) {
  return Simd<bool, 4>(false);
}

inline Simd<uint32_t, 4> operator&&(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  auto mask_a = (a != Simd<uint32_t, 4>(0));
  auto mask_b = (b != Simd<uint32_t, 4>(0));
  return Simd<uint32_t, 4>(_mm_and_si128(mask_a.value, mask_b.value));
}

inline Simd<uint32_t, 4> operator||(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  auto mask_a = (a != Simd<uint32_t, 4>(0));
  auto mask_b = (b != Simd<uint32_t, 4>(0));
  return Simd<uint32_t, 4>(_mm_or_si128(mask_a.value, mask_b.value));
}

// Unsigned comparisons - SSE4.1 has unsigned min/max which we can use
inline Simd<bool, 4> operator<(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  // a < b  <==>  min(a,b) == a && a != b
  auto minn = _mm_min_epu32(a.value, b.value);
  return Simd<bool, 4>(_mm_andnot_si128(_mm_cmpeq_epi32(a.value, b.value), _mm_cmpeq_epi32(minn, a.value)));
}

inline Simd<bool, 4> operator>(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return b < a;
}

inline Simd<bool, 4> operator<=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return !(a > b);
}

inline Simd<bool, 4> operator>=(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  return !(a < b);
}

inline Simd<uint32_t, 4> remainder(Simd<uint32_t, 4> a, Simd<uint32_t, 4> b) {
  // For unsigned, remainder is simpler
  return a - b * (a / b);
}

inline Simd<uint32_t, 4> pow(Simd<uint32_t, 4> base, Simd<uint32_t, 4> exp) {
  alignas(16) uint32_t tmp_base[4], tmp_exp[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_base, base.value);
  _mm_store_si128((__m128i*)tmp_exp, exp.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = 1;
    for (uint32_t j = 0; j < tmp_exp[i]; j++) {
      tmp_r[i] *= tmp_base[i];
    }
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Select (blend) for uint32x4
inline Simd<uint32_t, 4> select(
    Simd<bool, 4> mask,
    Simd<uint32_t, 4> x,
    Simd<uint32_t, 4> y) {
  return _mm_blendv_epi8(y.value, x.value, mask.value);
}

// Reductions uint32x4
inline uint32_t sum(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sums = _mm_add_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));
  sums = _mm_add_epi32(sums, shuf);
  return _mm_cvtsi128_si32(sums);
}

inline uint32_t max(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i maxs = _mm_max_epu32(v.value, shuf);
  shuf = _mm_shuffle_epi32(maxs, _MM_SHUFFLE(1, 0, 3, 2));
  maxs = _mm_max_epu32(maxs, shuf);
  return _mm_cvtsi128_si32(maxs);
}

inline uint32_t min(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i mins = _mm_min_epu32(v.value, shuf);
  shuf = _mm_shuffle_epi32(mins, _MM_SHUFFLE(1, 0, 3, 2));
  mins = _mm_min_epu32(mins, shuf);
  return _mm_cvtsi128_si32(mins);
}

inline uint32_t prod(Simd<uint32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i prods = _mm_mullo_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(prods, _MM_SHUFFLE(1, 0, 3, 2));
  prods = _mm_mullo_epi32(prods, shuf);
  return _mm_cvtsi128_si32(prods);
}

// Arithmetic int32x4
inline Simd<int32_t, 4> operator+(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_add_epi32(a.value, b.value);
}
inline Simd<int32_t, 4> operator-(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_sub_epi32(a.value, b.value);
}
inline Simd<int32_t, 4> operator*(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_mullo_epi32(a.value, b.value);
}
inline Simd<int32_t, 4> operator/(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  // SSE doesn't have integer division, use scalar fallback
  alignas(16) int32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] / tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}
inline Simd<int32_t, 4> operator-(Simd<int32_t, 4> a) {
  return _mm_sub_epi32(_mm_setzero_si128(), a.value);
}

// Bitwise int32x4
inline Simd<int32_t, 4> operator&(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_and_si128(a.value, b.value);
}
inline Simd<int32_t, 4> operator|(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_or_si128(a.value, b.value);
}
inline Simd<int32_t, 4> operator^(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_xor_si128(a.value, b.value);
}
inline Simd<int32_t, 4> operator~(Simd<int32_t, 4> a) {
  return _mm_xor_si128(a.value, _mm_set1_epi32(-1));
}

// Shifts int32x4 (with int scalar shift amount)
inline Simd<int32_t, 4> operator<<(Simd<int32_t, 4> a, int bits) {
  return _mm_slli_epi32(a.value, bits);
}
inline Simd<int32_t, 4> operator>>(Simd<int32_t, 4> a, int bits) {
  return _mm_srai_epi32(a.value, bits);
}

// Shifts int32x4 (with vector shift amount - element-wise)
inline Simd<int32_t, 4> operator<<(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  // SSE4.2 doesn't have variable shift per-element, need to do it manually
  alignas(16) int32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] << tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

inline Simd<int32_t, 4> operator>>(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  // SSE4.2 doesn't have variable shift per-element, need to do it manually
  alignas(16) int32_t tmp_a[4], tmp_b[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_a, a.value);
  _mm_store_si128((__m128i*)tmp_b, b.value);
  for (int i = 0; i < 4; i++) {
    tmp_r[i] = tmp_a[i] >> tmp_b[i];
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Comparisons int32x4
inline Simd<bool, 4> operator<(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmplt_epi32(a.value, b.value));
}
inline Simd<bool, 4> operator>(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmpgt_epi32(a.value, b.value));
}
inline Simd<bool, 4> operator<=(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return !(a > b);
}
inline Simd<bool, 4> operator>=(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return !(a < b);
}
inline Simd<bool, 4> operator==(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return Simd<bool, 4>(_mm_cmpeq_epi32(a.value, b.value));
}
inline Simd<bool, 4> operator!=(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return !(a == b);
}

// Logical NOT for int32 - returns bool mask where each element is true if input is zero
inline Simd<bool, 4> operator!(Simd<int32_t, 4> a) {
  return a == Simd<int32_t, 4>(0);
}

// isnan for int32 - integers are never NaN
inline Simd<bool, 4> isnan(Simd<int32_t, 4> a) {
  return Simd<bool, 4>(false);
}

inline Simd<int32_t, 4> maximum(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_max_epi32(a.value, b.value);
}

inline Simd<int32_t, 4> minimum(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  return _mm_min_epi32(a.value, b.value);
}

inline Simd<int32_t, 4> abs(Simd<int32_t, 4> a) {
  return _mm_abs_epi32(a.value);
}

inline Simd<int32_t, 4> clamp(Simd<int32_t, 4> v, Simd<int32_t, 4> min_val, Simd<int32_t, 4> max_val) {
  return minimum(maximum(v, min_val), max_val);
}

// Logical operators for int32x4 (return int with bit pattern, not bool)
inline Simd<int32_t, 4> operator&&(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  auto mask_a = (a != Simd<int32_t, 4>(0));
  auto mask_b = (b != Simd<int32_t, 4>(0));
  return Simd<int32_t, 4>(_mm_and_si128(mask_a.value, mask_b.value));
}

inline Simd<int32_t, 4> operator||(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  auto mask_a = (a != Simd<int32_t, 4>(0));
  auto mask_b = (b != Simd<int32_t, 4>(0));
  return Simd<int32_t, 4>(_mm_or_si128(mask_a.value, mask_b.value));
}

// Select (blend) int32x4 - needed for remainder
inline Simd<int32_t, 4> select(
    Simd<bool, 4> mask,
    Simd<int32_t, 4> x,
    Simd<int32_t, 4> y) {
  return _mm_blendv_epi8(y.value, x.value, mask.value);
}

// remainder for int32x4
inline Simd<int32_t, 4> remainder(Simd<int32_t, 4> a, Simd<int32_t, 4> b) {
  auto r = a - b * (a / b);
  auto mask = (r != Simd<int32_t, 4>(0)) && ((r < Simd<int32_t, 4>(0)) != (b < Simd<int32_t, 4>(0)));
  return select(mask, r + b, r);
}

// pow for int32x4 (using scalar fallback)
inline Simd<int32_t, 4> pow(Simd<int32_t, 4> base, Simd<int32_t, 4> exp) {
  alignas(16) int32_t tmp_base[4], tmp_exp[4], tmp_r[4];
  _mm_store_si128((__m128i*)tmp_base, base.value);
  _mm_store_si128((__m128i*)tmp_exp, exp.value);
  for (int i = 0; i < 4; i++) {
    // Integer power: raise base to exp
    if (tmp_exp[i] < 0) {
      tmp_r[i] = 0;  // Undefined for integers
    } else {
      tmp_r[i] = 1;
      for (int32_t j = 0; j < tmp_exp[i]; j++) {
        tmp_r[i] *= tmp_base[i];
      }
    }
  }
  return _mm_load_si128((__m128i*)tmp_r);
}

// Reductions int32x4
inline int32_t sum(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sums = _mm_add_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(1, 0, 3, 2));
  sums = _mm_add_epi32(sums, shuf);
  return _mm_cvtsi128_si32(sums);
}

inline int32_t max(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i maxs = _mm_max_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(maxs, _MM_SHUFFLE(1, 0, 3, 2));
  maxs = _mm_max_epi32(maxs, shuf);
  return _mm_cvtsi128_si32(maxs);
}

inline int32_t min(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i mins = _mm_min_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(mins, _MM_SHUFFLE(1, 0, 3, 2));
  mins = _mm_min_epi32(mins, shuf);
  return _mm_cvtsi128_si32(mins);
}

inline int32_t prod(Simd<int32_t, 4> v) {
  __m128i shuf = _mm_shuffle_epi32(v.value, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i prods = _mm_mullo_epi32(v.value, shuf);
  shuf = _mm_shuffle_epi32(prods, _MM_SHUFFLE(1, 0, 3, 2));
  prods = _mm_mullo_epi32(prods, shuf);
  return _mm_cvtsi128_si32(prods);
}

// ============================================================================
// Bool reductions (operators defined earlier)
inline bool all(Simd<bool, 4> x) {
  return _mm_movemask_ps(_mm_castsi128_ps(x.value)) == 0xF;
}

inline bool any(Simd<bool, 4> x) {
  return _mm_movemask_ps(_mm_castsi128_ps(x.value)) != 0;
}

inline bool all(Simd<bool, 2> x) {
  return _mm_movemask_pd(_mm_castsi128_pd(x.value)) == 0x3;
}

inline bool any(Simd<bool, 2> x) {
  return _mm_movemask_pd(_mm_castsi128_pd(x.value)) != 0;
}

// ============================================================================
// clz (count leading zeros) - use scalar fallback
// ============================================================================

inline Simd<uint32_t, 4> clz(Simd<uint32_t, 4> x) {
  alignas(16) uint32_t tmp[4], res[4];
  _mm_store_si128((__m128i*)tmp, x.value);
  for (int i = 0; i < 4; i++) {
    res[i] = tmp[i] == 0 ? 32 : __builtin_clz(tmp[i]);
  }
  return _mm_load_si128((__m128i*)res);
}

inline Simd<int32_t, 4> clz(Simd<int32_t, 4> x) {
  // Reinterpret as unsigned for clz
  return Simd<int32_t, 4>(clz(Simd<uint32_t, 4>(x.value)).value);
}

} // namespace mlx::core::simd

#endif // __SSE4_2__
