#pragma once

#ifdef __SSE4_2__

#include <nmmintrin.h> // SSE4.2
#include <smmintrin.h> // SSE4.1
#include <cmath>
#include <stdint.h>

#include "mlx/backend/cpu/simd/base_simd.h"

namespace mlx::core::simd {

// SSE: 128-bit SIMD
// Float, double, and bool types defined here
// Integer types provided by sse_int_types.h (shared with AVX)
template <>
inline constexpr int max_size<float> = 4;
template <>
inline constexpr int max_size<double> = 2;

// ============================================================================
// Bool specializations (forward declare early for comparison operators)
// Note: These are 128-bit bool vectors for SSE (4x32-bit or 2x64-bit masks)
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
  Simd(bool v) : value(_mm_set1_epi64x(v ? -1LL : 0)) {}
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

// Bool reductions (operators defined above)
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

} // namespace mlx::core::simd

// Include integer types after bool types are defined (needed for comparisons)
#include "mlx/backend/cpu/simd/sse_int_types.h"

namespace mlx::core::simd {

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

// Conversion constructor definitions (after all structs are complete)
inline Simd<float, 4>::Simd(Simd<int32_t, 4> v) : value(_mm_cvtepi32_ps(v.value)) {}
inline Simd<float, 4>::Simd(Simd<uint32_t, 4> v) : value(_mm_cvtepi32_ps(v.value)) {}
inline Simd<float, 4>::Simd(Simd<bool, 4> v) : value(_mm_castsi128_ps(v.value)) {}
inline Simd<double, 2>::Simd(Simd<bool, 2> v) : value(_mm_castsi128_pd(v.value)) {}

} // namespace mlx::core::simd

#endif // __SSE4_2__
